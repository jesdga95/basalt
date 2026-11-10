//! PostgreSQL integration for the context registry. Skipped unless
//! BASALT_TEST_DATABASE_URL points at a server where a throwaway database may be created.
use basalt_context_registry::*;
use sqlx::{postgres::{PgConnectOptions, PgPoolOptions}, ConnectOptions, PgPool};
use std::{str::FromStr, time::Duration};
use uuid::Uuid;

type R<T> = Result<T, RegistryError>;

fn conflict_of<T: std::fmt::Debug>(r: R<T>) -> Conflict {
    match r {
        Err(RegistryError::Conflict(c)) => c,
        other => panic!("expected conflict, got {other:?}"),
    }
}
fn not_found<T: std::fmt::Debug>(r: R<T>) {
    assert!(matches!(r, Err(RegistryError::NotFound)), "expected NotFound, got {r:?}");
}
fn req(event: Event, v: i64, lease: Option<i64>) -> TransitionRequest {
    TransitionRequest { event, expected_version: v, lease_token: lease, patch: Patch::default() }
}

struct Seed { owner: Uuid, thread: Uuid, branch: Uuid, messages: Vec<Uuid> }

async fn seed(db: &PgPool, n: usize) -> Seed {
    let (owner, thread, branch) = (Uuid::new_v4(), Uuid::new_v4(), Uuid::new_v4());
    let mut tx = db.begin().await.unwrap();
    sqlx::query("INSERT INTO thread_owners(id,workspace_id) VALUES($1,$2)")
        .bind(owner).bind(Uuid::new_v4()).execute(&mut *tx).await.unwrap();
    sqlx::query("INSERT INTO threads(id,owner_id,model_id,default_branch_id) VALUES($1,$2,'m',$3)")
        .bind(thread).bind(owner).bind(branch).execute(&mut *tx).await.unwrap();
    sqlx::query("INSERT INTO branches(id,thread_id) VALUES($1,$2)")
        .bind(branch).bind(thread).execute(&mut *tx).await.unwrap();
    let mut messages = Vec::new();
    let mut parent: Option<Uuid> = None;
    for i in 0..n {
        let id = Uuid::new_v4();
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        sqlx::query("INSERT INTO messages(id,thread_id,parent_id,role,content,content_hash)
                     VALUES($1,$2,$3,$4,'\"x\"'::jsonb,$5)")
            .bind(id).bind(thread).bind(parent).bind(role)
            .bind(vec![i as u8 + 1; 32]).execute(&mut *tx).await.unwrap();
        parent = Some(id);
        messages.push(id);
    }
    if let Some(head) = parent {
        sqlx::query("UPDATE branches SET head_message_id=$1 WHERE id=$2")
            .bind(head).bind(branch).execute(&mut *tx).await.unwrap();
    }
    tx.commit().await.unwrap();
    Seed { owner, thread, branch, messages }
}

async fn s_create_and_isolation(db: &PgPool) {
    let a = seed(db, 4).await;
    let b = seed(db, 2).await;
    let full = create_context(db, a.owner, a.thread, None, None).await.unwrap();
    assert_eq!((full.state, full.tier, full.version, full.validated), (State::Recorded, Tier::None, 0, false));
    assert_eq!(full.head_message_id, Some(a.messages[3]));
    assert_eq!(full.branch_id, a.branch);
    let cut = create_context(db, a.owner, a.thread, Some(a.branch), Some(a.messages[1])).await.unwrap();
    assert_ne!(cut.prefix_hash, full.prefix_hash);
    assert_eq!(cut.model_fingerprint, full.model_fingerprint);
    let again = create_context(db, a.owner, a.thread, None, Some(a.messages[1])).await.unwrap();
    assert_eq!(again.prefix_hash, cut.prefix_hash, "same prefix -> same identity");
    assert_eq!(again.id, cut.id, "one context per transcript head: a repeated create returns the live one");
    // cross-tenant and foreign message ids look like missing resources
    not_found(create_context(db, b.owner, a.thread, None, None).await);
    not_found(create_context(db, a.owner, a.thread, None, Some(b.messages[0])).await);
    not_found(get_context(db, b.owner, full.id).await);
    not_found(tombstone(db, b.owner, full.id).await);
    not_found(acquire_lease(db, b.owner, full.id, None).await);
    not_found(transition(db, b.owner, full.id, req(Event::Materialize, 0, Some(1))).await);
    not_found(fork_context(db, b.owner, full.id, 0, "k").await.map(|_| ()));
    // empty thread has a valid (empty) prefix; off-branch upto is invalid
    let empty = seed(db, 0).await;
    let e = create_context(db, empty.owner, empty.thread, None, None).await.unwrap();
    assert_eq!(e.head_message_id, None);
    let other_branch = Uuid::new_v4();
    sqlx::query("INSERT INTO branches(id,thread_id,head_message_id) VALUES($1,$2,$3)")
        .bind(other_branch).bind(a.thread).bind(a.messages[0]).execute(db).await.unwrap();
    assert!(matches!(
        create_context(db, a.owner, a.thread, Some(other_branch), Some(a.messages[2])).await,
        Err(RegistryError::Invalid(_))));
    // database-level isolation: owner B cannot point a context at owner A's thread
    let bad = sqlx::query("INSERT INTO contexts(id,owner_id,thread_id,branch_id,model_fingerprint,prefix_hash)
                           VALUES($1,$2,$3,$4,$5,$5)")
        .bind(Uuid::new_v4()).bind(b.owner).bind(a.thread).bind(a.branch).bind(vec![0u8; 32])
        .execute(db).await;
    assert!(bad.is_err(), "owner/thread FK must reject cross-tenant rows");
}

async fn s_fork_idempotency(db: &PgPool) {
    let s = seed(db, 2).await;
    let p = create_context(db, s.owner, s.thread, None, None).await.unwrap();
    let (c1, replay1) = fork_context(db, s.owner, p.id, 0, "key-1").await.unwrap();
    assert!(!replay1);
    assert_eq!((c1.parent_id, c1.state, c1.version), (Some(p.id), State::Recorded, 0));
    assert_eq!((c1.prefix_hash, c1.head_message_id), (p.prefix_hash, p.head_message_id));
    let (c2, replay2) = fork_context(db, s.owner, p.id, 0, "key-1").await.unwrap();
    assert!(replay2);
    assert_eq!(c2.id, c1.id);
    assert_eq!(conflict_of(fork_context(db, s.owner, p.id, 5, "key-1").await), Conflict::IdempotencyPayload);
    assert_eq!(conflict_of(fork_context(db, s.owner, p.id, 5, "key-2").await), Conflict::StaleVersion);
    assert!(matches!(fork_context(db, s.owner, p.id, 0, "").await, Err(RegistryError::Invalid(_))));
    // parent version is not bumped by forking
    assert_eq!(get_context(db, s.owner, p.id).await.unwrap().context.version, 0);
    // concurrent retries of one key converge on one child
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let (db, owner, pid) = (db.clone(), s.owner, p.id);
        tasks.push(tokio::spawn(async move { fork_context(&db, owner, pid, 0, "race").await.unwrap() }));
    }
    let mut ids = Vec::new();
    let mut fresh = 0;
    for t in tasks { let (c, replayed) = t.await.unwrap(); ids.push(c.id); fresh += (!replayed) as u32; }
    ids.dedup();
    assert_eq!(ids.len(), 1, "one key -> one child");
    assert_eq!(fresh, 1);
    // lineage / children / refs
    let (grand, _) = fork_context(db, s.owner, c1.id, 0, "g").await.unwrap();
    let v = get_context(db, s.owner, grand.id).await.unwrap();
    assert_eq!(v.lineage.iter().map(|l| l.id).collect::<Vec<_>>(), vec![p.id, c1.id]);
    let pv = get_context(db, s.owner, p.id).await.unwrap();
    assert_eq!(pv.children.len(), 2);
    assert_eq!(pv.child_refs, 2);
    assert!(pv.lineage.is_empty());
}

async fn s_lifecycle_and_leases(db: &PgPool) {
    let s = seed(db, 2).await;
    let c = create_context(db, s.owner, s.thread, None, None).await.unwrap();
    // invalid lease ttl
    assert!(matches!(acquire_lease(db, s.owner, c.id, Some(10)).await, Err(RegistryError::Invalid(_))));
    let (_, l1, src) = acquire_lease(db, s.owner, c.id, None).await.unwrap();
    assert_eq!(src, "replay");
    assert_eq!(conflict_of(acquire_lease(db, s.owner, c.id, None).await), Conflict::LeaseHeld);
    // materialize needs the exact lease token
    assert_eq!(conflict_of(transition(db, s.owner, c.id, req(Event::Materialize, 0, None)).await), Conflict::LeaseRequired);
    assert_eq!(conflict_of(transition(db, s.owner, c.id, req(Event::Materialize, 0, Some(l1.token + 1))).await), Conflict::LeaseInvalid);
    assert_eq!(conflict_of(transition(db, s.owner, c.id, req(Event::Materialize, 9, Some(l1.token))).await), Conflict::StaleVersion);
    let m = transition(db, s.owner, c.id, req(Event::Materialize, 0, Some(l1.token))).await.unwrap();
    assert_eq!((m.state, m.tier, m.version), (State::Materializing, Tier::None, 1));
    // illegal event in this state
    assert!(matches!(conflict_of(transition(db, s.owner, c.id, req(Event::Cool, 1, None)).await),
        Conflict::InvalidTransition(_)));
    assert!(matches!(conflict_of(transition(db, s.owner, c.id, req(Event::Record, 1, None)).await),
        Conflict::InvalidTransition(_)));
    let epoch = Uuid::new_v4();
    let mut ready = req(Event::Ready, 1, Some(l1.token));
    ready.patch = Patch { tier: Some(Tier::Hot), locator: Some("gpu:slot".into()), validated: Some(true),
        model_fingerprint: Some([9; 32]), prefix_hash: Some([8; 32]), prefix_tokens: Some(1234),
        engine_epoch: Some(epoch) };
    let h = transition(db, s.owner, c.id, ready).await.unwrap();
    assert_eq!((h.state, h.tier, h.version, h.validated, h.prefix_tokens), (State::Hot, Tier::Hot, 2, true, 1234));
    assert_eq!((h.model_fingerprint, h.prefix_hash, h.engine_epoch), ([9; 32], [8; 32], Some(epoch)));
    // eviction needs zero live leases
    assert_eq!(conflict_of(transition(db, s.owner, c.id, req(Event::Cool, 2, None)).await), Conflict::LeaseHeld);
    assert_eq!(conflict_of(acquire_lease(db, s.owner, c.id, None).await), Conflict::LeaseHeld);
    release_lease(db, s.owner, c.id, l1.token).await.unwrap();
    release_lease(db, s.owner, c.id, l1.token).await.unwrap(); // idempotent
    assert_eq!(conflict_of(release_lease(db, s.owner, c.id, l1.token + 99).await), Conflict::LeaseInvalid);
    let mut wrong_tier = req(Event::Cool, 2, None);
    wrong_tier.patch.tier = Some(Tier::Cold);
    assert!(matches!(transition(db, s.owner, c.id, wrong_tier).await, Err(RegistryError::Invalid(_))));
    let cooling = transition(db, s.owner, c.id, req(Event::Cool, 2, None)).await.unwrap();
    assert_eq!(cooling.state, State::Cooling);
    assert_eq!(conflict_of(acquire_lease(db, s.owner, c.id, None).await), Conflict::Busy);
    let warm = transition(db, s.owner, c.id, req(Event::Warm, 3, None)).await.unwrap();
    assert_eq!((warm.state, warm.tier), (State::Warm, Tier::Warm));
    let mut chill = req(Event::Chill, 4, None);
    chill.patch = Patch { locator: Some("/snap/c1".into()), validated: Some(true), ..Default::default() };
    let cold = transition(db, s.owner, c.id, chill).await.unwrap();
    assert_eq!((cold.state, cold.tier, cold.validated, cold.locator.as_deref()),
        (State::Cold, Tier::Cold, true, Some("/snap/c1")));
    // resume from cold -> fenced restore -> fingerprint mismatch is refused
    let (_, l2, src2) = acquire_lease(db, s.owner, c.id, None).await.unwrap();
    assert_eq!(src2, "cold");
    assert!(l2.token > l1.token, "fencing tokens increase");
    let r = transition(db, s.owner, c.id, req(Event::Restore, 5, Some(l2.token))).await.unwrap();
    assert_eq!((r.state, r.tier), (State::Restoring, Tier::Cold));
    let mut bad = req(Event::Ready, 6, Some(l2.token));
    bad.patch.model_fingerprint = Some([1; 32]);
    assert_eq!(conflict_of(transition(db, s.owner, c.id, bad).await), Conflict::FingerprintMismatch);
    let mut good = req(Event::Ready, 6, Some(l2.token));
    good.patch.model_fingerprint = Some([9; 32]);
    assert_eq!(transition(db, s.owner, c.id, good).await.unwrap().state, State::Hot);
    // concurrent CAS: exactly one of N writers wins
    release_lease(db, s.owner, c.id, l2.token).await.unwrap();
    let mut tasks = Vec::new();
    for _ in 0..6 {
        let (db, owner, id) = (db.clone(), s.owner, c.id);
        tasks.push(tokio::spawn(async move { transition(&db, owner, id, req(Event::Cool, 7, None)).await }));
    }
    let mut wins = 0;
    for t in tasks { if t.await.unwrap().is_ok() { wins += 1; } }
    assert_eq!(wins, 1);
    // DB guards: tier/state mismatch and untombstoned delete are rejected by the database
    assert!(sqlx::query("UPDATE contexts SET tier='cold' WHERE id=$1").bind(c.id).execute(db).await.is_err());
    assert!(sqlx::query("DELETE FROM contexts WHERE id=$1").bind(c.id).execute(db).await.is_err());
    assert!(sqlx::query("UPDATE contexts SET version=0 WHERE id=$1").bind(c.id).execute(db).await.is_err());
    assert!(sqlx::query("UPDATE contexts SET thread_id=$2 WHERE id=$1")
        .bind(c.id).bind(Uuid::new_v4()).execute(db).await.is_err());
}

async fn s_expiry_and_reconcile(db: &PgPool) {
    let s = seed(db, 2).await;
    // expired writer is fenced and its orphaned MATERIALIZING context is rebuilt
    let c = create_context(db, s.owner, s.thread, None, None).await.unwrap();
    let (_, l, _) = acquire_lease(db, s.owner, c.id, Some(1_000)).await.unwrap();
    transition(db, s.owner, c.id, req(Event::Materialize, 0, Some(l.token))).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1_300)).await;
    assert_eq!(conflict_of(transition(db, s.owner, c.id, req(Event::Ready, 1, Some(l.token))).await), Conflict::LeaseInvalid);
    let rep = reconcile(db, None, 100).await.unwrap();
    assert!(rep.leases_expired >= 1 && rep.contexts_recovered >= 1, "{rep:?}");
    let v = get_context(db, s.owner, c.id).await.unwrap();
    assert_eq!((v.context.state, v.context.version, v.active_leases), (State::Recorded, 3, 0));
    // a second reconcile is a no-op
    assert_eq!(reconcile(db, None, 100).await.unwrap(), ReconcileReport::default());

    // expired lease cleaned lazily by the next resume (no sweeper needed)
    let s2 = seed(db, 2).await;
    let c2 = create_context(db, s2.owner, s2.thread, None, None).await.unwrap();
    let (_, l2, _) = acquire_lease(db, s2.owner, c2.id, Some(1_000)).await.unwrap();
    transition(db, s2.owner, c2.id, req(Event::Materialize, 0, Some(l2.token))).await.unwrap();
    tokio::time::sleep(Duration::from_millis(1_300)).await;
    let (cur, l3, src) = acquire_lease(db, s2.owner, c2.id, None).await.unwrap();
    assert_eq!((cur.state, src), (State::Recorded, "replay"));
    assert!(l3.token > l2.token);

    // abandoning a build releases it back to RECORDED; abandoning a restore leaves FAILED
    release_lease(db, s2.owner, c2.id, l3.token).await.unwrap();
    let s3 = seed(db, 2).await;
    let c3 = create_context(db, s3.owner, s3.thread, None, None).await.unwrap();
    let (_, l4, _) = acquire_lease(db, s3.owner, c3.id, None).await.unwrap();
    transition(db, s3.owner, c3.id, req(Event::Materialize, 0, Some(l4.token))).await.unwrap();
    release_lease(db, s3.owner, c3.id, l4.token).await.unwrap();
    assert_eq!(get_context(db, s3.owner, c3.id).await.unwrap().context.state, State::Recorded);
    let (_, l5, _) = acquire_lease(db, s3.owner, c3.id, None).await.unwrap();
    let cur = get_context(db, s3.owner, c3.id).await.unwrap().context;
    transition(db, s3.owner, c3.id, req(Event::Materialize, cur.version, Some(l5.token))).await.unwrap();
    let mut rdy = req(Event::Ready, cur.version + 1, Some(l5.token));
    rdy.patch.engine_epoch = Some(Uuid::new_v4());
    transition(db, s3.owner, c3.id, rdy).await.unwrap();
    release_lease(db, s3.owner, c3.id, l5.token).await.unwrap();
    let ver = get_context(db, s3.owner, c3.id).await.unwrap().context.version;
    transition(db, s3.owner, c3.id, req(Event::Cool, ver, None)).await.unwrap();
    transition(db, s3.owner, c3.id, req(Event::Warm, ver + 1, None)).await.unwrap();
    let mut chill = req(Event::Chill, ver + 2, None);
    chill.patch = Patch { locator: Some("/snap/c3".into()), validated: Some(true), ..Default::default() };
    transition(db, s3.owner, c3.id, chill).await.unwrap();
    let (_, l6, _) = acquire_lease(db, s3.owner, c3.id, None).await.unwrap();
    transition(db, s3.owner, c3.id, req(Event::Restore, ver + 3, Some(l6.token))).await.unwrap();
    release_lease(db, s3.owner, c3.id, l6.token).await.unwrap();
    let after = get_context(db, s3.owner, c3.id).await.unwrap().context;
    assert_eq!((after.state, after.tier), (State::Failed, Tier::None));
    assert_eq!(after.locator.as_deref(), Some("/snap/c3"), "durable snapshot locator survives a failed restore");

    // engine restart: HOT context of a dead epoch is replayed, COLD one survives
    // one context per transcript head: each scenario below needs its own thread
    let s4 = seed(db, 2).await;
    let hot = create_context(db, s4.owner, s4.thread, None, None).await.unwrap();
    let (_, lh, _) = acquire_lease(db, s4.owner, hot.id, None).await.unwrap();
    transition(db, s4.owner, hot.id, req(Event::Materialize, 0, Some(lh.token))).await.unwrap();
    let old_epoch = Uuid::new_v4();
    let mut r = req(Event::Ready, 1, Some(lh.token));
    r.patch = Patch { engine_epoch: Some(old_epoch), locator: Some("gpu".into()), validated: Some(true), ..Default::default() };
    transition(db, s4.owner, hot.id, r).await.unwrap();
    release_lease(db, s4.owner, hot.id, lh.token).await.unwrap();
    reconcile(db, Some(old_epoch), 100).await.unwrap();
    assert_eq!(get_context(db, s4.owner, hot.id).await.unwrap().context.state, State::Hot,
        "context of the live epoch is left alone");
    let rep = reconcile(db, Some(Uuid::new_v4()), 100).await.unwrap();
    assert!(rep.contexts_recovered >= 1);
    let lost = get_context(db, s4.owner, hot.id).await.unwrap().context;
    assert_eq!((lost.state, lost.tier, lost.validated, lost.locator.clone(), lost.engine_epoch),
        (State::Recorded, Tier::None, false, None, None));
    assert_eq!(get_context(db, s3.owner, c3.id).await.unwrap().context.state, State::Failed);
}

async fn s_tombstone_and_gc(db: &PgPool) {
    let s = seed(db, 2).await;
    let p = create_context(db, s.owner, s.thread, None, None).await.unwrap();
    let (c, _) = fork_context(db, s.owner, p.id, 0, "k").await.unwrap();
    assert_eq!(tombstone(db, s.owner, p.id).await.unwrap().tombstoned, true);
    assert!(tombstone(db, s.owner, p.id).await.unwrap().tombstoned, "idempotent");
    not_found(get_context(db, s.owner, p.id).await);
    not_found(fork_context(db, s.owner, p.id, 0, "k2").await.map(|_| ()));
    not_found(acquire_lease(db, s.owner, p.id, None).await);
    let ids = |v: Vec<GcCandidate>| v.into_iter().map(|c| c.id).collect::<Vec<_>>();
    // child still references the tombstoned parent
    assert!(!ids(gc_candidates(db, 100).await.unwrap()).contains(&p.id));
    assert!(gc_purge(db, &[p.id]).await.unwrap().is_empty());
    // live lease pins the child
    let (_, l, _) = acquire_lease(db, s.owner, c.id, None).await.unwrap();
    tombstone(db, s.owner, c.id).await.unwrap();
    assert!(!ids(gc_candidates(db, 100).await.unwrap()).contains(&c.id));
    assert!(gc_purge(db, &[c.id]).await.unwrap().is_empty());
    // the thread itself is untouched
    let alive: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM threads WHERE id=$1 AND status='active')")
        .bind(s.thread).fetch_one(db).await.unwrap();
    assert!(alive);
    release_lease(db, s.owner, c.id, l.token).await.unwrap();
    let cands = gc_candidates(db, 100).await.unwrap();
    assert!(ids(cands.clone()).contains(&c.id) && !ids(cands).contains(&p.id));
    assert_eq!(gc_purge(db, &[c.id]).await.unwrap(), vec![c.id]);
    assert!(ids(gc_candidates(db, 100).await.unwrap()).contains(&p.id));
    assert_eq!(gc_purge(db, &[p.id]).await.unwrap(), vec![p.id]);
    let left: i64 = sqlx::query_scalar("SELECT count(*) FROM contexts WHERE id=ANY($1)")
        .bind(vec![p.id, c.id]).fetch_one(db).await.unwrap();
    assert_eq!(left, 0);
    let idem: i64 = sqlx::query_scalar("SELECT count(*) FROM context_idempotency WHERE target=$1")
        .bind(p.id).fetch_one(db).await.unwrap();
    assert_eq!(idem, 0, "idempotency record is collected with its child");
}

#[tokio::test]
async fn registry_against_postgres() {
    let Ok(admin_url) = std::env::var("BASALT_TEST_DATABASE_URL") else {
        eprintln!("BASALT_TEST_DATABASE_URL not set; skipping PostgreSQL integration");
        return;
    };
    let name = format!("ctxreg_{}", Uuid::new_v4().simple());
    let admin_opts = PgConnectOptions::from_str(&admin_url).unwrap();
    let mut admin = admin_opts.clone().connect().await.unwrap();
    sqlx::query(&format!("CREATE DATABASE {name}")).execute(&mut admin).await.unwrap();
    let pool = PgPoolOptions::new().max_connections(12)
        .connect_with(admin_opts.clone().database(&name)).await.unwrap();
    let outcome = tokio::spawn({
        let pool = pool.clone();
        async move {
            sqlx::migrate!("../migrations").run(&pool).await.unwrap();
            s_create_and_isolation(&pool).await;
            s_fork_idempotency(&pool).await;
            s_lifecycle_and_leases(&pool).await;
            s_expiry_and_reconcile(&pool).await;
            s_tombstone_and_gc(&pool).await;
        }
    }).await;
    pool.close().await;
    sqlx::query(&format!("DROP DATABASE {name} WITH (FORCE)")).execute(&mut admin).await.unwrap();
    outcome.unwrap();
}
