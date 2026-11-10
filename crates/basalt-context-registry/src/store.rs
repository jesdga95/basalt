//! PostgreSQL context registry. Every mutation locks the context row, checks
//! `version` (CAS) and, where the contract demands, the fenced writer lease.
use crate::fsm::{recovery_plan, tier_after, Event, LeaseRule, State, Tier, TransitionError};
use sha2::{Digest, Sha256};
use sqlx::{postgres::PgRow, PgConnection, PgPool, Row};
use uuid::Uuid;

pub type Digest32 = [u8; 32];

pub const DEFAULT_LEASE_TTL_MS: i64 = 30_000;
pub const MAX_LEASE_TTL_MS: i64 = 3_600_000;
const LINEAGE_DEPTH: i64 = 1024;
const COLS: &str = "id,owner_id,thread_id,branch_id,head_message_id,parent_id,state,tier,version,\
    model_fingerprint,prefix_hash,prefix_tokens,locator,validated,engine_epoch,\
    (deleted_at IS NOT NULL) AS tombstoned";
const LIVE_LEASE: &str = "l.released_at IS NULL AND l.expires_at > clock_timestamp()";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conflict {
    StaleVersion,
    IdempotencyPayload,
    InvalidTransition(TransitionError),
    LeaseHeld,
    LeaseRequired,
    LeaseInvalid,
    FingerprintMismatch,
    Busy,
}
impl Conflict {
    pub fn code(&self) -> &'static str {
        match self {
            Conflict::StaleVersion => "stale_version",
            Conflict::IdempotencyPayload => "idempotency_conflict",
            Conflict::InvalidTransition(_) => "invalid_transition",
            Conflict::LeaseHeld => "lease_held",
            Conflict::LeaseRequired => "lease_required",
            Conflict::LeaseInvalid => "lease_invalid",
            Conflict::FingerprintMismatch => "fingerprint_mismatch",
            Conflict::Busy => "context_busy",
        }
    }
    pub fn message(&self) -> String {
        match self {
            Conflict::StaleVersion => "expected_version does not match the current version".into(),
            Conflict::IdempotencyPayload => "Idempotency-Key was already used with a different payload".into(),
            Conflict::InvalidTransition(e) =>
                format!("event {} is not allowed in state {}", e.event.as_str(), e.state.as_str()),
            Conflict::LeaseHeld => "another writer lease or live reference holds this context".into(),
            Conflict::LeaseRequired => "this event requires the active lease_token".into(),
            Conflict::LeaseInvalid => "lease token is stale, expired or released".into(),
            Conflict::FingerprintMismatch => "published manifest does not match the stored fingerprint".into(),
            Conflict::Busy => "context is being evicted; retry later".into(),
        }
    }
}

#[derive(Debug)]
pub enum RegistryError {
    NotFound,
    Conflict(Conflict),
    Invalid(String),
    Database(sqlx::Error),
}
impl From<sqlx::Error> for RegistryError {
    fn from(e: sqlx::Error) -> Self {
        if let sqlx::Error::Database(d) = &e {
            if d.constraint() == Some("context_leases_one_active") {
                return Self::Conflict(Conflict::LeaseHeld);
            }
            if d.code().as_deref() == Some("40001") {
                return Self::Conflict(Conflict::StaleVersion);
            }
        }
        Self::Database(e)
    }
}
type Res<T> = Result<T, RegistryError>;
fn invalid<T>(msg: impl Into<String>) -> Res<T> { Err(RegistryError::Invalid(msg.into())) }
fn conflict<T>(c: Conflict) -> Res<T> { Err(RegistryError::Conflict(c)) }

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context {
    pub id: Uuid,
    pub owner_id: Uuid,
    pub thread_id: Uuid,
    pub branch_id: Uuid,
    pub head_message_id: Option<Uuid>,
    pub parent_id: Option<Uuid>,
    pub state: State,
    pub tier: Tier,
    pub version: i64,
    pub model_fingerprint: Digest32,
    pub prefix_hash: Digest32,
    pub prefix_tokens: i64,
    pub locator: Option<String>,
    pub validated: bool,
    pub engine_epoch: Option<Uuid>,
    pub tombstoned: bool,
}

fn digest(v: Vec<u8>) -> Res<Digest32> {
    v.try_into().map_err(|_| RegistryError::Invalid("stored digest has wrong length".into()))
}
fn context_from(r: &PgRow) -> Res<Context> {
    let state: String = r.try_get("state")?;
    let tier: String = r.try_get("tier")?;
    Ok(Context {
        id: r.try_get("id")?, owner_id: r.try_get("owner_id")?, thread_id: r.try_get("thread_id")?,
        branch_id: r.try_get("branch_id")?, head_message_id: r.try_get("head_message_id")?,
        parent_id: r.try_get("parent_id")?,
        state: State::parse(&state).ok_or_else(|| RegistryError::Invalid(format!("unknown state {state}")))?,
        tier: Tier::parse(&tier).ok_or_else(|| RegistryError::Invalid(format!("unknown tier {tier}")))?,
        version: r.try_get("version")?,
        model_fingerprint: digest(r.try_get("model_fingerprint")?)?,
        prefix_hash: digest(r.try_get("prefix_hash")?)?,
        prefix_tokens: r.try_get("prefix_tokens")?, locator: r.try_get("locator")?,
        validated: r.try_get("validated")?, engine_epoch: r.try_get("engine_epoch")?,
        tombstoned: r.try_get("tombstoned")?,
    })
}

pub fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn parse_hex32(s: &str) -> Option<Digest32> {
    if s.len() != 64 || !s.is_ascii() { return None; }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// Domain-separated identity of the transcript prefix: ordered (role, content_hash).
/// Tenant isolation never relies on this value alone; `owner_id` scopes every query.
pub fn prefix_hash(chain: &[(String, Digest32)]) -> Digest32 {
    let mut h = Sha256::new();
    h.update(b"basalt.context.prefix.v1\0");
    for (role, content_hash) in chain {
        h.update((role.len() as u64).to_le_bytes());
        h.update(role.as_bytes());
        h.update(content_hash);
    }
    h.finalize().into()
}
/// Provisional label until the engine publishes its real manifest fingerprint.
pub fn provisional_fingerprint(model_id: &str) -> Digest32 {
    let mut h = Sha256::new();
    h.update(b"basalt.context.model-label.v1\0");
    h.update(model_id.as_bytes());
    h.finalize().into()
}
fn fork_request_hash(expected_version: i64) -> Digest32 {
    let mut h = Sha256::new();
    h.update(b"basalt.context.fork.v1\0");
    h.update(expected_version.to_le_bytes());
    h.finalize().into()
}

async fn select_context(conn: &mut PgConnection, id: Uuid, owner: Option<Uuid>,
                        lock: bool, skip_locked: bool) -> Res<Option<Context>> {
    let suffix = match (lock, skip_locked) {
        (false, _) => "", (true, false) => " FOR UPDATE", (true, true) => " FOR UPDATE SKIP LOCKED",
    };
    let sql = format!("SELECT {COLS} FROM contexts WHERE id=$1 AND ($2::uuid IS NULL OR owner_id=$2){suffix}");
    match sqlx::query(&sql).bind(id).bind(owner).fetch_optional(conn).await? {
        Some(r) => Ok(Some(context_from(&r)?)),
        None => Ok(None),
    }
}
/// Owner-scoped lock. Foreign and unknown contexts are indistinguishable (404).
async fn lock_owned(conn: &mut PgConnection, owner: Uuid, id: Uuid, allow_tombstoned: bool) -> Res<Context> {
    match select_context(conn, id, Some(owner), true, false).await? {
        Some(c) if allow_tombstoned || !c.tombstoned => Ok(c),
        _ => Err(RegistryError::NotFound),
    }
}

async fn live_lease(conn: &mut PgConnection, ctx: Uuid) -> Res<Option<i64>> {
    let sql = format!("SELECT l.token FROM context_leases l WHERE l.context_id=$1 AND {LIVE_LEASE}");
    Ok(sqlx::query_scalar(&sql).bind(ctx).fetch_optional(conn).await?)
}
async fn expire_leases(conn: &mut PgConnection, ctx: Uuid) -> Res<u64> {
    Ok(sqlx::query(
        "UPDATE context_leases SET released_at=clock_timestamp(), release_reason='expired'
         WHERE context_id=$1 AND released_at IS NULL AND expires_at <= clock_timestamp()")
        .bind(ctx).execute(conn).await?.rows_affected())
}

#[derive(Debug, Default, Clone)]
pub struct Patch {
    pub tier: Option<Tier>,
    pub locator: Option<String>,
    pub validated: Option<bool>,
    pub model_fingerprint: Option<Digest32>,
    pub prefix_hash: Option<Digest32>,
    pub prefix_tokens: Option<i64>,
    pub engine_epoch: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct TransitionRequest {
    pub event: Event,
    pub expected_version: i64,
    pub lease_token: Option<i64>,
    pub patch: Patch,
}

fn validate_patch(event: Event, p: &Patch) -> Res<()> {
    let snapshot_event = matches!(event, Event::Ready | Event::Warm | Event::Chill);
    if (p.locator.is_some() || p.validated.is_some()) && !snapshot_event {
        return invalid("locator/validated are only accepted with ready, warm or chill");
    }
    if (p.model_fingerprint.is_some() || p.prefix_hash.is_some() || p.prefix_tokens.is_some())
        && event != Event::Ready {
        return invalid("manifest fields are only accepted with ready");
    }
    if p.engine_epoch.is_some() && !matches!(event, Event::Materialize | Event::Restore | Event::Ready) {
        return invalid("engine_epoch is only accepted with materialize, restore or ready");
    }
    if p.locator.as_ref().is_some_and(|l| l.is_empty() || l.len() > 4096 || l.contains('\0')) {
        return invalid("locator must be 1..4096 characters without NUL");
    }
    if p.prefix_tokens.is_some_and(|n| n < 0) {
        return invalid("prefix_tokens must be non-negative");
    }
    if event == Event::Record {
        return conflict(Conflict::InvalidTransition(TransitionError { state: State::Recorded, event }));
    }
    Ok(())
}

/// Apply one FSM transition to a locked row. Lease policy is the caller's job.
async fn step(conn: &mut PgConnection, c: &Context, event: Event, p: &Patch) -> Res<Context> {
    let next = c.state.transition(event)
        .map_err(|e| RegistryError::Conflict(Conflict::InvalidTransition(e)))?;
    let tier = tier_after(c.state, next);
    if p.tier.is_some_and(|t| t != tier) {
        return invalid(format!("tier must be {} for {} -> {}", tier.as_str(), c.state.as_str(), next.as_str()));
    }
    let (mut fp, mut ph, mut pt) = (c.model_fingerprint, c.prefix_hash, c.prefix_tokens);
    if event == Event::Ready {
        if c.state == State::Restoring {
            // A restore must reproduce the identity that was published when it was built.
            if p.model_fingerprint.is_some_and(|v| v != fp) || p.prefix_hash.is_some_and(|v| v != ph)
                || p.prefix_tokens.is_some_and(|v| v != pt) {
                return conflict(Conflict::FingerprintMismatch);
            }
        } else {
            fp = p.model_fingerprint.unwrap_or(fp);
            ph = p.prefix_hash.unwrap_or(ph);
            pt = p.prefix_tokens.unwrap_or(pt);
        }
    }
    let validated = tier != Tier::None && p.validated.unwrap_or(c.validated);
    let locator = p.locator.clone().or_else(|| c.locator.clone());
    let epoch = p.engine_epoch.or(c.engine_epoch);
    let sql = format!(
        "UPDATE contexts SET state=$3,tier=$4,version=version+1,model_fingerprint=$5,prefix_hash=$6,
           prefix_tokens=$7,locator=$8,validated=$9,engine_epoch=$10,updated_at=clock_timestamp()
         WHERE id=$1 AND owner_id=$2 AND version=$11 RETURNING {COLS}");
    let row = sqlx::query(&sql).bind(c.id).bind(c.owner_id).bind(next.as_str()).bind(tier.as_str())
        .bind(fp.as_slice()).bind(ph.as_slice()).bind(pt).bind(locator).bind(validated).bind(epoch)
        .bind(c.version).fetch_optional(conn).await?
        .ok_or(RegistryError::Conflict(Conflict::StaleVersion))?;
    context_from(&row)
}

async fn check_lease(conn: &mut PgConnection, c: &Context, event: Event, token: Option<i64>) -> Res<()> {
    let live = live_lease(conn, c.id).await?;
    match event.lease_rule(c.state) {
        LeaseRule::Required => match (token, live) {
            (None, _) => conflict(Conflict::LeaseRequired),
            (Some(t), Some(l)) if t == l => Ok(()),
            _ => conflict(Conflict::LeaseInvalid),
        },
        LeaseRule::Optional => match token {
            Some(t) if Some(t) != live => conflict(Conflict::LeaseInvalid),
            _ => Ok(()),
        },
        LeaseRule::Forbidden => {
            if token.is_some() { return invalid("this event does not take a lease_token"); }
            if live.is_some() { return conflict(Conflict::LeaseHeld); }
            Ok(())
        }
    }
}

/// Engine/scheduler-driven transition: CAS on `version`, fenced by the lease token.
pub async fn transition(db: &PgPool, owner: Uuid, id: Uuid, req: TransitionRequest) -> Res<Context> {
    validate_patch(req.event, &req.patch)?;
    let mut tx = db.begin().await?;
    let c = lock_owned(&mut tx, owner, id, false).await?;
    if c.version != req.expected_version { return conflict(Conflict::StaleVersion); }
    c.state.transition(req.event).map_err(|e| RegistryError::Conflict(Conflict::InvalidTransition(e)))?;
    check_lease(&mut tx, &c, req.event, req.lease_token).await?;
    let out = step(&mut tx, &c, req.event, &req.patch).await?;
    tx.commit().await?;
    Ok(out)
}

/// Create a RECORDED context from a thread branch prefix (optionally cut at `upto`).
pub async fn create_context(db: &PgPool, owner: Uuid, thread: Uuid, branch: Option<Uuid>,
                            upto: Option<Uuid>) -> Res<Context> {
    let mut tx = db.begin().await?;
    let t = sqlx::query("SELECT model_id,default_branch_id FROM threads
                         WHERE id=$1 AND owner_id=$2 AND status='active'")
        .bind(thread).bind(owner).fetch_optional(&mut *tx).await?.ok_or(RegistryError::NotFound)?;
    let model: String = t.try_get("model_id")?;
    let default_branch: Option<Uuid> = t.try_get("default_branch_id")?;
    let Some(branch_id) = branch.or(default_branch) else { return invalid("thread has no branch") };
    let b = sqlx::query("SELECT head_message_id FROM branches WHERE id=$1 AND thread_id=$2")
        .bind(branch_id).bind(thread).fetch_optional(&mut *tx).await?.ok_or(RegistryError::NotFound)?;
    let branch_head: Option<Uuid> = b.try_get("head_message_id")?;
    let mut chain: Vec<(Uuid, String, Digest32)> = Vec::new();
    if let Some(head) = branch_head {
        let rows = sqlx::query(
            "WITH RECURSIVE chain AS (
               SELECT m.id,m.parent_id,m.role,m.content_hash,1::bigint AS depth
               FROM messages m WHERE m.id=$1 AND m.thread_id=$2
               UNION ALL
               SELECT p.id,p.parent_id,p.role,p.content_hash,c.depth+1
               FROM messages p JOIN chain c ON p.id=c.parent_id WHERE p.thread_id=$2)
             SELECT id,role,content_hash FROM chain ORDER BY depth DESC")
            .bind(head).bind(thread).fetch_all(&mut *tx).await?;
        for r in rows {
            chain.push((r.try_get("id")?, r.try_get("role")?, digest(r.try_get("content_hash")?)?));
        }
    }
    let head = match upto {
        None => branch_head,
        Some(m) => {
            let known: bool = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM messages WHERE id=$1 AND thread_id=$2)")
                .bind(m).bind(thread).fetch_one(&mut *tx).await?;
            if !known { return Err(RegistryError::NotFound); }
            let Some(pos) = chain.iter().position(|(id, _, _)| *id == m) else {
                return invalid("upto_message_id is not on the requested branch");
            };
            chain.truncate(pos + 1);
            Some(m)
        }
    };
    let roles: Vec<(String, Digest32)> = chain.into_iter().map(|(_, r, h)| (r, h)).collect();
    // One context per transcript head: a repeated attach reuses the branch's live context at this position
    // instead of fragmenting the lineage with a fresh row per request. The branch's own (unforked) context
    // wins over forked children that share the head; children keep their private leases through /fork.
    let existing = sqlx::query(&format!(
        "SELECT {COLS} FROM contexts WHERE owner_id=$1 AND thread_id=$2 AND branch_id=$3 \
         AND head_message_id IS NOT DISTINCT FROM $4 AND deleted_at IS NULL \
         ORDER BY (parent_id IS NULL) DESC, updated_at DESC LIMIT 1"))
        .bind(owner).bind(thread).bind(branch_id).bind(head)
        .fetch_optional(&mut *tx).await?;
    if let Some(r) = existing {
        let out = context_from(&r)?;
        tx.commit().await?;
        return Ok(out);
    }
    let id = Uuid::new_v4();
    let sql = format!(
        "INSERT INTO contexts(id,owner_id,thread_id,branch_id,head_message_id,model_fingerprint,prefix_hash)
         VALUES($1,$2,$3,$4,$5,$6,$7) RETURNING {COLS}");
    let row = sqlx::query(&sql).bind(id).bind(owner).bind(thread).bind(branch_id).bind(head)
        .bind(provisional_fingerprint(&model).as_slice()).bind(prefix_hash(&roles).as_slice())
        .fetch_one(&mut *tx).await?;
    let out = context_from(&row)?;
    tx.commit().await?;
    Ok(out)
}

fn validate_key(key: &str) -> Res<()> {
    if key.is_empty() || key.len() > 256 || key.bytes().any(|b| !(0x20..0x7f).contains(&b)) {
        return invalid("Idempotency-Key must be 1..256 printable ASCII characters");
    }
    Ok(())
}

/// Idempotent child creation. Returns (child, replayed). The parent row lock serializes
/// concurrent forks, so one key can never produce two children.
pub async fn fork_context(db: &PgPool, owner: Uuid, parent: Uuid, expected_version: i64,
                          key: &str) -> Res<(Context, bool)> {
    validate_key(key)?;
    if expected_version < 0 { return invalid("expected_version must be non-negative"); }
    let hash = fork_request_hash(expected_version);
    let mut tx = db.begin().await?;
    let p = lock_owned(&mut tx, owner, parent, false).await?;
    let prior = sqlx::query("SELECT request_hash,result_context_id FROM context_idempotency
                             WHERE owner_id=$1 AND operation='fork' AND target=$2 AND key=$3")
        .bind(owner).bind(parent).bind(key).fetch_optional(&mut *tx).await?;
    if let Some(r) = prior {
        let old: Vec<u8> = r.try_get("request_hash")?;
        if old != hash { return conflict(Conflict::IdempotencyPayload); }
        let child_id: Uuid = r.try_get("result_context_id")?;
        let child = select_context(&mut tx, child_id, Some(owner), false, false).await?
            .ok_or(RegistryError::NotFound)?;
        tx.commit().await?;
        return Ok((child, true));
    }
    if p.version != expected_version { return conflict(Conflict::StaleVersion); }
    let id = Uuid::new_v4();
    let sql = format!(
        "INSERT INTO contexts(id,owner_id,thread_id,branch_id,head_message_id,parent_id,
                              model_fingerprint,prefix_hash,prefix_tokens)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING {COLS}");
    let row = sqlx::query(&sql).bind(id).bind(owner).bind(p.thread_id).bind(p.branch_id)
        .bind(p.head_message_id).bind(p.id).bind(p.model_fingerprint.as_slice())
        .bind(p.prefix_hash.as_slice()).bind(p.prefix_tokens).fetch_one(&mut *tx).await?;
    sqlx::query("INSERT INTO context_idempotency(owner_id,operation,target,key,request_hash,result_context_id)
                 VALUES($1,'fork',$2,$3,$4,$5)")
        .bind(owner).bind(parent).bind(key).bind(hash.as_slice()).bind(id).execute(&mut *tx).await?;
    let child = context_from(&row)?;
    tx.commit().await?;
    Ok((child, false))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease { pub token: i64, pub expires_at: String }

/// Orphan = MATERIALIZING/RESTORING without a live lease: nobody is writing it any more.
async fn recover_if_orphan(conn: &mut PgConnection, c: Context) -> Res<Context> {
    if !matches!(c.state, State::Materializing | State::Restoring)
        || live_lease(conn, c.id).await?.is_some() {
        return Ok(c);
    }
    recover_row(conn, c, false).await
}

async fn recover_row(conn: &mut PgConnection, c: Context, engine_lost: bool) -> Res<Context> {
    let plan = recovery_plan(c.state, engine_lost);
    let mut cur = c;
    for ev in &plan {
        cur = step(conn, &cur, *ev, &Patch::default()).await?;
    }
    if engine_lost && !plan.is_empty() {
        // Volatile residency died with the engine: the locator points at nothing.
        sqlx::query("UPDATE contexts SET locator=NULL,engine_epoch=NULL WHERE id=$1")
            .bind(cur.id).execute(&mut *conn).await?;
        cur.locator = None;
        cur.engine_epoch = None;
    }
    Ok(cur)
}

/// Acquire the single fenced writer lease. `source` tells the caller what to restore from.
pub async fn acquire_lease(db: &PgPool, owner: Uuid, id: Uuid, ttl_ms: Option<i64>)
    -> Res<(Context, Lease, &'static str)> {
    let ttl = ttl_ms.unwrap_or(DEFAULT_LEASE_TTL_MS);
    if !(1_000..=MAX_LEASE_TTL_MS).contains(&ttl) {
        return invalid(format!("lease_ttl_ms must be within 1000..{MAX_LEASE_TTL_MS}"));
    }
    let mut tx = db.begin().await?;
    let c = lock_owned(&mut tx, owner, id, false).await?;
    expire_leases(&mut tx, id).await?;
    let c = recover_if_orphan(&mut tx, c).await?;
    if c.state == State::Cooling { return conflict(Conflict::Busy); }
    if live_lease(&mut tx, id).await?.is_some() { return conflict(Conflict::LeaseHeld); }
    let row = sqlx::query(
        "INSERT INTO context_leases(token,context_id,owner_id,expires_at)
         VALUES(nextval('basalt_thread_fencing_seq'),$1,$2,
                clock_timestamp()+make_interval(secs => $3::double precision / 1000.0))
         RETURNING token,to_char(expires_at AT TIME ZONE 'UTC','YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') AS exp")
        .bind(id).bind(owner).bind(ttl).fetch_one(&mut *tx).await?;
    let lease = Lease { token: row.try_get("token")?, expires_at: row.try_get("exp")? };
    tx.commit().await?;
    let source = c.state.source();
    Ok((c, lease, source))
}

/// Release a lease. A repeated release of the same token succeeds; an expired or
/// unknown token is rejected so a fenced writer learns it lost the context.
pub async fn release_lease(db: &PgPool, owner: Uuid, id: Uuid, token: i64) -> Res<()> {
    let mut tx = db.begin().await?;
    let c = lock_owned(&mut tx, owner, id, true).await?;
    let row = sqlx::query("SELECT released_at IS NOT NULL AS released,release_reason,
                                  (expires_at <= clock_timestamp()) AS expired
                           FROM context_leases WHERE token=$1 AND context_id=$2")
        .bind(token).bind(id).fetch_optional(&mut *tx).await?
        .ok_or(RegistryError::Conflict(Conflict::LeaseInvalid))?;
    let released: bool = row.try_get("released")?;
    let reason: Option<String> = row.try_get("release_reason")?;
    let expired: bool = row.try_get("expired")?;
    if released {
        return if reason.as_deref() == Some("released") { Ok(()) } else { conflict(Conflict::LeaseInvalid) };
    }
    if expired {
        expire_leases(&mut tx, id).await?;
        recover_if_orphan(&mut tx, c).await?;
        tx.commit().await?;
        return conflict(Conflict::LeaseInvalid);
    }
    sqlx::query("UPDATE context_leases SET released_at=clock_timestamp(),release_reason='released'
                 WHERE token=$1").bind(token).execute(&mut *tx).await?;
    // The writer walked away from an unfinished build/restore: nothing else will finish it.
    recover_if_orphan(&mut tx, c).await?;
    tx.commit().await?;
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineageEntry { pub id: Uuid, pub state: State, pub version: i64, pub tombstoned: bool }

#[derive(Debug, Clone)]
pub struct ContextView {
    pub context: Context,
    /// Root first, direct parent last.
    pub lineage: Vec<LineageEntry>,
    pub children: Vec<Uuid>,
    /// Children rows (tombstoned but not yet collected included) that pin the parent.
    pub child_refs: i64,
    pub active_leases: i64,
}

pub async fn get_context(db: &PgPool, owner: Uuid, id: Uuid) -> Res<ContextView> {
    let mut conn = db.acquire().await?;
    let context = select_context(&mut conn, id, Some(owner), false, false).await?
        .filter(|c| !c.tombstoned).ok_or(RegistryError::NotFound)?;
    let rows = sqlx::query(
        "WITH RECURSIVE up AS (
           SELECT c.id,c.parent_id,c.state,c.version,(c.deleted_at IS NOT NULL) AS tomb,1 AS depth
           FROM contexts c WHERE c.id=$1 AND c.owner_id=$2
           UNION ALL
           SELECT c.id,c.parent_id,c.state,c.version,(c.deleted_at IS NOT NULL),u.depth+1
           FROM contexts c JOIN up u ON c.id=u.parent_id WHERE c.owner_id=$2 AND u.depth < $3)
         SELECT id,state,version,tomb FROM up WHERE id <> $1 ORDER BY depth DESC")
        .bind(id).bind(owner).bind(LINEAGE_DEPTH).fetch_all(&mut *conn).await?;
    let mut lineage = Vec::new();
    for r in rows {
        let s: String = r.try_get("state")?;
        lineage.push(LineageEntry {
            id: r.try_get("id")?, version: r.try_get("version")?, tombstoned: r.try_get("tomb")?,
            state: State::parse(&s).ok_or_else(|| RegistryError::Invalid(format!("unknown state {s}")))?,
        });
    }
    let children: Vec<Uuid> = sqlx::query_scalar(
        "SELECT id FROM contexts WHERE parent_id=$1 AND owner_id=$2 AND deleted_at IS NULL
         ORDER BY created_at,id").bind(id).bind(owner).fetch_all(&mut *conn).await?;
    let child_refs: i64 = sqlx::query_scalar("SELECT count(*) FROM contexts WHERE parent_id=$1 AND owner_id=$2")
        .bind(id).bind(owner).fetch_one(&mut *conn).await?;
    let active_leases: i64 = sqlx::query_scalar(&format!(
        "SELECT count(*) FROM context_leases l WHERE l.context_id=$1 AND {LIVE_LEASE}"))
        .bind(id).fetch_one(&mut *conn).await?;
    Ok(ContextView { context, lineage, children, child_refs, active_leases })
}

/// One-way tombstone; physical removal is deferred to GC. Never touches the thread.
pub async fn tombstone(db: &PgPool, owner: Uuid, id: Uuid) -> Res<Context> {
    let mut tx = db.begin().await?;
    let c = lock_owned(&mut tx, owner, id, true).await?;
    if c.tombstoned {
        tx.commit().await?;
        return Ok(c);
    }
    let sql = format!("UPDATE contexts SET deleted_at=clock_timestamp(),version=version+1,
                         updated_at=clock_timestamp() WHERE id=$1 RETURNING {COLS}");
    let row = sqlx::query(&sql).bind(id).fetch_one(&mut *tx).await?;
    let out = context_from(&row)?;
    tx.commit().await?;
    Ok(out)
}

/// Engine confirmation that a tombstoned context's snapshot payload is gone: the locator,
/// tier, validated flag and epoch are cleared so the sweeper's GC can purge the row. Only a
/// tombstoned context can be released this way; a live context keeps its recoverable snapshot.
/// Idempotent: an already-released row is returned unchanged.
pub async fn release_payload(db: &PgPool, owner: Uuid, id: Uuid) -> Res<Context> {
    let mut tx = db.begin().await?;
    let c = lock_owned(&mut tx, owner, id, true).await?;
    if !c.tombstoned {
        return Err(RegistryError::Invalid("payload release requires a tombstoned context".into()));
    }
    if c.locator.is_none() && c.tier == Tier::None && !c.validated && c.engine_epoch.is_none() {
        tx.commit().await?;
        return Ok(c);
    }
    let sql = format!("UPDATE contexts SET state='RECORDED',tier='none',validated=false,locator=NULL,\
                         engine_epoch=NULL,version=version+1,updated_at=clock_timestamp() \
                         WHERE id=$1 RETURNING {COLS}");
    let row = sqlx::query(&sql).bind(id).fetch_one(&mut *tx).await?;
    let out = context_from(&row)?;
    tx.commit().await?;
    Ok(out)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcCandidate { pub id: Uuid, pub owner_id: Uuid, pub tier: Tier, pub locator: Option<String> }

const GC_PREDICATE: &str = "c.deleted_at IS NOT NULL
    AND NOT EXISTS (SELECT 1 FROM contexts ch WHERE ch.parent_id=c.id)
    AND NOT EXISTS (SELECT 1 FROM context_leases l WHERE l.context_id=c.id AND l.released_at IS NULL
                    AND l.expires_at > clock_timestamp())";

/// Tombstoned contexts that no child or live lease references. The caller frees the
/// external payload (`tier`/`locator`) and then confirms with `gc_purge`.
pub async fn gc_candidates(db: &PgPool, limit: i64) -> Res<Vec<GcCandidate>> {
    if !(1..=10_000).contains(&limit) { return invalid("limit out of range"); }
    let rows = sqlx::query(&format!(
        "SELECT c.id,c.owner_id,c.tier,c.locator FROM contexts c WHERE {GC_PREDICATE}
         ORDER BY c.deleted_at LIMIT $1")).bind(limit).fetch_all(db).await?;
    rows.iter().map(|r| {
        let t: String = r.try_get("tier")?;
        Ok(GcCandidate { id: r.try_get("id")?, owner_id: r.try_get("owner_id")?,
            tier: Tier::parse(&t).ok_or_else(|| RegistryError::Invalid(format!("unknown tier {t}")))?,
            locator: r.try_get("locator")? })
    }).collect()
}

/// Physically remove collectable contexts (conditions re-checked under row lock).
pub async fn gc_purge(db: &PgPool, ids: &[Uuid]) -> Res<Vec<Uuid>> {
    let mut purged = Vec::new();
    for id in ids {
        let mut tx = db.begin().await?;
        let ok: Option<Uuid> = sqlx::query_scalar(&format!(
            "SELECT c.id FROM contexts c WHERE c.id=$1 AND {GC_PREDICATE} FOR UPDATE SKIP LOCKED"))
            .bind(id).fetch_optional(&mut *tx).await?;
        if ok.is_none() { continue; }
        sqlx::query("DELETE FROM context_leases WHERE context_id=$1").bind(id).execute(&mut *tx).await?;
        sqlx::query("DELETE FROM contexts WHERE id=$1").bind(id).execute(&mut *tx).await?;
        tx.commit().await?;
        purged.push(*id);
    }
    Ok(purged)
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ReconcileReport { pub leases_expired: u64, pub contexts_recovered: u64 }

/// Restart/sweep recovery: expire stale leases, fail or rebuild orphaned
/// MATERIALIZING/RESTORING contexts, and (when the live engine epoch is known)
/// send HOT/COOLING/WARM contexts of a dead engine back to RECORDED for replay.
pub async fn reconcile(db: &PgPool, current_epoch: Option<Uuid>, batch: i64) -> Res<ReconcileReport> {
    if !(1..=1_000).contains(&batch) { return invalid("batch out of range"); }
    let leases_expired = sqlx::query(
        "UPDATE context_leases SET released_at=clock_timestamp(),release_reason='expired'
         WHERE released_at IS NULL AND expires_at <= clock_timestamp()")
        .execute(db).await?.rows_affected();
    let ids: Vec<Uuid> = sqlx::query_scalar(&format!(
        "SELECT c.id FROM contexts c WHERE
           (c.state IN ('MATERIALIZING','RESTORING') AND NOT EXISTS
             (SELECT 1 FROM context_leases l WHERE l.context_id=c.id AND {LIVE_LEASE}))
           OR ($1::uuid IS NOT NULL AND c.state IN ('HOT','COOLING','WARM')
               AND c.engine_epoch IS DISTINCT FROM $1)
         ORDER BY c.updated_at LIMIT $2"))
        .bind(current_epoch).bind(batch).fetch_all(db).await?;
    let mut recovered = 0;
    for id in ids {
        let mut tx = db.begin().await?;
        let Some(c) = select_context(&mut tx, id, None, true, true).await? else { continue };
        let lost = current_epoch.is_some()
            && matches!(c.state, State::Hot | State::Cooling | State::Warm)
            && c.engine_epoch != current_epoch;
        let orphan = matches!(c.state, State::Materializing | State::Restoring)
            && live_lease(&mut tx, id).await?.is_none();
        if !lost && !orphan { continue; }
        recover_row(&mut tx, c, lost).await?;
        tx.commit().await?;
        recovered += 1;
    }
    Ok(ReconcileReport { leases_expired, contexts_recovered: recovered })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip_and_rejects_bad_input() {
        let d: Digest32 = core::array::from_fn(|i| i as u8 * 7);
        assert_eq!(parse_hex32(&hex(&d)), Some(d));
        assert_eq!(hex(&[0xab, 0x01]), "ab01");
        assert_eq!(parse_hex32("zz"), None);
        assert_eq!(parse_hex32(&"g".repeat(64)), None);
        assert_eq!(parse_hex32(&"é".repeat(32)), None);
    }

    #[test]
    fn prefix_hash_is_ordered_and_length_framed() {
        let a = ("user".to_string(), [1u8; 32]);
        let b = ("assistant".to_string(), [2u8; 32]);
        assert_eq!(prefix_hash(&[a.clone(), b.clone()]), prefix_hash(&[a.clone(), b.clone()]));
        assert_ne!(prefix_hash(&[a.clone(), b.clone()]), prefix_hash(&[b.clone(), a.clone()]));
        assert_ne!(prefix_hash(&[a.clone()]), prefix_hash(&[a, b]));
        assert_ne!(prefix_hash(&[]), provisional_fingerprint(""));
    }

    #[test]
    fn fork_hash_depends_on_payload() {
        assert_eq!(fork_request_hash(3), fork_request_hash(3));
        assert_ne!(fork_request_hash(3), fork_request_hash(4));
    }

    #[test]
    fn idempotency_key_validation() {
        assert!(validate_key("abc-123").is_ok());
        assert!(validate_key("").is_err());
        assert!(validate_key(&"k".repeat(257)).is_err());
        assert!(validate_key("bad\nkey").is_err());
        assert!(validate_key("키").is_err());
    }

    #[test]
    fn patch_validation_scopes_fields_to_events() {
        let loc = Patch { locator: Some("/snap/a".into()), ..Default::default() };
        assert!(validate_patch(Event::Chill, &loc).is_ok());
        assert!(validate_patch(Event::Cool, &loc).is_err());
        let manifest = Patch { prefix_tokens: Some(10), ..Default::default() };
        assert!(validate_patch(Event::Ready, &manifest).is_ok());
        assert!(validate_patch(Event::Warm, &manifest).is_err());
        let epoch = Patch { engine_epoch: Some(Uuid::nil()), ..Default::default() };
        assert!(validate_patch(Event::Materialize, &epoch).is_ok());
        assert!(validate_patch(Event::Fail, &epoch).is_err());
        assert!(validate_patch(Event::Warm, &Patch { locator: Some(String::new()), ..Default::default() }).is_err());
        assert!(matches!(validate_patch(Event::Record, &Patch::default()),
            Err(RegistryError::Conflict(Conflict::InvalidTransition(_)))));
    }
}
