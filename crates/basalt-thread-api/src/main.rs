//! Loopback-only API sidecar. Basalt's existing HTTP handler authenticates and forwards
//! thread calls; this process has a separate mandatory shared secret.
mod contexts;
use axum::{
    extract::{Path, State}, http::{HeaderMap, StatusCode}, routing::{get, post},
    Json, Router,
};
use basalt_thread_store::{accept_turn, commit_turn, history, request_hash, LedgerError};
use basalt_thread_runtime::{claim_recovery, heartbeat, reconcile_expired, RecoveryError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use std::{env, net::SocketAddr, sync::Arc, time::{Duration, Instant}};
use uuid::Uuid;

#[derive(Clone)]
struct App {
    pool: PgPool,
    client: reqwest::Client,
    secret: String,
    owner: Uuid,
    workspace: Uuid,
    upstream: String,
}
type Result<T> = std::result::Result<T, (StatusCode, Json<Value>)>;
fn error(code: StatusCode, reason: &str) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"error":{"type":"thread_error","message":reason}})))
}
fn db_error(err: sqlx::Error) -> (StatusCode, Json<Value>) {
    eprintln!("thread database operation failed: {err}");
    error(StatusCode::INTERNAL_SERVER_ERROR, "thread database error")
}
fn recovery_error(err: RecoveryError) -> (StatusCode, Json<Value>) {
    match err {
        RecoveryError::Conflict => error(StatusCode::CONFLICT, "recovery is not claimable"),
        RecoveryError::UnsafeToolState => error(StatusCode::CONFLICT,
            "external tool state uncertain; manual reconciliation required"),
        RecoveryError::Database(e) => db_error(e),
    }
}
fn ledger_error(err: LedgerError) -> (StatusCode, Json<Value>) {
    match err {
        LedgerError::Conflict => error(StatusCode::CONFLICT, "branch or idempotency conflict"),
        LedgerError::Missing => error(StatusCode::NOT_FOUND, "thread or branch not found"),
        LedgerError::Database(e) => db_error(e),
    }
}
fn authorize(headers: &HeaderMap, app: &App) -> Result<()> {
    // Only localhost's Basalt handler should know this separate sidecar secret.
    let bearer = headers.get("authorization").and_then(|v|v.to_str().ok())
        .and_then(|s|s.strip_prefix("Bearer "));
    if bearer != Some(app.secret.as_str()) {
        return Err(error(StatusCode::UNAUTHORIZED, "unauthorized"));
    }
    Ok(())
}
#[derive(Deserialize)]
struct NewThread { model: String }
#[derive(Serialize)]
struct ThreadView {
    id: Uuid, branch_id: Uuid, head_message_id: Option<Uuid>,
    version: i64, model: String,
}
async fn create_thread(
    State(app): State<Arc<App>>, headers: HeaderMap, Json(req): Json<NewThread>,
) -> Result<(StatusCode,Json<ThreadView>)> {
    authorize(&headers,&app)?;
    if req.model.trim().is_empty() || req.model.len() > 256 {
        return Err(error(StatusCode::BAD_REQUEST, "invalid model"));
    }
    let tid=Uuid::new_v4(); let bid=Uuid::new_v4();
    let mut tx=app.pool.begin().await.map_err(db_error)?;
    sqlx::query("INSERT INTO thread_owners(id,workspace_id) VALUES($1,$2) ON CONFLICT(id) DO NOTHING")
       .bind(app.owner).bind(app.workspace).execute(&mut *tx).await.map_err(db_error)?;
    sqlx::query("INSERT INTO threads(id,owner_id,model_id,default_branch_id) VALUES($1,$2,$3,$4)")
       .bind(tid).bind(app.owner).bind(&req.model).bind(bid)
       .execute(&mut *tx).await.map_err(db_error)?;
    sqlx::query("INSERT INTO branches(id,thread_id) VALUES($1,$2)")
       .bind(bid).bind(tid).execute(&mut *tx).await.map_err(db_error)?;
    tx.commit().await.map_err(db_error)?;
    Ok((StatusCode::CREATED,Json(ThreadView{id:tid,branch_id:bid,
       head_message_id:None,version:0,model:req.model})))
}
async fn read_thread(
    State(app): State<Arc<App>>, headers: HeaderMap, Path(tid): Path<Uuid>,
) -> Result<Json<ThreadView>> {
    authorize(&headers,&app)?;
    let row=sqlx::query(
      "SELECT t.model_id,b.id AS branch_id,b.head_message_id,b.version
       FROM threads t JOIN branches b ON b.id=t.default_branch_id AND b.thread_id=t.id
       WHERE t.id=$1 AND t.owner_id=$2 AND t.status='active'")
       .bind(tid).bind(app.owner).fetch_optional(&app.pool).await.map_err(db_error)?
       .ok_or_else(||error(StatusCode::NOT_FOUND,"thread not found"))?;
    Ok(Json(ThreadView{id:tid,branch_id:row.try_get("branch_id").map_err(db_error)?,
       head_message_id:row.try_get("head_message_id").map_err(db_error)?,
       version:row.try_get("version").map_err(db_error)?,
       model:row.try_get("model_id").map_err(db_error)?}))
}
async fn read_messages(
    State(app): State<Arc<App>>, headers: HeaderMap, Path(tid): Path<Uuid>,
) -> Result<Json<Value>> {
    authorize(&headers,&app)?;
    let row=sqlx::query(
      "SELECT b.head_message_id FROM threads t JOIN branches b
       ON b.id=t.default_branch_id AND b.thread_id=t.id
       WHERE t.id=$1 AND t.owner_id=$2 AND t.status='active'")
       .bind(tid).bind(app.owner).fetch_optional(&app.pool).await.map_err(db_error)?
       .ok_or_else(||error(StatusCode::NOT_FOUND,"thread not found"))?;
    let head:Option<Uuid>=row.try_get("head_message_id").map_err(db_error)?;
    let messages=match head {
       Some(id)=>history(&app.pool,app.owner,tid,id,100_000).await.map_err(ledger_error)?
             .into_iter().map(|(id,role,content)|json!({"id":id,"role":role,"content":content})).collect::<Vec<_>>(),
       None=>Vec::new(),
    };
    Ok(Json(json!({"object":"list","data":messages})))
}
#[derive(Deserialize)]
struct ForkBranch {
    from_message_id: Uuid,
}
#[derive(Serialize)]
struct BranchView {
    id: Uuid, thread_id: Uuid, head_message_id: Option<Uuid>, version: i64,
}
async fn create_branch(
    State(app): State<Arc<App>>, headers: HeaderMap, Path(tid): Path<Uuid>,
    Json(req): Json<ForkBranch>,
) -> Result<(StatusCode, Json<BranchView>)> {
    authorize(&headers,&app)?;
    let mut tx=app.pool.begin().await.map_err(db_error)?;
    let found:bool=sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM messages m JOIN threads t ON t.id=m.thread_id
          WHERE m.id=$1 AND m.thread_id=$2 AND t.owner_id=$3 AND t.status='active')"
    ).bind(req.from_message_id).bind(tid).bind(app.owner)
      .fetch_one(&mut *tx).await.map_err(db_error)?;
    if !found {return Err(error(StatusCode::NOT_FOUND,"fork source does not belong to this owner/thread"));}
    let bid=Uuid::new_v4();
    sqlx::query("INSERT INTO branches(id,thread_id,head_message_id,version) VALUES($1,$2,$3,0)")
        .bind(bid).bind(tid).bind(req.from_message_id)
        .execute(&mut *tx).await.map_err(db_error)?;
    tx.commit().await.map_err(db_error)?;
    Ok((StatusCode::CREATED,Json(BranchView{id:bid,thread_id:tid,
        head_message_id:Some(req.from_message_id),version:0})))
}
async fn read_branch(
    State(app): State<Arc<App>>, headers: HeaderMap, Path((tid,bid)): Path<(Uuid,Uuid)>,
) -> Result<Json<BranchView>> {
    authorize(&headers,&app)?;
    let row=sqlx::query(
      "SELECT b.head_message_id,b.version FROM branches b
       JOIN threads t ON t.id=b.thread_id
       WHERE b.thread_id=$1 AND b.id=$2 AND t.owner_id=$3 AND t.status='active'")
       .bind(tid).bind(bid).bind(app.owner).fetch_optional(&app.pool).await.map_err(db_error)?
       .ok_or_else(||error(StatusCode::NOT_FOUND,"branch not found"))?;
    Ok(Json(BranchView{id:bid,thread_id:tid,
       head_message_id:row.try_get("head_message_id").map_err(db_error)?,
       version:row.try_get("version").map_err(db_error)?}))
}
async fn read_branch_messages(
    State(app): State<Arc<App>>, headers: HeaderMap, Path((tid,bid)): Path<(Uuid,Uuid)>,
) -> Result<Json<Value>> {
    let Json(view)=read_branch(State(app.clone()),headers,Path((tid,bid))).await?;
    let messages=match view.head_message_id {
       Some(head)=>history(&app.pool,app.owner,tid,head,100_000).await.map_err(ledger_error)?
           .into_iter().map(|(id,role,content)|json!({"id":id,"role":role,"content":content})).collect::<Vec<_>>(),
       None=>Vec::new(),
    };
    Ok(Json(json!({"object":"list","data":messages,"branch_id":bid})))
}
async fn read_metrics(
    State(app): State<Arc<App>>, headers: HeaderMap, Path(tid): Path<Uuid>,
) -> Result<Json<Value>> {
    authorize(&headers,&app)?;
    let permitted:bool=sqlx::query_scalar(
       "SELECT EXISTS(SELECT 1 FROM threads WHERE id=$1 AND owner_id=$2 AND status='active')"
    ).bind(tid).bind(app.owner).fetch_one(&app.pool).await.map_err(db_error)?;
    if !permitted {return Err(error(StatusCode::NOT_FOUND,"thread not found"));}
    let rows=sqlx::query(
       "SELECT r.id AS run_id, tm.prompt_tokens,tm.reported_cached_tokens,
               tm.completion_tokens,tm.roundtrip_ms,tm.ttft_ms,tm.source
        FROM runs r JOIN turns t ON t.id=r.turn_id
        JOIN thread_run_telemetry tm ON tm.run_id=r.id
        WHERE t.thread_id=$1 ORDER BY tm.captured_at,r.id"
    ).bind(tid).fetch_all(&app.pool).await.map_err(db_error)?;
    let mut out=Vec::new();
    for row in rows {
        out.push(json!({
          "run_id":row.try_get::<Uuid,_>("run_id").map_err(db_error)?,
          "prompt_tokens":row.try_get::<Option<i64>,_>("prompt_tokens").map_err(db_error)?,
          "reported_cached_tokens":row.try_get::<Option<i64>,_>("reported_cached_tokens").map_err(db_error)?,
          "completion_tokens":row.try_get::<Option<i64>,_>("completion_tokens").map_err(db_error)?,
          "roundtrip_ms":row.try_get::<Option<i64>,_>("roundtrip_ms").map_err(db_error)?,
          "ttft_ms":row.try_get::<Option<i64>,_>("ttft_ms").map_err(db_error)?,
          "source":row.try_get::<String,_>("source").map_err(db_error)?
        }));
    }
    Ok(Json(json!({"object":"list","data":out,
       "note":"reported_cached_tokens comes from chat response usage, not independently verified native checkpoint reuse; null means unknown"})))
}

#[derive(Deserialize)]
struct NewTurn {
    branch_id: Uuid,
    expected_version: i64,
    parent_message_id: Option<Uuid>,
    idempotency_key: String,
    content: Value,
    max_tokens: Option<u32>,
}
async fn run_turn(
    State(app): State<Arc<App>>, headers: HeaderMap, Path(tid): Path<Uuid>,
    Json(req): Json<NewTurn>,
) -> Result<(StatusCode,Json<Value>)> {
    authorize(&headers,&app)?;
    if !req.content.is_string() && !req.content.is_array() {
        return Err(error(StatusCode::BAD_REQUEST,"content must be string or content parts"));
    }
    if req.max_tokens.is_some_and(|n|n==0 || n>65536) {
        return Err(error(StatusCode::BAD_REQUEST,"max_tokens outside accepted range"));
    }
    // Hash the logical request payload, not its HTTP formatting; key and branch scope
    // are defined by the unique constraint in the ledger.
    let canonical=serde_json::to_vec(&json!({
       "content":req.content,"max_tokens":req.max_tokens,
       "parent":req.parent_message_id,"version":req.expected_version
    })).map_err(|_|error(StatusCode::BAD_REQUEST,"invalid content"))?;
    let hash=request_hash(&canonical);
    let accepted=accept_turn(&app.pool,app.owner,tid,req.branch_id,
        req.parent_message_id,req.expected_version,&req.idempotency_key,hash,&req.content)
        .await.map_err(ledger_error)?;
    if accepted.existing {
        if accepted.status=="COMMITTED" {
            return Ok((StatusCode::OK,Json(json!({
                "turn_id":accepted.turn_id,"status":"COMMITTED",
                "message_id":accepted.committed_message_id,"replayed":true}))));
        }
        if accepted.status != "RECOVERABLE" {
            return Ok((StatusCode::ACCEPTED,Json(json!({
                "turn_id":accepted.turn_id,"status":accepted.status,"replayed":true}))));
        }
    }
    let model_row=sqlx::query("SELECT model_id FROM threads WHERE id=$1 AND owner_id=$2")
       .bind(tid).bind(app.owner).fetch_one(&app.pool).await.map_err(db_error)?;
    let model:String=model_row.try_get("model_id").map_err(db_error)?;
    // This is a stable run label, NOT a verified model/weights/cache fingerprint.
    // No cache reuse is authorized by this value until the engine publishes its digest.
    let model_label: [u8;32] = Sha256::digest(model.as_bytes()).into();
    let (run_id, token) = if accepted.existing {
        let lease=claim_recovery(&app.pool,accepted.turn_id,model_label)
            .await.map_err(recovery_error)?;
        (lease.run_id,lease.fencing_token)
    } else {
        let run_id=Uuid::new_v4();
        let mut tx=app.pool.begin().await.map_err(db_error)?;
        let token:i64=sqlx::query_scalar("SELECT nextval('basalt_thread_fencing_seq')")
            .fetch_one(&mut *tx).await.map_err(db_error)?;
        sqlx::query("UPDATE turns SET status='RUNNING' WHERE id=$1 AND status='ACCEPTED'")
            .bind(accepted.turn_id).execute(&mut *tx).await.map_err(db_error)?;
        sqlx::query(
            "INSERT INTO runs(id,turn_id,attempt,state,lease_token,model_fingerprint,started_at,lease_expires_at)
             VALUES($1,$2,1,'RUNNING',$3,$4,clock_timestamp(),clock_timestamp()+interval '1 hour')")
            .bind(run_id).bind(accepted.turn_id).bind(token).bind(model_label.as_slice())
            .execute(&mut *tx).await.map_err(db_error)?;
        tx.commit().await.map_err(db_error)?;
        (run_id,token)
    };

    let mut messages=match req.parent_message_id {
       Some(head)=>history(&app.pool,app.owner,tid,head,100_000).await
           .map_err(ledger_error)?.into_iter().map(|(_,role,content)|json!({"role":role,"content":content}))
           .collect::<Vec<_>>(),
       None=>Vec::new(),
    };
    messages.push(json!({"role":"user","content":req.content}));
    let (stop_tx,mut stop_rx)=tokio::sync::oneshot::channel::<()>();
    let heartbeat_pool=app.pool.clone();
    let turn_id=accepted.turn_id;
    let heart_task=tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                _ = tokio::time::sleep(Duration::from_secs(60)) => {
                    match heartbeat(&heartbeat_pool,turn_id,run_id,token).await {
                        Ok(true) => {},
                        _ => break,
                    }
                }
            }
        }
    });
    let inference_started=Instant::now();
    let upstream=app.client.post(format!("{}/v1/chat/completions",app.upstream))
         .json(&json!({"model":model,"messages":messages,"stream":false,
                      "max_tokens":req.max_tokens.unwrap_or(1024)}))
         .send().await;
    let _=stop_tx.send(());
    let _=heart_task.await;
    let result=match upstream {
       Ok(x) if x.status().is_success()=>x.json::<Value>().await.ok(),
       _=>None,
    };
    // The first synchronous slice intentionally accepts only plain text output.
    // Do not silently discard tool calls or reasoning payloads.
    let reply=result.as_ref().and_then(|v|v.pointer("/choices/0/message"))
      .filter(|m|m.get("tool_calls").is_none() && m.get("reasoning_content").is_none())
      .and_then(|m|m.get("content")).filter(|v|v.is_string()).cloned();
    let Some(reply)=reply else {
        sqlx::query("UPDATE turns SET status='RECOVERABLE' WHERE id=$1 AND status='RUNNING'")
            .bind(accepted.turn_id).execute(&app.pool).await.map_err(db_error)?;
        sqlx::query("UPDATE runs SET state='RECOVERABLE' WHERE id=$1 AND state='RUNNING'")
            .bind(run_id).execute(&app.pool).await.map_err(db_error)?;
        return Err(error(StatusCode::BAD_GATEWAY,"inference failed; turn remains recoverable"));
    };
    let mut tx=app.pool.begin().await.map_err(db_error)?;
    sqlx::query("UPDATE turns SET status='FINALIZING' WHERE id=$1 AND status='RUNNING'")
       .bind(accepted.turn_id).execute(&mut *tx).await.map_err(db_error)?;
    sqlx::query("UPDATE runs SET state='FINALIZING' WHERE id=$1 AND state='RUNNING'")
       .bind(run_id).execute(&mut *tx).await.map_err(db_error)?;
    tx.commit().await.map_err(db_error)?;
    // Commit telemetry best-effort. A metrics write must never turn a successfully
    // committed message into a client-visible retry failure.
    let usage=result.as_ref().and_then(|v|v.get("usage"));
    let prompt=usage.and_then(|v|v.get("prompt_tokens")).and_then(Value::as_i64);
    let reported=usage.and_then(|v|v.pointer("/prompt_tokens_details/cached_tokens"))
        .and_then(Value::as_i64);
    let completion=usage.and_then(|v|v.get("completion_tokens")).and_then(Value::as_i64);
    let measured_roundtrip_ms=i64::try_from(inference_started.elapsed().as_millis()).unwrap_or(i64::MAX);
    let message_id=commit_turn(&app.pool,accepted.turn_id,run_id,token,
       Uuid::new_v4(),Uuid::new_v4(),&reply,Some(&model)).await.map_err(ledger_error)?;
    if let Err(e)=sqlx::query(
         "INSERT INTO thread_run_telemetry(run_id,prompt_tokens,reported_cached_tokens,
              completion_tokens,roundtrip_ms,ttft_ms,source)
          VALUES($1,$2,$3,$4,$5,NULL,'basalt.chat_completion.usage')
          ON CONFLICT(run_id) DO NOTHING")
         .bind(run_id).bind(prompt).bind(reported).bind(completion)
         .bind(measured_roundtrip_ms).execute(&app.pool).await {
        eprintln!("thread telemetry capture failed: {e}");
    }
    Ok((StatusCode::OK,Json(json!({"turn_id":accepted.turn_id,"status":"COMMITTED",
       "message_id":message_id,"response":reply}))))
}
#[tokio::main]
async fn main() -> std::result::Result<(),Box<dyn std::error::Error>> {
    let secret=env::var("BASALT_THREAD_API_KEY")?;
    if secret.len()<24 {return Err("BASALT_THREAD_API_KEY must have at least 24 characters".into());}
    let database=env::var("DATABASE_URL")?;
    let owner=env::var("BASALT_THREAD_OWNER_ID")?.parse()?;
    let workspace=env::var("BASALT_THREAD_WORKSPACE_ID")?.parse()?;
    let upstream=env::var("BASALT_UPSTREAM_URL").unwrap_or_else(|_|"http://127.0.0.1:8080".into());
    if !upstream.starts_with("http://127.0.0.1:") {
       return Err("upstream must be bound to 127.0.0.1".into());
    }
    let pool=PgPool::connect(&database).await?;
    sqlx::migrate!("../migrations").run(&pool).await?;
    let client=reqwest::Client::builder().timeout(Duration::from_secs(3600)).build()?;
    let app=Arc::new(App{pool,client,secret,owner,workspace,upstream});
    let recover_pool=app.pool.clone();
    tokio::spawn(async move {
        loop {
            if let Err(e)=reconcile_expired(&recover_pool,128).await {
                eprintln!("thread recovery sweep failed: {e:?}");
            }
            tokio::time::sleep(Duration::from_secs(15)).await;
        }
    });
    let engine_epoch=env::var("BASALT_ENGINE_EPOCH").ok().map(|v|v.parse::<Uuid>()).transpose()?;
    contexts::spawn_sweeper(app.pool.clone(),engine_epoch);
    let router=Router::new()
       .route("/v1/threads",post(create_thread))
       .route("/v1/threads/{id}",get(read_thread))
       .route("/v1/threads/{id}/messages",get(read_messages))
       .route("/v1/threads/{id}/metrics",get(read_metrics))
       .route("/v1/threads/{id}/branches",post(create_branch))
       .route("/v1/threads/{id}/branches/{branch}",get(read_branch))
       .route("/v1/threads/{id}/branches/{branch}/messages",get(read_branch_messages))
       .route("/v1/threads/{id}/turns",post(run_turn))
       .merge(contexts::router())
       .with_state(app);
    // The port is configurable because the loopback range is shared with other local services;
    // the address stays loopback-only unless the operator points it at a private interface
    // (BASALT_THREAD_API_HOST), e.g. the docker bridge gateway when serve runs in a container.
    let port:u16=match env::var("BASALT_THREAD_API_PORT"){
        Ok(p)=>p.parse().map_err(|_|format!("BASALT_THREAD_API_PORT is not a port: {p}"))?,
        Err(_)=>"8095".parse()?,
    };
    let host:std::net::Ipv4Addr=match env::var("BASALT_THREAD_API_HOST"){
        Ok(h)=>h.parse().map_err(|_|format!("BASALT_THREAD_API_HOST is not an IPv4 address: {h}"))?,
        Err(_)=>std::net::Ipv4Addr::LOCALHOST,
    };
    let addr:SocketAddr=(host, port).into();
    let listener=tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener,router).await?;
    Ok(())
}
