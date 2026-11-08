//! Loopback-only API sidecar. Basalt's existing HTTP handler authenticates and forwards
//! thread calls; this process has a separate mandatory shared secret.
use axum::{
    extract::{Path, State}, http::{HeaderMap, StatusCode}, routing::{get, post},
    Json, Router,
};
use basalt_thread_store::{accept_turn, commit_turn, history, request_hash, LedgerError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};
use std::{env, net::SocketAddr, sync::Arc, time::Duration};
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
        return Ok((StatusCode::ACCEPTED,Json(json!({
            "turn_id":accepted.turn_id,"status":accepted.status,"replayed":true}))));
    }
    let run_id=Uuid::new_v4();
    // Queue and claim in one transaction. A future scheduler can replace this
    // synchronous runner without changing accepted-turn/idempotency semantics.
    let mut tx=app.pool.begin().await.map_err(db_error)?;
    let token:i64=sqlx::query_scalar("SELECT nextval('basalt_thread_fencing_seq')")
        .fetch_one(&mut *tx).await.map_err(db_error)?;
    sqlx::query("UPDATE turns SET status='RUNNING' WHERE id=$1 AND status='ACCEPTED'")
        .bind(accepted.turn_id).execute(&mut *tx).await.map_err(db_error)?;
    sqlx::query(
        "INSERT INTO runs(id,turn_id,attempt,state,lease_token,model_fingerprint,started_at,lease_expires_at)
         VALUES($1,$2,1,'RUNNING',$3,$4,clock_timestamp(),clock_timestamp()+interval '1 hour')")
        .bind(run_id).bind(accepted.turn_id).bind(token).bind([0u8;32].as_slice())
        .execute(&mut *tx).await.map_err(db_error)?;
    tx.commit().await.map_err(db_error)?;

    let model_row=sqlx::query("SELECT model_id FROM threads WHERE id=$1 AND owner_id=$2")
       .bind(tid).bind(app.owner).fetch_one(&app.pool).await.map_err(db_error)?;
    let model:String=model_row.try_get("model_id").map_err(db_error)?;
    let mut messages=match req.parent_message_id {
       Some(head)=>history(&app.pool,app.owner,tid,head,100_000).await
           .map_err(ledger_error)?.into_iter().map(|(_,role,content)|json!({"role":role,"content":content}))
           .collect::<Vec<_>>(),
       None=>Vec::new(),
    };
    messages.push(json!({"role":"user","content":req.content}));
    let upstream=app.client.post(format!("{}/v1/chat/completions",app.upstream))
         .json(&json!({"model":model,"messages":messages,"stream":false,
                      "max_tokens":req.max_tokens.unwrap_or(1024)}))
         .send().await;
    let result=match upstream {
       Ok(x) if x.status().is_success()=>x.json::<Value>().await.ok(),
       _=>None,
    };
    let reply=result.as_ref().and_then(|v|v.pointer("/choices/0/message"))
      .cloned();
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
    let message_id=commit_turn(&app.pool,accepted.turn_id,run_id,token,
       Uuid::new_v4(),Uuid::new_v4(),&reply,Some(&model)).await.map_err(ledger_error)?;
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
    let router=Router::new()
       .route("/v1/threads",post(create_thread))
       .route("/v1/threads/{id}",get(read_thread))
       .route("/v1/threads/{id}/messages",get(read_messages))
       .route("/v1/threads/{id}/turns",post(run_turn))
       .with_state(app);
    let addr:SocketAddr="127.0.0.1:8095".parse()?;
    let listener=tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener,router).await?;
    Ok(())
}
