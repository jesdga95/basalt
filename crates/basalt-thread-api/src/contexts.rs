//! `/v1/contexts*` routes (docs/architecture/agent-context-contract.md). Tenant is the
//! sidecar owner; foreign and unknown contexts are both 404. Errors: {error:{code,message}}.
use crate::{authorize, App};
use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use basalt_context_registry as reg;
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{json, Value};
use sqlx::PgPool;
use std::{sync::Arc, time::Duration};
use uuid::Uuid;

type ApiError = (StatusCode, Json<Value>);
type ApiResult<T> = std::result::Result<T, ApiError>;

fn err(status: StatusCode, code: &str, message: impl Into<String>) -> ApiError {
    (status, Json(json!({"error": {"code": code, "message": message.into()}})))
}
fn registry_error(e: reg::RegistryError) -> ApiError {
    match e {
        reg::RegistryError::NotFound => err(StatusCode::NOT_FOUND, "not_found", "context not found"),
        reg::RegistryError::Conflict(c) => err(StatusCode::CONFLICT, c.code(), c.message()),
        reg::RegistryError::Invalid(m) => err(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", m),
        reg::RegistryError::Database(e) => {
            eprintln!("context database operation failed: {e}");
            err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", "context database error")
        }
    }
}
fn context_id(raw: &str) -> ApiResult<Uuid> {
    raw.parse().map_err(|_| err(StatusCode::NOT_FOUND, "not_found", "context not found"))
}
fn parse_body<T: DeserializeOwned>(body: &Bytes) -> ApiResult<T> {
    serde_json::from_slice(body)
        .map_err(|e| err(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", format!("invalid body: {e}")))
}
fn parse_optional_body<T: DeserializeOwned + Default>(body: &Bytes) -> ApiResult<T> {
    if body.iter().all(u8::is_ascii_whitespace) { Ok(T::default()) } else { parse_body(body) }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateReq { thread_id: Uuid, branch_id: Option<Uuid>, upto_message_id: Option<Uuid> }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForkReq { expected_version: i64 }
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct ResumeReq { lease_ttl_ms: Option<i64> }
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct EmptyReq {}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleaseReq { token: i64 }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransitionReq {
    event: String,
    expected_version: i64,
    lease_token: Option<i64>,
    tier: Option<String>,
    locator: Option<String>,
    validated: Option<bool>,
    model_fingerprint: Option<String>,
    prefix_hash: Option<String>,
    prefix_tokens: Option<i64>,
    engine_epoch: Option<Uuid>,
}

fn summary(c: &reg::Context) -> Value {
    json!({
        "context_id": c.id, "state": c.state.as_str(), "version": c.version,
        "model_fingerprint": reg::hex(&c.model_fingerprint), "prefix_hash": reg::hex(&c.prefix_hash),
        "prefix_tokens": c.prefix_tokens, "tier": c.tier.as_str(),
    })
}

async fn create(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes)
    -> ApiResult<(StatusCode, Json<Value>)> {
    authorize(&headers, &app)?;
    let req: CreateReq = parse_body(&body)?;
    let c = reg::create_context(&app.pool, app.owner, req.thread_id, req.branch_id, req.upto_message_id)
        .await.map_err(registry_error)?;
    Ok((StatusCode::CREATED, Json(summary(&c))))
}

async fn fork(State(app): State<Arc<App>>, headers: HeaderMap, Path(id): Path<String>, body: Bytes)
    -> ApiResult<(StatusCode, Json<Value>)> {
    authorize(&headers, &app)?;
    let id = context_id(&id)?;
    let key = headers.get("idempotency-key").and_then(|v| v.to_str().ok())
        .ok_or_else(|| err(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", "Idempotency-Key header is required"))?;
    let req: ForkReq = parse_body(&body)?;
    let (child, replayed) = reg::fork_context(&app.pool, app.owner, id, req.expected_version, key)
        .await.map_err(registry_error)?;
    let status = if replayed { StatusCode::OK } else { StatusCode::CREATED };
    Ok((status, Json(json!({
        "context_id": child.id, "parent_id": child.parent_id,
        "state": child.state.as_str(), "version": child.version,
    }))))
}

async fn resume(State(app): State<Arc<App>>, headers: HeaderMap, Path(id): Path<String>, body: Bytes)
    -> ApiResult<Json<Value>> {
    authorize(&headers, &app)?;
    let id = context_id(&id)?;
    let req: ResumeReq = parse_optional_body(&body)?;
    let (_, lease, source) = reg::acquire_lease(&app.pool, app.owner, id, req.lease_ttl_ms)
        .await.map_err(registry_error)?;
    Ok(Json(json!({
        "context_id": id, "source": source,
        "lease": {"token": lease.token, "expires_at": lease.expires_at},
    })))
}

async fn release(State(app): State<Arc<App>>, headers: HeaderMap, Path(id): Path<String>, body: Bytes)
    -> ApiResult<Json<Value>> {
    authorize(&headers, &app)?;
    let id = context_id(&id)?;
    let req: ReleaseReq = parse_body(&body)?;
    reg::release_lease(&app.pool, app.owner, id, req.token).await.map_err(registry_error)?;
    Ok(Json(json!({"context_id": id, "released": true})))
}

async fn transition(State(app): State<Arc<App>>, headers: HeaderMap, Path(id): Path<String>, body: Bytes)
    -> ApiResult<Json<Value>> {
    authorize(&headers, &app)?;
    let id = context_id(&id)?;
    let req: TransitionReq = parse_body(&body)?;
    let bad = |m: &str| err(StatusCode::UNPROCESSABLE_ENTITY, "invalid_request", m.to_string());
    let event = reg::Event::parse(&req.event).ok_or_else(|| bad("unknown event"))?;
    let tier = match req.tier.as_deref() {
        None => None,
        Some(t) => Some(reg::Tier::parse(t).ok_or_else(|| bad("unknown tier"))?),
    };
    let digest = |v: &Option<String>, name: &str| -> ApiResult<Option<reg::Digest32>> {
        match v {
            None => Ok(None),
            Some(s) => reg::parse_hex32(s).map(Some).ok_or_else(|| bad(&format!("{name} must be 64 hex characters"))),
        }
    };
    let patch = reg::Patch {
        tier, locator: req.locator, validated: req.validated,
        model_fingerprint: digest(&req.model_fingerprint, "model_fingerprint")?,
        prefix_hash: digest(&req.prefix_hash, "prefix_hash")?,
        prefix_tokens: req.prefix_tokens, engine_epoch: req.engine_epoch,
    };
    let c = reg::transition(&app.pool, app.owner, id, reg::TransitionRequest {
        event, expected_version: req.expected_version, lease_token: req.lease_token, patch,
    }).await.map_err(registry_error)?;
    let mut out = summary(&c);
    out["validated"] = json!(c.validated);
    Ok(Json(out))
}

async fn read(State(app): State<Arc<App>>, headers: HeaderMap, Path(id): Path<String>)
    -> ApiResult<Json<Value>> {
    authorize(&headers, &app)?;
    let id = context_id(&id)?;
    let v = reg::get_context(&app.pool, app.owner, id).await.map_err(registry_error)?;
    let c = &v.context;
    Ok(Json(json!({
        "context_id": c.id, "thread_id": c.thread_id, "branch_id": c.branch_id,
        "parent_id": c.parent_id, "state": c.state.as_str(), "tier": c.tier.as_str(),
        "version": c.version, "validated": c.validated,
        "model_fingerprint": reg::hex(&c.model_fingerprint), "prefix_hash": reg::hex(&c.prefix_hash),
        "position": {"prefix_tokens": c.prefix_tokens, "head_message_id": c.head_message_id},
        "locator": c.locator,
        "lineage": v.lineage.iter().map(|l| json!({
            "context_id": l.id, "state": l.state.as_str(), "version": l.version,
            "tombstoned": l.tombstoned})).collect::<Vec<_>>(),
        "children": v.children,
        "refs": {"children": v.child_refs, "leases": v.active_leases},
    })))
}

async fn delete(State(app): State<Arc<App>>, headers: HeaderMap, Path(id): Path<String>)
    -> ApiResult<(StatusCode, Json<Value>)> {
    authorize(&headers, &app)?;
    let id = context_id(&id)?;
    reg::tombstone(&app.pool, app.owner, id).await.map_err(registry_error)?;
    Ok((StatusCode::ACCEPTED, Json(json!({"context_id": id, "deleted": true}))))
}

async fn payload_released(State(app): State<Arc<App>>, headers: HeaderMap, Path(id): Path<String>, body: Bytes)
    -> ApiResult<Json<Value>> {
    authorize(&headers, &app)?;
    let id = context_id(&id)?;
    parse_optional_body::<EmptyReq>(&body)?;
    let c = reg::release_payload(&app.pool, app.owner, id).await.map_err(registry_error)?;
    Ok(Json(json!({"context_id": id, "state": c.state.as_str(), "tier": c.tier.as_str(),
                   "locator": c.locator, "version": c.version})))
}

pub fn router() -> Router<Arc<App>> {
    Router::new()
        .route("/v1/contexts", post(create))
        .route("/v1/contexts/{id}", get(read).delete(delete))
        .route("/v1/contexts/{id}/fork", post(fork))
        .route("/v1/contexts/{id}/resume", post(resume))
        .route("/v1/contexts/{id}/lease/release", post(release))
        .route("/v1/contexts/{id}/transition", post(transition))
        .route("/v1/contexts/{id}/payload/released", post(payload_released))
}

/// Restart-safe sweeper: expire leases, rebuild orphans and collect tombstones that own
/// no external payload. Contexts that still point at a snapshot wait for the engine.
pub fn spawn_sweeper(pool: PgPool, engine_epoch: Option<Uuid>) {
    tokio::spawn(async move {
        loop {
            if let Err(e) = reg::reconcile(&pool, engine_epoch, 256).await {
                eprintln!("context reconcile failed: {e:?}");
            }
            match reg::gc_candidates(&pool, 256).await {
                Ok(c) => {
                    let ids: Vec<Uuid> = c.into_iter()
                        .filter(|x| x.tier == reg::Tier::None && x.locator.is_none())
                        .map(|x| x.id).collect();
                    if let Err(e) = reg::gc_purge(&pool, &ids).await {
                        eprintln!("context gc failed: {e:?}");
                    }
                }
                Err(e) => eprintln!("context gc scan failed: {e:?}"),
            }
            tokio::time::sleep(Duration::from_secs(15)).await;
        }
    });
}
