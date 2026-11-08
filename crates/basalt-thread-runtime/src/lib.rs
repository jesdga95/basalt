//! Durable run leasing and recovery. Never infer a successful commit from an
//! inference attempt, and never reuse a worker's old fencing token.
use sqlx::{PgPool, Row};
use uuid::Uuid;

#[derive(Debug)]
pub enum RecoveryError {
    Conflict,
    UnsafeToolState,
    Database(sqlx::Error),
}
impl From<sqlx::Error> for RecoveryError {
    fn from(value: sqlx::Error) -> Self { Self::Database(value) }
}

#[derive(Debug, Clone, Copy)]
pub struct Lease {
    pub run_id: Uuid,
    pub fencing_token: i64,
    pub attempt: i32,
}

/// Reconcile expired latest runs. Lock order is turns then runs, identical to
/// commit_thread_exchange. SKIP LOCKED prevents blocking a concurrent commit.
pub async fn reconcile_expired(db: &PgPool, batch: i64) -> Result<usize, RecoveryError> {
    if batch < 1 || batch > 1_000 { return Err(RecoveryError::Conflict); }
    let mut tx = db.begin().await?;
    let rows = sqlx::query(
        "SELECT t.id AS turn_id, r.id AS run_id FROM turns t
         JOIN runs r ON r.turn_id=t.id
         WHERE t.status IN ('RUNNING','WAITING_TOOL','FINALIZING')
           AND r.state IN ('RUNNING','WAITING_TOOL','FINALIZING')
           AND r.lease_expires_at <= clock_timestamp()
           AND NOT EXISTS (SELECT 1 FROM runs newer
                           WHERE newer.turn_id=t.id AND newer.lease_token>r.lease_token)
         ORDER BY t.created_at
         LIMIT $1 FOR UPDATE OF t SKIP LOCKED"
    ).bind(batch).fetch_all(&mut *tx).await?;
    for row in &rows {
        let turn:Uuid=row.try_get("turn_id")?;
        let run:Uuid=row.try_get("run_id")?;
        sqlx::query("UPDATE runs SET state='RECOVERABLE'
                     WHERE id=$1 AND state IN ('RUNNING','WAITING_TOOL','FINALIZING')")
            .bind(run).execute(&mut *tx).await?;
        sqlx::query("UPDATE turns SET status='RECOVERABLE'
                     WHERE id=$1 AND status IN ('RUNNING','WAITING_TOOL','FINALIZING')")
            .bind(turn).execute(&mut *tx).await?;
    }
    tx.commit().await?;
    Ok(rows.len())
}

/// A retry may reexecute GPU inference, but the logical input/turn remains one.
/// Refuse automatic retry if any tool may have produced external side effects.
pub async fn claim_recovery(
    db: &PgPool, turn_id: Uuid, model_label: [u8; 32],
) -> Result<Lease, RecoveryError> {
    let mut tx=db.begin().await?;
    let row=sqlx::query("SELECT status FROM turns WHERE id=$1 FOR UPDATE")
        .bind(turn_id).fetch_optional(&mut *tx).await?;
    let row=row.ok_or(RecoveryError::Conflict)?;
    let state:String=row.try_get("status")?;
    if state!="RECOVERABLE" {return Err(RecoveryError::Conflict);}
    let uncertain_tool:bool=sqlx::query_scalar(
       "SELECT EXISTS(SELECT 1 FROM tool_executions te
                      JOIN runs r ON r.id=te.run_id
                      WHERE r.turn_id=$1 AND te.status IN ('APPROVED','RUNNING','SUCCEEDED','UNKNOWN'))"
    ).bind(turn_id).fetch_one(&mut *tx).await?;
    if uncertain_tool {return Err(RecoveryError::UnsafeToolState);}
    let attempt:i32=sqlx::query_scalar(
        "SELECT COALESCE(MAX(attempt),0)+1 FROM runs WHERE turn_id=$1"
    ).bind(turn_id).fetch_one(&mut *tx).await?;
    let fencing:i64=sqlx::query_scalar("SELECT nextval('basalt_thread_fencing_seq')")
        .fetch_one(&mut *tx).await?;
    let run_id=Uuid::new_v4();
    sqlx::query(
        "INSERT INTO runs(id,turn_id,attempt,state,lease_token,model_fingerprint,
                          started_at,lease_expires_at)
         VALUES($1,$2,$3,'RUNNING',$4,$5,clock_timestamp(),clock_timestamp()+interval '1 hour')"
    ).bind(run_id).bind(turn_id).bind(attempt).bind(fencing)
        .bind(model_label.as_slice()).execute(&mut *tx).await?;
    sqlx::query("UPDATE turns SET status='RUNNING' WHERE id=$1")
       .bind(turn_id).execute(&mut *tx).await?;
    tx.commit().await?;
    Ok(Lease{run_id,fencing_token:fencing,attempt})
}

/// Extend only the current active attempt. A stale worker cannot revive its lease.
pub async fn heartbeat(
    db: &PgPool, turn_id: Uuid, run_id: Uuid, fencing: i64,
) -> Result<bool,RecoveryError> {
    let rows=sqlx::query(
        "UPDATE runs r SET lease_expires_at=clock_timestamp()+interval '1 hour'
         WHERE r.id=$1 AND r.turn_id=$2 AND r.lease_token=$3
           AND r.state='RUNNING' AND r.lease_expires_at > clock_timestamp()
           AND NOT EXISTS(SELECT 1 FROM runs newer WHERE newer.turn_id=r.turn_id
                          AND newer.lease_token>r.lease_token)"
    ).bind(run_id).bind(turn_id).bind(fencing).execute(db).await?.rows_affected();
    Ok(rows==1)
}
