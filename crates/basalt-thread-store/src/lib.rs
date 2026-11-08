//! PostgreSQL is the source of truth. Do not use an in-memory message index for correctness.
use sha2::{Digest, Sha256};
use sqlx::{PgPool, Row};
use uuid::Uuid;

#[derive(Debug)]
pub enum LedgerError {
    Conflict,
    Missing,
    Database(sqlx::Error),
}
impl From<sqlx::Error> for LedgerError {
    fn from(value: sqlx::Error) -> Self {
        match &value {
            sqlx::Error::Database(e) if e.code().as_deref()==Some("40001") => Self::Conflict,
            _ => Self::Database(value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcceptedTurn {
    pub turn_id: Uuid,
    pub existing: bool,
    pub status: String,
    pub committed_message_id: Option<Uuid>,
}

/// Request hashing must use the canonical raw request bytes supplied by the API boundary.
/// A single idempotency key with different content is never silently reused.
pub fn request_hash(canonical_bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(canonical_bytes).into()
}

/// Accept a turn atomically against a specific branch HEAD. Owner scope is mandatory.
/// Exact key+hash retries return the original turn without advancing the branch.
/// Parallel new requests against the same HEAD are accepted but only one can CAS commit.
pub async fn accept_turn(
    db: &PgPool, owner: Uuid, thread: Uuid, branch: Uuid,
    parent: Option<Uuid>, expected_version: i64,
    key: &str, hash: [u8;32], input: &serde_json::Value,
) -> Result<AcceptedTurn, LedgerError> {
    if key.is_empty() || key.len() > 256 || expected_version < 0 {
        return Err(LedgerError::Conflict);
    }
    let mut tx = db.begin().await?;
    let exists = sqlx::query(
        "SELECT t.id, t.request_hash, t.status, t.committed_message_id
         FROM turns t JOIN threads h ON h.id=t.thread_id
         WHERE t.thread_id=$1 AND t.branch_id=$2 AND t.idempotency_key=$3
           AND h.owner_id=$4"
    ).bind(thread).bind(branch).bind(key).bind(owner)
        .fetch_optional(&mut *tx).await?;
    if let Some(row) = exists {
        let old: Vec<u8> = row.try_get("request_hash")?;
        if old != hash { return Err(LedgerError::Conflict); }
        let out = AcceptedTurn {
            turn_id: row.try_get("id")?, existing: true,
            status: row.try_get("status")?,
            committed_message_id: row.try_get("committed_message_id")?,
        };
        tx.commit().await?;
        return Ok(out);
    }
    let found = sqlx::query(
        "SELECT b.version, b.head_message_id FROM branches b
         JOIN threads t ON t.id=b.thread_id
         WHERE b.thread_id=$1 AND b.id=$2 AND t.owner_id=$3 FOR UPDATE OF b"
    ).bind(thread).bind(branch).bind(owner).fetch_optional(&mut *tx).await?;
    let row = found.ok_or(LedgerError::Missing)?;
    let version: i64 = row.try_get("version")?;
    let head: Option<Uuid> = row.try_get("head_message_id")?;
    if version != expected_version || head != parent {
        return Err(LedgerError::Conflict);
    }
    let new_id = Uuid::new_v4();
    // Concurrent identical-key inserts can race. ON CONFLICT lets the winner's immutable
    // request hash be checked under the unique constraint.
    let inserted = sqlx::query(
        "INSERT INTO turns(id,thread_id,branch_id,parent_message_id,idempotency_key,
              request_hash,status,expected_branch_version)
         VALUES($1,$2,$3,$4,$5,$6,'ACCEPTED',$7)
         ON CONFLICT(thread_id,branch_id,idempotency_key) DO NOTHING
         RETURNING id"
    ).bind(new_id).bind(thread).bind(branch).bind(parent)
        .bind(key).bind(hash.as_slice()).bind(expected_version)
        .fetch_optional(&mut *tx).await?;
    let result = if inserted.is_some() {
        AcceptedTurn {turn_id: new_id, existing:false, status:"ACCEPTED".into(), committed_message_id:None}
    } else {
        let row = sqlx::query(
            "SELECT id,request_hash,status,committed_message_id FROM turns
             WHERE thread_id=$1 AND branch_id=$2 AND idempotency_key=$3"
        ).bind(thread).bind(branch).bind(key).fetch_one(&mut *tx).await?;
        let old: Vec<u8> = row.try_get("request_hash")?;
        if old != hash { return Err(LedgerError::Conflict); }
        AcceptedTurn {turn_id:row.try_get("id")?, existing:true,
            status:row.try_get("status")?, committed_message_id:row.try_get("committed_message_id")?}
    };
    if !result.existing {
        let encoded = serde_json::to_vec(input).map_err(|_| LedgerError::Conflict)?;
        let input_digest = request_hash(&encoded);
        sqlx::query("INSERT INTO turn_inputs(turn_id,role,content,content_hash)
                     VALUES($1,'user',$2,$3)")
            .bind(new_id).bind(input).bind(input_digest.as_slice())
            .execute(&mut *tx).await?;
    }
    tx.commit().await?;
    Ok(result)
}

/// The database function checks the current lease, fencing token and branch CAS.
/// A failed function call rolls back both message insertion and HEAD movement.
pub async fn commit_turn(
    db: &PgPool, turn: Uuid, run: Uuid, fencing: i64,
    user_message: Uuid, assistant_message: Uuid,
    content: &serde_json::Value, model: Option<&str>,
) -> Result<Uuid, LedgerError> {
    let serialized = serde_json::to_vec(content).map_err(|_| LedgerError::Conflict)?;
    let digest = request_hash(&serialized);
    let row = sqlx::query_scalar::<_, Uuid>(
        "SELECT commit_thread_exchange($1,$2,$3,$4,$5,$6,$7,$8)"
    ).bind(turn).bind(run).bind(fencing).bind(user_message).bind(assistant_message)
        .bind(content).bind(digest.as_slice()).bind(model)
        .fetch_one(db).await?;
    Ok(row)
}

/// Rebuild a linear prompt by walking immutable parent links from the expected HEAD.
/// Never use created_at to determine causal order. Enforce tenant access on every walk.
pub async fn history(
    db: &PgPool, owner: Uuid, thread: Uuid, head: Uuid, limit: i64,
) -> Result<Vec<(Uuid,String,serde_json::Value)>,LedgerError> {
    if !(1..=100_000).contains(&limit) { return Err(LedgerError::Conflict); }
    let rows = sqlx::query(
        "WITH RECURSIVE chain AS (
           SELECT m.id,m.parent_id,m.role,m.content,1::bigint AS depth
           FROM messages m JOIN threads t ON t.id=m.thread_id
           WHERE m.id=$1 AND m.thread_id=$2 AND t.owner_id=$3
           UNION ALL
           SELECT p.id,p.parent_id,p.role,p.content,c.depth+1
           FROM messages p JOIN chain c ON p.id=c.parent_id
           WHERE p.thread_id=$2 AND c.depth < $4
         )
         SELECT id,role,content FROM chain ORDER BY depth DESC"
    ).bind(head).bind(thread).bind(owner).bind(limit)
       .fetch_all(db).await?;
    if rows.is_empty() { return Err(LedgerError::Missing); }
    rows.into_iter().map(|r| Ok((r.try_get("id")?, r.try_get("role")?,
            r.try_get("content")?))).collect()
}
