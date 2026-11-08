use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::fsm::State;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchRef {
    pub thread_id: Uuid,
    pub branch_id: Uuid,
    pub head_message_id: Option<Uuid>,
    pub version: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub parent_id: Option<Uuid>,
    pub role: String,
    pub content: serde_json::Value,
    pub content_hash: [u8; 32],
    pub model_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    pub id: Uuid,
    pub thread_id: Uuid,
    pub branch_id: Uuid,
    pub parent_message_id: Option<Uuid>,
    pub idempotency_key: String,
    pub request_hash: [u8; 32],
    pub state: State,
    pub committed_message_id: Option<Uuid>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub id: Uuid,
    pub turn_id: Uuid,
    pub attempt: i32,
    pub state: State,
    pub lease_token: i64,
    pub model_fingerprint: [u8; 32],
}
