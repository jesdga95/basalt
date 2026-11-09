//! Context registry: see docs/architecture/agent-context-contract.md
//!
//! `fsm` is pure and table-tested; `store` applies it to PostgreSQL under CAS on
//! `version` and the fenced writer lease. The transcript ledger stays authoritative.
pub mod fsm;
pub mod store;

pub use fsm::{recovery_plan, tier_after, Event, LeaseRule, State, Tier, TransitionError};
pub use store::*;
