//! Basalt thread contracts. The PostgreSQL message ledger is authoritative; KV is derived.
pub mod fsm;
pub mod model;
pub use fsm::{Event, State, TransitionError};
pub use model::{BranchRef, Message, Run, Turn};
