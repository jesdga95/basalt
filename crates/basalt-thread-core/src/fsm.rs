use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum State {
    Accepted, Queued, Running, WaitingTool, Finalizing, Committed,
    Failed, Cancelled, Recoverable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Enqueue, Start, WaitForTool, ResumeTool, FinishInference, Commit,
    Fail, Cancel, LoseLease, Reconcile, Retry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionError { pub state: State, pub event: Event }

impl State {
    pub fn transition(self, event: Event) -> Result<Self, TransitionError> {
        use Event::*;
        use State::*;
        let next = match (self, event) {
            (Accepted, Enqueue) => Queued,
            (Queued, Start) => Running,
            (Running, WaitForTool) => WaitingTool,
            (WaitingTool, ResumeTool) => Queued,
            (Running, FinishInference) => Finalizing,
            (Finalizing, Commit) => Committed,
            (Running | WaitingTool | Finalizing, LoseLease) => Recoverable,
            (Recoverable, Reconcile) => Finalizing,
            (Recoverable, Retry) => Queued,
            (Accepted | Queued | Running | WaitingTool | Recoverable, Fail) => Failed,
            (Accepted | Queued | Running | WaitingTool, Cancel) => Cancelled,
            _ => return Err(TransitionError { state: self, event }),
        };
        Ok(next)
    }
    pub fn terminal(self) -> bool {
        matches!(self, Self::Committed | Self::Failed | Self::Cancelled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn happy_path() {
        let s = State::Accepted.transition(Event::Enqueue).unwrap()
            .transition(Event::Start).unwrap()
            .transition(Event::FinishInference).unwrap()
            .transition(Event::Commit).unwrap();
        assert_eq!(s, State::Committed);
    }
    #[test]
    fn rejected_terminal_transition() {
        assert!(State::Committed.transition(Event::Start).is_err());
        assert!(State::Cancelled.transition(Event::Retry).is_err());
    }
    #[test]
    fn tool_resume_is_requeued() {
        assert_eq!(State::Running.transition(Event::WaitForTool).unwrap()
            .transition(Event::ResumeTool).unwrap(), State::Queued);
    }
    #[test]
    fn recovered_run_requires_reconciliation() {
        assert_eq!(State::Running.transition(Event::LoseLease).unwrap()
            .transition(Event::Retry).unwrap(), State::Queued);
    }
}
