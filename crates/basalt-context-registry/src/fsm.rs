//! Pure context lifecycle FSM (docs/architecture/agent-context-contract.md).
//! No I/O: the PostgreSQL store applies these transitions under CAS on `version`.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum State {
    Recorded, Materializing, Hot, Cooling, Warm, Cold, Restoring, Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Event {
    Record, Materialize, Ready, Cool, Warm, Chill, Restore, Fail, Reset,
}

/// Where the snapshot payload currently lives. `None` means "replay from transcript".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier { None, Hot, Warm, Cold }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransitionError { pub state: State, pub event: Event }

/// Lease requirement of an event; `Forbidden` also demands that no live lease exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeaseRule { Required, Optional, Forbidden }

impl State {
    pub const ALL: [State; 8] = [
        State::Recorded, State::Materializing, State::Hot, State::Cooling,
        State::Warm, State::Cold, State::Restoring, State::Failed,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            State::Recorded => "RECORDED", State::Materializing => "MATERIALIZING",
            State::Hot => "HOT", State::Cooling => "COOLING", State::Warm => "WARM",
            State::Cold => "COLD", State::Restoring => "RESTORING", State::Failed => "FAILED",
        }
    }
    pub fn parse(s: &str) -> Option<State> {
        State::ALL.into_iter().find(|v| v.as_str() == s)
    }

    /// Creation state; `record` is the only event that is not a transition.
    pub fn initial() -> State { State::Recorded }

    pub fn transition(self, event: Event) -> Result<State, TransitionError> {
        use Event as E;
        use State as S;
        let next = match (self, event) {
            (S::Recorded | S::Cold | S::Failed, E::Materialize) => S::Materializing,
            (S::Materializing | S::Restoring, E::Ready) => S::Hot,
            (S::Hot, E::Cool) => S::Cooling,
            (S::Cooling, E::Warm) => S::Warm,
            (S::Warm | S::Cooling, E::Chill) => S::Cold,
            (S::Warm | S::Cold, E::Restore) => S::Restoring,
            (S::Materializing | S::Restoring | S::Cooling, E::Fail) => S::Failed,
            (S::Failed, E::Reset) => S::Recorded,
            _ => return Err(TransitionError { state: self, event }),
        };
        Ok(next)
    }

    /// Context bytes available for reuse without replay (HOT/COOLING/WARM/COLD).
    pub fn source(self) -> &'static str {
        match self {
            State::Hot => "hot",
            State::Warm => "warm",
            State::Cold => "cold",
            _ => "replay",
        }
    }
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self { Tier::None => "none", Tier::Hot => "hot", Tier::Warm => "warm", Tier::Cold => "cold" }
    }
    pub fn parse(s: &str) -> Option<Tier> {
        [Tier::None, Tier::Hot, Tier::Warm, Tier::Cold].into_iter().find(|v| v.as_str() == s)
    }
}

/// Tier implied by a transition. RESTORING keeps the tier it restores from.
pub fn tier_after(from: State, to: State) -> Tier {
    match to {
        State::Recorded | State::Materializing | State::Failed => Tier::None,
        State::Hot | State::Cooling => Tier::Hot,
        State::Warm => Tier::Warm,
        State::Cold => Tier::Cold,
        State::Restoring => if from == State::Cold { Tier::Cold } else { Tier::Warm },
    }
}

impl Event {
    /// Events accepted over HTTP. `record` only exists at creation.
    pub fn parse(s: &str) -> Option<Event> {
        Some(match s {
            "record" => Event::Record, "materialize" => Event::Materialize,
            "ready" => Event::Ready, "cool" => Event::Cool, "warm" => Event::Warm,
            "chill" => Event::Chill, "restore" => Event::Restore,
            "fail" => Event::Fail, "reset" => Event::Reset,
            _ => return None,
        })
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Event::Record => "record", Event::Materialize => "materialize", Event::Ready => "ready",
            Event::Cool => "cool", Event::Warm => "warm", Event::Chill => "chill",
            Event::Restore => "restore", Event::Fail => "fail", Event::Reset => "reset",
        }
    }

    /// Writer-side events bring a context towards HOT and need the fenced lease;
    /// eviction (`cool`) needs the opposite: no live lease at all.
    pub fn lease_rule(self, from: State) -> LeaseRule {
        match self {
            Event::Materialize | Event::Restore | Event::Ready => LeaseRule::Required,
            Event::Fail if matches!(from, State::Materializing | State::Restoring) => LeaseRule::Required,
            Event::Cool => LeaseRule::Forbidden,
            _ => LeaseRule::Optional,
        }
    }
}

/// Legal event path that returns a context whose volatile residency is gone to a
/// rebuildable state. Only contract events are used, so recovery never invents a
/// transition. `engine_lost` covers HOT/COOLING/WARM after an engine restart.
pub fn recovery_plan(state: State, engine_lost: bool) -> Vec<Event> {
    match state {
        State::Materializing => vec![Event::Fail, Event::Reset],
        State::Restoring => vec![Event::Fail],
        State::Hot if engine_lost => vec![Event::Cool, Event::Fail, Event::Reset],
        State::Cooling if engine_lost => vec![Event::Fail, Event::Reset],
        State::Warm if engine_lost => vec![Event::Restore, Event::Fail, Event::Reset],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Event as E;
    use State as S;

    /// The complete contract table: every (state, event) pair not listed must be rejected.
    const ALLOWED: &[(State, Event, State)] = &[
        (S::Recorded, E::Materialize, S::Materializing),
        (S::Cold, E::Materialize, S::Materializing),
        (S::Failed, E::Materialize, S::Materializing),
        (S::Materializing, E::Ready, S::Hot),
        (S::Restoring, E::Ready, S::Hot),
        (S::Hot, E::Cool, S::Cooling),
        (S::Cooling, E::Warm, S::Warm),
        (S::Warm, E::Chill, S::Cold),
        (S::Cooling, E::Chill, S::Cold),
        (S::Warm, E::Restore, S::Restoring),
        (S::Cold, E::Restore, S::Restoring),
        (S::Materializing, E::Fail, S::Failed),
        (S::Restoring, E::Fail, S::Failed),
        (S::Cooling, E::Fail, S::Failed),
        (S::Failed, E::Reset, S::Recorded),
    ];
    const EVENTS: [Event; 9] = [E::Record, E::Materialize, E::Ready, E::Cool, E::Warm,
        E::Chill, E::Restore, E::Fail, E::Reset];

    #[test]
    fn exhaustive_transition_table() {
        for s in State::ALL {
            for e in EVENTS {
                let expect = ALLOWED.iter().find(|(f, ev, _)| *f == s && *ev == e).map(|t| t.2);
                match (s.transition(e), expect) {
                    (Ok(got), Some(want)) => assert_eq!(got, want, "{s:?} --{e:?}-->"),
                    (Err(err), None) => assert_eq!(err, TransitionError { state: s, event: e }),
                    (got, want) => panic!("{s:?} --{e:?}--> got {got:?} want {want:?}"),
                }
            }
        }
    }

    #[test]
    fn record_is_creation_only() {
        for s in State::ALL { assert!(s.transition(E::Record).is_err()); }
        assert_eq!(State::initial(), S::Recorded);
    }

    #[test]
    fn full_lifecycle_round_trip() {
        let path = [E::Materialize, E::Ready, E::Cool, E::Warm, E::Chill, E::Restore, E::Ready];
        let end = path.iter().try_fold(State::initial(), |s, e| s.transition(*e)).unwrap();
        assert_eq!(end, S::Hot);
    }

    #[test]
    fn names_round_trip() {
        for s in State::ALL { assert_eq!(State::parse(s.as_str()), Some(s)); }
        for e in EVENTS { assert_eq!(Event::parse(e.as_str()), Some(e)); }
        for t in [Tier::None, Tier::Hot, Tier::Warm, Tier::Cold] {
            assert_eq!(Tier::parse(t.as_str()), Some(t));
        }
        assert_eq!(Event::parse("evict"), None);
        assert_eq!(State::parse("hot"), None);
        assert_eq!(serde_json::to_string(&S::Materializing).unwrap(), "\"MATERIALIZING\"");
    }

    #[test]
    fn tier_follows_state() {
        for (from, ev, to) in ALLOWED {
            let tier = tier_after(*from, *to);
            let ok = match to {
                S::Recorded | S::Materializing | S::Failed => tier == Tier::None,
                S::Hot | S::Cooling => tier == Tier::Hot,
                S::Warm => tier == Tier::Warm,
                S::Cold => tier == Tier::Cold,
                S::Restoring => tier == if *from == S::Cold { Tier::Cold } else { Tier::Warm },
            };
            assert!(ok, "{from:?} --{ev:?}--> {to:?} tier {tier:?}");
        }
    }

    #[test]
    fn lease_rules() {
        assert_eq!(E::Materialize.lease_rule(S::Recorded), LeaseRule::Required);
        assert_eq!(E::Restore.lease_rule(S::Cold), LeaseRule::Required);
        assert_eq!(E::Ready.lease_rule(S::Restoring), LeaseRule::Required);
        assert_eq!(E::Fail.lease_rule(S::Materializing), LeaseRule::Required);
        assert_eq!(E::Fail.lease_rule(S::Cooling), LeaseRule::Optional);
        assert_eq!(E::Cool.lease_rule(S::Hot), LeaseRule::Forbidden);
        assert_eq!(E::Chill.lease_rule(S::Warm), LeaseRule::Optional);
        assert_eq!(E::Reset.lease_rule(S::Failed), LeaseRule::Optional);
    }

    #[test]
    fn recovery_plans_use_only_legal_events_and_end_rebuildable() {
        for s in State::ALL {
            for lost in [false, true] {
                let plan = recovery_plan(s, lost);
                let end = plan.iter().try_fold(s, |cur, e| cur.transition(*e))
                    .unwrap_or_else(|e| panic!("illegal plan for {s:?}: {e:?}"));
                if plan.is_empty() { assert_eq!(end, s); continue; }
                assert!(matches!(end, S::Recorded | S::Failed), "{s:?} -> {end:?}");
            }
        }
        assert!(recovery_plan(S::Hot, false).is_empty());
        assert!(recovery_plan(S::Cold, true).is_empty());
        assert_eq!(recovery_plan(S::Materializing, false), vec![E::Fail, E::Reset]);
        assert_eq!(recovery_plan(S::Restoring, false), vec![E::Fail]);
    }

    #[test]
    fn source_labels() {
        assert_eq!(S::Hot.source(), "hot");
        assert_eq!(S::Warm.source(), "warm");
        assert_eq!(S::Cold.source(), "cold");
        for s in [S::Recorded, S::Failed, S::Materializing, S::Restoring, S::Cooling] {
            assert_eq!(s.source(), "replay");
        }
    }
}
