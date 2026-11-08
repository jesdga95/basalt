-- Basalt Stateful Threads v1: immutable ledger and branch-constrained execution.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE thread_owners (
 id UUID PRIMARY KEY, workspace_id UUID NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE threads (
 id UUID PRIMARY KEY, owner_id UUID NOT NULL REFERENCES thread_owners(id),
 model_id TEXT NOT NULL, default_branch_id UUID,
 status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','deleted')),
 version BIGINT NOT NULL DEFAULT 0 CHECK(version >= 0),
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 UNIQUE(id, owner_id)
);
CREATE TABLE messages (
 id UUID PRIMARY KEY, thread_id UUID NOT NULL REFERENCES threads(id),
 parent_id UUID, role TEXT NOT NULL CHECK(role IN ('system','developer','user','assistant','tool')),
 content JSONB NOT NULL, content_hash BYTEA NOT NULL CHECK(octet_length(content_hash)=32),
 model_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 UNIQUE(thread_id,id),
 FOREIGN KEY(thread_id,parent_id) REFERENCES messages(thread_id,id) DEFERRABLE INITIALLY IMMEDIATE,
 CHECK(parent_id IS NULL OR parent_id <> id)
);
CREATE TABLE branches (
 id UUID PRIMARY KEY, thread_id UUID NOT NULL REFERENCES threads(id),
 head_message_id UUID, version BIGINT NOT NULL DEFAULT 0 CHECK(version >= 0),
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 UNIQUE(thread_id,id),
 FOREIGN KEY(thread_id,head_message_id) REFERENCES messages(thread_id,id)
);
ALTER TABLE threads ADD CONSTRAINT thread_default_branch_same_thread
 FOREIGN KEY(id,default_branch_id) REFERENCES branches(thread_id,id) DEFERRABLE INITIALLY DEFERRED;
CREATE TABLE turns (
 id UUID PRIMARY KEY, thread_id UUID NOT NULL, branch_id UUID NOT NULL,
 parent_message_id UUID, idempotency_key TEXT NOT NULL CHECK(length(idempotency_key) BETWEEN 1 AND 256),
 request_hash BYTEA NOT NULL CHECK(octet_length(request_hash)=32),
 status TEXT NOT NULL DEFAULT 'ACCEPTED'
 CHECK(status IN ('ACCEPTED','QUEUED','RUNNING','WAITING_TOOL','FINALIZING','COMMITTED','FAILED','CANCELLED','RECOVERABLE')),
 committed_message_id UUID,
 expected_branch_version BIGINT NOT NULL CHECK(expected_branch_version >= 0),
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 UNIQUE(thread_id,branch_id,idempotency_key),
 FOREIGN KEY(thread_id,branch_id) REFERENCES branches(thread_id,id),
 FOREIGN KEY(thread_id,parent_message_id) REFERENCES messages(thread_id,id),
 FOREIGN KEY(thread_id,committed_message_id) REFERENCES messages(thread_id,id)
);
CREATE TABLE runs (
 id UUID PRIMARY KEY, turn_id UUID NOT NULL REFERENCES turns(id),
 attempt INTEGER NOT NULL CHECK(attempt > 0),
 state TEXT NOT NULL CHECK(state IN ('ACCEPTED','QUEUED','RUNNING','WAITING_TOOL','FINALIZING','COMMITTED','FAILED','CANCELLED','RECOVERABLE')),
 lease_token BIGINT NOT NULL CHECK(lease_token > 0),
 model_fingerprint BYTEA NOT NULL CHECK(octet_length(model_fingerprint)=32),
 lease_expires_at TIMESTAMPTZ, started_at TIMESTAMPTZ, completed_at TIMESTAMPTZ,
 UNIQUE(turn_id,attempt), UNIQUE(turn_id,lease_token)
);
CREATE TABLE message_blocks (
 message_id UUID NOT NULL REFERENCES messages(id), ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
 kind TEXT NOT NULL, raw_content JSONB NOT NULL,
 tool_call_id TEXT, run_id UUID REFERENCES runs(id),
 PRIMARY KEY(message_id,ordinal)
);
CREATE TABLE tool_executions (
 id UUID PRIMARY KEY, run_id UUID NOT NULL REFERENCES runs(id),
 tool_call_id TEXT NOT NULL, arguments JSONB NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('PENDING','APPROVED','RUNNING','SUCCEEDED','FAILED','UNKNOWN')),
 idempotency_key TEXT NOT NULL, result JSONB,
 UNIQUE(run_id,tool_call_id), UNIQUE(run_id,idempotency_key)
);
CREATE TABLE thread_events (
 turn_id UUID NOT NULL REFERENCES turns(id), sequence BIGINT NOT NULL CHECK(sequence > 0),
 event_type TEXT NOT NULL, payload JSONB NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(turn_id,sequence)
);
CREATE TABLE cache_manifests (
 id UUID PRIMARY KEY, owner_id UUID NOT NULL REFERENCES thread_owners(id),
 model_fingerprint BYTEA NOT NULL CHECK(octet_length(model_fingerprint)=32),
 prefix_hash BYTEA NOT NULL CHECK(octet_length(prefix_hash)=32),
 prefix_tokens BIGINT NOT NULL CHECK(prefix_tokens >= 0),
 format_version INTEGER NOT NULL CHECK(format_version > 0),
 engine_epoch UUID NOT NULL, residency TEXT NOT NULL CHECK(residency IN ('GPU','HOST','DISK','MISSING')),
 locator TEXT, validated BOOLEAN NOT NULL DEFAULT false,
 updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 UNIQUE(owner_id,model_fingerprint,prefix_hash,format_version)
);
CREATE INDEX messages_parent_idx ON messages(thread_id,parent_id);
CREATE INDEX turns_pending_idx ON turns(status,created_at) WHERE status NOT IN ('COMMITTED','FAILED','CANCELLED');
CREATE INDEX runs_lease_idx ON runs(lease_expires_at) WHERE state IN ('RUNNING','WAITING_TOOL','FINALIZING');
CREATE INDEX thread_events_created_idx ON thread_events(created_at);
CREATE INDEX cache_owner_idx ON cache_manifests(owner_id,updated_at);

-- Immutable message rows: edits must be represented by new graph nodes.
CREATE FUNCTION reject_message_mutation() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'messages are immutable; insert a new message instead'; END $$;
CREATE TRIGGER messages_immutable BEFORE UPDATE OR DELETE ON messages
 FOR EACH ROW EXECUTE FUNCTION reject_message_mutation();

-- Branch CAS must be executed in a transaction that has locked the branch row.
-- UPDATE branches SET head_message_id=$new, version=version+1
-- WHERE id=$branch AND thread_id=$thread AND version=$expected
--   AND head_message_id IS NOT DISTINCT FROM $parent;
-- Exactly one updated row means commit may proceed; zero rows means 409 conflict.
