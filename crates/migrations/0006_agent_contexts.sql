-- Agent context registry (docs/architecture/agent-context-contract.md).
-- Metadata only: the transcript in `messages` stays authoritative; a context is a
-- model-specific, disposable acceleration handle that can always be rebuilt (RECORDED).
CREATE TABLE contexts (
 id UUID PRIMARY KEY,
 owner_id UUID NOT NULL REFERENCES thread_owners(id),
 thread_id UUID NOT NULL,
 branch_id UUID NOT NULL,
 head_message_id UUID,
 parent_id UUID,
 state TEXT NOT NULL DEFAULT 'RECORDED' CHECK(state IN
  ('RECORDED','MATERIALIZING','HOT','COOLING','WARM','COLD','RESTORING','FAILED')),
 tier TEXT NOT NULL DEFAULT 'none' CHECK(tier IN ('none','hot','warm','cold')),
 version BIGINT NOT NULL DEFAULT 0 CHECK(version >= 0),
 model_fingerprint BYTEA NOT NULL CHECK(octet_length(model_fingerprint)=32),
 prefix_hash BYTEA NOT NULL CHECK(octet_length(prefix_hash)=32),
 prefix_tokens BIGINT NOT NULL DEFAULT 0 CHECK(prefix_tokens >= 0),
 locator TEXT CHECK(locator IS NULL OR length(locator) BETWEEN 1 AND 4096),
 validated BOOLEAN NOT NULL DEFAULT false,
 engine_epoch UUID,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 deleted_at TIMESTAMPTZ,
 UNIQUE(id, owner_id),
 UNIQUE(id, owner_id, thread_id),
 -- tenant isolation: a context, its thread, its parent and its leases share one owner.
 FOREIGN KEY(thread_id, owner_id) REFERENCES threads(id, owner_id),
 FOREIGN KEY(thread_id, branch_id) REFERENCES branches(thread_id, id),
 FOREIGN KEY(thread_id, head_message_id) REFERENCES messages(thread_id, id),
 FOREIGN KEY(parent_id, owner_id, thread_id) REFERENCES contexts(id, owner_id, thread_id)
  ON DELETE RESTRICT,
 CHECK(parent_id IS NULL OR parent_id <> id),
 CHECK((state IN ('RECORDED','MATERIALIZING','FAILED') AND tier='none')
    OR (state IN ('HOT','COOLING') AND tier='hot')
    OR (state='WARM' AND tier='warm')
    OR (state='COLD' AND tier='cold')
    OR (state='RESTORING' AND tier IN ('warm','cold'))),
 CHECK(NOT validated OR tier <> 'none')
);
CREATE INDEX contexts_owner_thread_idx ON contexts(owner_id, thread_id) WHERE deleted_at IS NULL;
CREATE INDEX contexts_parent_idx ON contexts(parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX contexts_recovery_idx ON contexts(state)
 WHERE state IN ('MATERIALIZING','RESTORING','HOT','COOLING','WARM');
CREATE INDEX contexts_gc_idx ON contexts(deleted_at) WHERE deleted_at IS NOT NULL;

-- One fenced writer lease per context. Tokens come from the shared global sequence
-- (0004), so they never go backwards across restarts.
CREATE TABLE context_leases (
 token BIGINT PRIMARY KEY CHECK(token > 0),
 context_id UUID NOT NULL,
 owner_id UUID NOT NULL,
 acquired_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
 expires_at TIMESTAMPTZ NOT NULL,
 released_at TIMESTAMPTZ,
 release_reason TEXT CHECK(release_reason IN ('released','expired')),
 FOREIGN KEY(context_id, owner_id) REFERENCES contexts(id, owner_id) ON DELETE RESTRICT,
 CHECK(expires_at > acquired_at),
 CHECK((released_at IS NULL) = (release_reason IS NULL))
);
CREATE UNIQUE INDEX context_leases_one_active ON context_leases(context_id)
 WHERE released_at IS NULL;

-- Idempotency scoped by tenant + operation + target. Dropped together with the result
-- context once that context is garbage-collected.
CREATE TABLE context_idempotency (
 owner_id UUID NOT NULL REFERENCES thread_owners(id),
 operation TEXT NOT NULL CHECK(operation IN ('fork')),
 target UUID NOT NULL,
 key TEXT NOT NULL CHECK(length(key) BETWEEN 1 AND 256),
 request_hash BYTEA NOT NULL CHECK(octet_length(request_hash)=32),
 result_context_id UUID NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 PRIMARY KEY(owner_id, operation, target, key),
 FOREIGN KEY(result_context_id, owner_id) REFERENCES contexts(id, owner_id) ON DELETE CASCADE
);

-- Identity is immutable, a tombstone is one-way and versions never go backwards.
CREATE FUNCTION contexts_guard_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NEW.id IS DISTINCT FROM OLD.id OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
    OR NEW.thread_id IS DISTINCT FROM OLD.thread_id OR NEW.branch_id IS DISTINCT FROM OLD.branch_id
    OR NEW.head_message_id IS DISTINCT FROM OLD.head_message_id
    OR NEW.parent_id IS DISTINCT FROM OLD.parent_id OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
  RAISE EXCEPTION 'context identity is immutable' USING ERRCODE='23514';
 END IF;
 IF OLD.deleted_at IS NOT NULL AND NEW.deleted_at IS DISTINCT FROM OLD.deleted_at THEN
  RAISE EXCEPTION 'context tombstone is irreversible' USING ERRCODE='23514';
 END IF;
 IF NEW.version < OLD.version THEN
  RAISE EXCEPTION 'context version must not decrease' USING ERRCODE='23514';
 END IF;
 RETURN NEW;
END $$;
CREATE TRIGGER contexts_guard_update BEFORE UPDATE ON contexts
 FOR EACH ROW EXECUTE FUNCTION contexts_guard_update();

-- GC constraint: only a tombstoned context without live lease can be removed. Child
-- rows are blocked by the RESTRICT foreign key until the children are collected.
CREATE FUNCTION contexts_guard_delete() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF OLD.deleted_at IS NULL THEN
  RAISE EXCEPTION 'context must be tombstoned before GC' USING ERRCODE='23514';
 END IF;
 IF EXISTS (SELECT 1 FROM context_leases l WHERE l.context_id=OLD.id
            AND l.released_at IS NULL AND l.expires_at > clock_timestamp()) THEN
  RAISE EXCEPTION 'context holds a live lease' USING ERRCODE='23514';
 END IF;
 RETURN OLD;
END $$;
CREATE TRIGGER contexts_guard_delete BEFORE DELETE ON contexts
 FOR EACH ROW EXECUTE FUNCTION contexts_guard_delete();
