-- Persist the complete incoming user message BEFORE scheduling inference.
-- The replay source is never an in-process Python buffer.
CREATE TABLE turn_inputs (
 turn_id UUID PRIMARY KEY REFERENCES turns(id),
 role TEXT NOT NULL CHECK(role IN ('user','tool')),
 content JSONB NOT NULL,
 content_hash BYTEA NOT NULL CHECK(octet_length(content_hash)=32),
 created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TRIGGER turn_inputs_immutable BEFORE UPDATE OR DELETE ON turn_inputs
 FOR EACH ROW EXECUTE FUNCTION reject_message_mutation();

-- The first immutable user message is inserted in the same final commit as the
-- assistant message. There is no orphan user message on generation failure.
CREATE FUNCTION commit_thread_exchange(
 p_turn_id UUID, p_run_id UUID, p_fencing BIGINT,
 p_user_message_id UUID, p_assistant_message_id UUID,
 p_assistant_content JSONB, p_assistant_hash BYTEA, p_model_id TEXT
) RETURNS UUID LANGUAGE plpgsql AS $$
DECLARE
 t turns%ROWTYPE; r runs%ROWTYPE; b branches%ROWTYPE; i turn_inputs%ROWTYPE;
BEGIN
 SELECT * INTO t FROM turns WHERE id=p_turn_id FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'turn not found' USING ERRCODE='P0002'; END IF;
 IF t.status = 'COMMITTED' THEN RETURN t.committed_message_id; END IF;
 IF t.status <> 'FINALIZING' THEN
    RAISE EXCEPTION 'turn not finalizing' USING ERRCODE='23514';
 END IF;
 SELECT * INTO r FROM runs WHERE id=p_run_id AND turn_id=t.id FOR UPDATE;
 IF NOT FOUND OR r.lease_token <> p_fencing OR r.state <> 'FINALIZING'
    OR r.lease_expires_at IS NULL OR r.lease_expires_at <= clock_timestamp()
    OR EXISTS (SELECT 1 FROM runs WHERE turn_id=t.id AND lease_token > p_fencing) THEN
    RAISE EXCEPTION 'stale or expired worker lease' USING ERRCODE='40001';
 END IF;
 SELECT * INTO b FROM branches WHERE thread_id=t.thread_id AND id=t.branch_id FOR UPDATE;
 IF NOT FOUND OR b.version <> t.expected_branch_version
    OR b.head_message_id IS DISTINCT FROM t.parent_message_id THEN
    RAISE EXCEPTION 'branch CAS conflict' USING ERRCODE='40001';
 END IF;
 SELECT * INTO i FROM turn_inputs WHERE turn_id=t.id;
 IF NOT FOUND THEN RAISE EXCEPTION 'missing durable user input' USING ERRCODE='23514'; END IF;
 INSERT INTO messages(id,thread_id,parent_id,role,content,content_hash)
 VALUES(p_user_message_id,t.thread_id,t.parent_message_id,i.role,i.content,i.content_hash);
 INSERT INTO messages(id,thread_id,parent_id,role,content,content_hash,model_id)
 VALUES(p_assistant_message_id,t.thread_id,p_user_message_id,'assistant',p_assistant_content,
        p_assistant_hash,p_model_id);
 UPDATE branches SET head_message_id=p_assistant_message_id,version=version+1
 WHERE id=t.branch_id;
 UPDATE turns SET status='COMMITTED',committed_message_id=p_assistant_message_id WHERE id=t.id;
 UPDATE runs SET state='COMMITTED',completed_at=clock_timestamp() WHERE id=p_run_id;
 RETURN p_assistant_message_id;
END $$;
