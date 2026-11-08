-- Transactional commit primitive; PostgreSQL function call runs atomically.
-- The worker's fencing token must still equal the newest active run token.
CREATE FUNCTION commit_thread_turn(
 p_turn_id UUID,
 p_run_id UUID,
 p_lease_token BIGINT,
 p_message_id UUID,
 p_role TEXT,
 p_content JSONB,
 p_content_hash BYTEA,
 p_model_id TEXT DEFAULT NULL
) RETURNS UUID LANGUAGE plpgsql AS $$
DECLARE
 t turns%ROWTYPE;
 r runs%ROWTYPE;
 b branches%ROWTYPE;
BEGIN
 SELECT * INTO t FROM turns WHERE id=p_turn_id FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'turn not found' USING ERRCODE='P0002'; END IF;
 IF t.status = 'COMMITTED' THEN
   RETURN t.committed_message_id;
 END IF;
 IF t.status <> 'FINALIZING' THEN
   RAISE EXCEPTION 'turn not finalizing' USING ERRCODE='23514';
 END IF;
 SELECT * INTO r FROM runs WHERE id=p_run_id AND turn_id=t.id FOR UPDATE;
 IF NOT FOUND OR r.lease_token <> p_lease_token OR r.state <> 'FINALIZING'
    OR r.lease_expires_at IS NULL OR r.lease_expires_at <= clock_timestamp() THEN
   RAISE EXCEPTION 'run fencing token or lease invalid' USING ERRCODE='40001';
 END IF;
 IF EXISTS (SELECT 1 FROM runs WHERE turn_id=t.id AND lease_token > p_lease_token) THEN
   RAISE EXCEPTION 'stale run' USING ERRCODE='40001';
 END IF;
 SELECT * INTO b FROM branches WHERE thread_id=t.thread_id AND id=t.branch_id FOR UPDATE;
 IF b.version <> t.expected_branch_version
    OR b.head_message_id IS DISTINCT FROM t.parent_message_id THEN
   RAISE EXCEPTION 'branch head conflict' USING ERRCODE='40001';
 END IF;
 INSERT INTO messages(id,thread_id,parent_id,role,content,content_hash,model_id)
 VALUES(p_message_id,t.thread_id,t.parent_message_id,p_role,p_content,p_content_hash,p_model_id);
 UPDATE branches SET head_message_id=p_message_id, version=version+1 WHERE id=t.branch_id;
 UPDATE turns SET status='COMMITTED', committed_message_id=p_message_id WHERE id=t.id;
 UPDATE runs SET state='COMMITTED', completed_at=clock_timestamp() WHERE id=p_run_id;
 RETURN p_message_id;
END $$;

-- Deny mutations to the ledger once inserted; runtime state lives in turns/runs.
-- Retention must use an explicit tenant-authorized tombstoning/purge path.
