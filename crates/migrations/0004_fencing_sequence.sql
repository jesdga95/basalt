-- Global PostgreSQL sequence: fencing tokens never go backwards across worker restarts.
CREATE SEQUENCE IF NOT EXISTS basalt_thread_fencing_seq AS BIGINT START WITH 1;
CREATE INDEX turns_thread_status_idx ON turns(thread_id,status);
