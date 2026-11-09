-- Null means "not observed": never invent zero cache reuse or TTFT.
CREATE TABLE thread_run_telemetry (
 run_id UUID PRIMARY KEY REFERENCES runs(id),
 prompt_tokens BIGINT CHECK(prompt_tokens >= 0),
 reported_cached_tokens BIGINT CHECK(reported_cached_tokens >= 0),
 completion_tokens BIGINT CHECK(completion_tokens >= 0),
 roundtrip_ms BIGINT CHECK(roundtrip_ms >= 0),
 ttft_ms BIGINT CHECK(ttft_ms >= 0),
 source TEXT NOT NULL,
 captured_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 CHECK(prompt_tokens IS NULL OR reported_cached_tokens IS NULL
       OR reported_cached_tokens <= prompt_tokens)
);
CREATE INDEX thread_run_telemetry_captured_idx ON thread_run_telemetry(captured_at);
