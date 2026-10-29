# Model pricing for clients

An optional top-level `pricing` object in the run config is advertised unchanged in
`/models` and `/v1/models`. Clients such as Hermes can use these operator-supplied
rates to estimate session costs.

```json
{
  "server": { "model": "model.basalt", "alias": "local-qwen" },
  "engine": { "max-context": 262144, "kv": "int8" },
  "pricing": {
    "prompt": "0.00000015",
    "completion": "0.00000047",
    "input_cache_read": "0.000000016",
    "input_cache_write": "0.0000002",
    "request": "0"
  }
}
```

These are accounting rates, not measured operating costs. Names and units follow
the [OpenRouter pricing object](https://openrouter.ai/docs/guides/overview/models):
USD per input token (`prompt`), output token (`completion`), cached input token
(`input_cache_read`) or cache-write token (`input_cache_write`). `request` is USD
per API request. `cache_read` and `cache_write` are also accepted.

Each known rate must be a finite, nonnegative number or decimal string; zero means
free. Invalid rates stop startup before the model loads. Unknown fields are
preserved for client extensions. An omitted block, `null` or `{}` leaves the model
listing unchanged. Basalt advertises the rates; the client calculates estimates
from its token usage. Restart the server after changing the config.
