# Community runs

[basalt.jgulfo.com/community](https://basalt.jgulfo.com/community/) lists benchmark runs submitted by users, all taken with the same preset so the rows compare. The data is `docs/community/benchmarks.json`; the page reads it from `main`, so a merged row shows up within minutes.

## Add your run

1. Stop any running Basalt server (the GPUs must be free), then run `basalt-bench --community --model PACK` (a Flash-Next or a DeepSeek pack). It starts its own engine (three times, once per measurement), shows the load progress on stderr while each one starts, adds its row to `docs/community/benchmarks.json` and prints it. It uses every visible Blackwell GPU: a second card is the peer, and `CUDA_VISIBLE_DEVICES=0` gives a single-card row.
2. `--submitter NAME` is required: the name you want shown (one name with no spaces, up to 40 characters; it is not verified). `notes` (up to 200 characters) is optional and added by hand.
3. Review the row in `docs/community/benchmarks.json`, commit it and open a pull request. If the file already has a row of your setup (see below), the bench replaces that row in place, keeps its `notes` and says so; check the notes still apply. If the bench could not write the file, it prints the row and says where to paste it.

The row records `model` (`Qwen3.8-Flash-Next` or `DeepSeek-V4.1-Flash`, the name the model's adapter reports for the pack's architecture, not its file name) and `kv`, the KV cache format the run used: the preset passes `--kv int8` to Flash-Next and `--kv fp8` to DeepSeek, and a model that takes neither runs its own format. Rows of different models are not comparable.

You need enough free RAM that no expert is read from the SSD; the bench checks this and refuses to print a row otherwise.

Optional fields, recorded by `--community` before the engine starts (`basalt-bench --hardware` prints them alone):

- `cpu`: the processor and its physical core count, e.g. "AMD Ryzen 9 9950X, 16 cores".
- `ram_gbps`: host RAM read speed in GB/s, one thread per physical core streaming a 2 GiB buffer, best of several passes.
- per GPU, `pcie`: the link as the driver reports it, e.g. "5.0 x16": the maximum generation (the current one drops at idle) and the current width (the maximum width is the card's, not the slot's, so a slot shared between two cards reads x8).
- per GPU, `h2d_gbps`: pinned host-to-device copy speed in GB/s, best of several 256 MiB copies.

Optional fields, recorded by `--community` from the same three decode requests (rows from before they existed lack them; `decode_tps` keeps its meaning, so old and new rows compare):

- `decode_conv_tps`, `decode_struct_tps`, `decode_prose_tps`: that fixture's generated tokens over its decode time. `decode_tps` stays the sum of the three fixtures' tokens over the sum of their decode times (token-weighted, not the mean of these three).
- `accept_conv_pct`, `accept_struct_pct`, `accept_prose_pct`: draft tokens the model accepted over draft tokens it offered, in percent, and `draft_len_conv`, `draft_len_struct`, `draft_len_prose`: the mean number of draft tokens offered per decode window. Both are left out for a fixture the model did not draft in.
- `engine_args`: the arguments the preset passed the engine, without the model path (at most 300 characters).

The bench writes the file in the house format. After a hand edit, run `python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json`; it rewrites the file in the house format and validates it. CI runs the same checker without `--fix` and fails on any other format. A benchmark-only pull request skips the full build.

One row per setup: the checker refuses a row whose submitter (ASCII letters compared without case), model, quant, every GPU (name, VRAM, power limit, in that order), CPU, RAM size and system all match an earlier row. A field one of the two rows lacks counts as matching, so rows from before a field existed still match. The KV format is not compared. To change a run, update the earlier row (`--community` does this itself); do not add another.
