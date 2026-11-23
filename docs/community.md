# Community runs

[basalt.jgulfo.com/community](https://basalt.jgulfo.com/community/) lists benchmark runs submitted by users, all taken with the same preset so the rows compare. Each run is one file, `docs/community/runs/<model>/<id>.json`, and the preset all of them share is `docs/community/preset.json`. When a run lands on `main`, an action copies the files into the table the page reads, so a merged run shows up within minutes.

## Add your run

1. Stop any running Basalt server (the GPUs must be free), then run `basalt-bench --community --model PACK` (a Flash-Next, a 27B, a DeepSeek, a Hy4-flash or an Inkling 2 Mini pack). It starts its own engine (three times, once per measurement), shows the load progress on stderr while each one starts, writes its row to its own file under `docs/community/runs/` and prints it. It uses every visible Blackwell GPU: a second card is the peer, and `CUDA_VISIBLE_DEVICES=0` gives a single-card row. Where the driver enumerates an older card first - a Blackwell card with an `-DPEER_ADA=ON` peer - put the Blackwell card first anyway: `CUDA_VISIBLE_DEVICES=1` for a single-card row, `CUDA_VISIBLE_DEVICES=1,0` with the peer, because the harness samples power and clocks from the first visible card while it measures.
2. `--submitter NAME` is required: the name you want shown (one name with no spaces, up to 40 characters; it is not verified). `notes` (up to 200 characters) is optional and added by hand.
3. Review the new file under `docs/community/runs/`, commit it and open a pull request. The file name is the run's id (see below): if you already have a run of the same setup, the bench overwrites that file, keeps its `notes` and says so; check the notes still apply. `--community-dir DIR` writes under another community folder (the one holding `runs/`). If the bench could not write the file, it prints the row and says where to save it.

The row records `model` (`Qwen3.8-Flash-Next`, `Qwen3.8-27B`, `DeepSeek-V4.1-Flash`, `Hy4-flash` or `Inkling-2-Mini`, the name the model's adapter reports for the pack's architecture, not its file name) and `kv`, the KV cache format the run used: the preset passes `--kv int8` to Flash-Next and the 27B and `--kv fp8` to DeepSeek, and a model that takes neither runs its own format (Hy4-flash and Inkling 2 Mini run their default, `fp16`). Rows of different models are not comparable.

You need enough free RAM that no expert is read from the SSD; the bench checks this and refuses to print a row otherwise. The 27B has no experts: the bench instead refuses a row if any of its weights sit in host RAM; its token embedding, which the engine keeps in page-locked RAM by design, and the `--vision dynamic` encoder, which waits in RAM and loads per request, do not count.

Optional fields, recorded by `--community` before the engine starts (`basalt-bench --hardware` prints them alone):

- `cpu`: the processor and its physical core count, e.g. "AMD Ryzen 9 9950X, 16 cores".
- `ram_gbps`: host RAM read speed in GB/s, one thread per physical core streaming a 2 GiB buffer, best of several passes.
- per GPU, `pcie`: the link as the driver reports it, e.g. "5.0 x16": the maximum generation (the current one drops at idle) and the current width (the maximum width is the card's, not the slot's, so a slot shared between two cards reads x8).
- per GPU, `h2d_gbps`: pinned host-to-device copy speed in GB/s, best of several 256 MiB copies.

Optional fields, recorded by `--community` from the same three decode requests (rows from before they existed lack them; `decode_tps` keeps its meaning, so old and new rows compare):

- `decode_conv_tps`, `decode_struct_tps`, `decode_prose_tps`: that fixture's generated tokens over its decode time. `decode_tps` stays the sum of the three fixtures' tokens over the sum of their decode times (token-weighted, not the mean of these three).
- `accept_conv_pct`, `accept_struct_pct`, `accept_prose_pct`: draft tokens the model accepted over draft tokens it offered, in percent, and `draft_len_conv`, `draft_len_struct`, `draft_len_prose`: the mean number of draft tokens offered per decode window. Both are left out for a fixture the model did not draft in.
- `engine_args`: the arguments the preset passed the engine, without the model path (at most 300 characters).

The bench writes the file in the house format. After a hand edit, run `python3 tools/check_benchmarks.py --fix docs/community`; it rewrites the files in the house format and validates the folder. CI runs the same checker without `--fix` and fails on any other format. A benchmark-only pull request skips the full build.

One run per setup: the checker refuses two files whose rows have the same submitter (ASCII letters compared without case), model, quant, every GPU (name, VRAM, power limit, in that order), CPU, RAM size and system. A field one of the two rows lacks counts as matching, so rows from before a field existed still match. The KV format is not compared. To change a run, change its file (`--community` overwrites it itself); do not add another.

The file's place says which run it is. The folder is the model's name in ASCII lower case (`qwen3.8-flash-next`) and the file name is the run's id, the first 12 hex digits of the SHA-256 of the setup above taken as exact values: the submitter in lower case, model, quant, CPU, RAM size, system and every GPU's name, VRAM and power limit. The text hashed is the compact JSON array `[submitter, model, quant, cpu, ram_gib, os, [[gpu name, vram_gib, power_w], ...]]` with the numbers as whole numbers and a missing CPU as `""`; `row_id` in `tools/check_benchmarks.py` spells it out and `basalt-bench` computes the same id. The same submitter and setup always give the same file, so a new run overwrites your own, and no file name holds a user name. The checker refuses a file whose name or folder does not match its content.
