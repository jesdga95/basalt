# Community runs

[jesdga95.github.io/basalt](https://jesdga95.github.io/basalt/) lists benchmark runs submitted by users, all taken with the same preset so the rows compare. The data is `docs/community/benchmarks.json`.

## Add your run

1. Stop any running Basalt server (the GPUs must be free), then run `basalt-bench --community`. It starts its own engine (three times, once per measurement), shows the load progress on stderr while each one starts, adds its row to `docs/community/benchmarks.json` and prints it. It uses every visible Blackwell GPU: a second card is the peer, and `CUDA_VISIBLE_DEVICES=0` gives a single-card row.
2. `--submitter NAME` is required: the name you want shown (any text, up to 40 characters; it is not verified). `notes` (up to 200 characters) is optional and added by hand.
3. Review the row in `docs/community/benchmarks.json`, commit it and open a pull request. If the bench could not write the file, it prints the row and says where to paste it.

You need enough free RAM that no expert is read from the SSD; the bench checks this and refuses to print a row otherwise.

Optional fields, recorded by `--community` before the engine starts (`basalt-bench --hardware` prints them alone):

- `cpu`: the processor and its physical core count, e.g. "AMD Ryzen 9 9950X, 16 cores".
- `ram_gbps`: host RAM read speed in GB/s, one thread per physical core streaming a 2 GiB buffer, best of several passes.
- per GPU, `pcie`: the link as the driver reports it, e.g. "5.0 x16": the maximum generation (the current one drops at idle) and the current width (the maximum width is the card's, not the slot's, so a slot shared between two cards reads x8).
- per GPU, `h2d_gbps`: pinned host-to-device copy speed in GB/s, best of several 256 MiB copies.

The bench writes the file in the house format. After a hand edit, run `python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json`; it rewrites the file in the house format and validates it. CI runs the same checker without `--fix` and fails on any other format. A benchmark-only pull request skips the full build.
