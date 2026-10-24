# Community runs

[jesdga95.github.io/basalt](https://jesdga95.github.io/basalt/) lists benchmark runs submitted by users, all taken with the same preset so the rows compare. The data is `docs/community/benchmarks.json`.

## Add your run

1. Stop any running Basalt server (the GPUs must be free), then run `basalt-bench --community`. It starts its own engine (three times, once per measurement), shows the load progress on stderr while each one starts, adds its row to `docs/community/benchmarks.json` and prints it.
2. `--submitter NAME` is required: the name you want shown (any text, up to 40 characters; it is not verified). `notes` (up to 200 characters) is optional and added by hand.
3. Review the row in `docs/community/benchmarks.json`, commit it and open a pull request. If the bench could not write the file, it prints the row and says where to paste it.

You need enough free RAM that no expert is read from the SSD; the bench checks this and refuses to print a row otherwise.

The bench writes the file in the house format. After a hand edit, run `python3 tools/check_benchmarks.py --fix docs/community/benchmarks.json`; it rewrites the file in the house format and validates it. CI runs the same checker without `--fix` and fails on any other format. A benchmark-only pull request skips the full build.
