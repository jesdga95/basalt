# Community runs

[jesdga95.github.io/basalt](https://jesdga95.github.io/basalt/) lists benchmark runs submitted by users, all taken with the same preset so the rows compare. The data is `docs/community/benchmarks.json`.

## Add your run

1. Stop any running Basalt server (the GPUs must be free), then run `basalt-bench --community`. It starts its own engine (three times, once per measurement), shows the load progress on stderr while each one starts, and prints one JSON row.
2. Replace `YOUR_GITHUB_HANDLE` in the row with your GitHub username (no `@`; letters, digits and single hyphens, up to 39 characters). `link` (an https URL) and `notes` (up to 200 characters) are optional.
3. Add the row to the `runs` list in `docs/community/benchmarks.json` and open a pull request.

You need enough free RAM that no expert is read from the SSD; the bench checks this and refuses to print a row otherwise.

CI runs `python3 tools/check_benchmarks.py docs/community/benchmarks.json` on the pull request; run it locally to see the same messages. A benchmark-only pull request skips the full build.
