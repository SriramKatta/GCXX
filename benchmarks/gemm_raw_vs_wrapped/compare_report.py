#!/usr/bin/env python3
"""Pair the raw-GemmEx and wrapped matrix_product benchmark rows into one
side-by-side table with the wrapper delta and overhead called out.

google benchmark's console reporter lists one benchmark per row and its
bundled tools/compare.py only pairs same-named benchmarks across two runs,
so neither shows raw-vs-wrapped next to each other. Feed this script the
JSON output instead:

  ./bench_gemm_raw_vs_wrapped --benchmark_format=json \
      --benchmark_out=run.json
  python3 compare_report.py run.json        # or: compare_report.py < run.json

Rows are paired by (family, arg): everything from the IssueOnly/WithSync
token onward must match between the BM_Raw* and BM_Wrapped* families.
"""

import json
import re
import sys

RAW_PREFIX = "BM_Raw"
WRAPPED_PREFIX = "BM_Wrapped"
FAMILY_RE = re.compile(r"_(IssueOnly|WithSync)/(.+)$")

TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}


def load_entries(stream):
    data = json.load(stream)
    entries = {}  # (variant, family, arg) -> [real_time us, items/s or None]
    for row in data.get("benchmarks", []):
        if row.get("run_type", "iteration") != "iteration":
            continue  # skip aggregate rows from --benchmark_repetitions
        name = row["name"]
        if name.startswith(RAW_PREFIX):
            variant = "raw"
        elif name.startswith(WRAPPED_PREFIX):
            variant = "wrapped"
        else:
            continue
        match = FAMILY_RE.search(name)
        if not match:
            continue
        family, arg = match.groups()
        # Later rows for the same key (manual repetitions) win.
        entries[(variant, family, arg)] = (
            row["real_time"] * TO_US[row["time_unit"]],
            row.get("items_per_second"),
        )
    return entries


def arg_sort_key(arg):
    return (0, int(arg), "") if arg.isdigit() else (1, 0, arg)


def fmt_rate(items_per_second):
    if items_per_second is None:
        return "-"
    if items_per_second >= 1e12:
        return f"{items_per_second / 1e12:.2f} T"
    if items_per_second >= 1e9:
        return f"{items_per_second / 1e9:.1f} G"
    return f"{items_per_second / 1e6:.1f} M"


def main():
    source = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    entries = load_entries(source)

    families = []
    pairs = {}
    for (variant, family, arg), value in entries.items():
        if family not in families:
            families.append(family)
        if (family, arg) not in pairs:
            pairs[(family, arg)] = {}
        pairs[(family, arg)][variant] = value

    for family in families:
        rows = [(arg, v) for (fam, arg), v in pairs.items() if fam == family]
        rows.sort(key=lambda kv: arg_sort_key(kv[0]))
        has_rate = any(v.get("wrapped", (None, None))[1] for _, v in rows)

        print(f"\n### {family}\n")
        header = "| arg | raw | wrapped | delta | overhead |"
        if has_rate:
            header = "| arg | raw | wrapped | delta | overhead | rate raw | rate wrapped |"
        print(header)
        print("|---" * (header.count("|") - 1) + "|")
        for arg, v in rows:
            if "raw" not in v or "wrapped" not in v:
                print(f"WARN: unpaired row {family}/{arg}", file=sys.stderr)
                continue
            raw_us, raw_rate = v["raw"]
            wrapped_us, wrapped_rate = v["wrapped"]
            delta = wrapped_us - raw_us
            overhead = 100.0 * delta / raw_us
            cells = [
                arg,
                f"{raw_us:.2f} us",
                f"{wrapped_us:.2f} us",
                f"{delta:+.2f} us",
                f"{overhead:+.1f}%",
            ]
            if has_rate:
                cells += [fmt_rate(raw_rate), fmt_rate(wrapped_rate)]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
