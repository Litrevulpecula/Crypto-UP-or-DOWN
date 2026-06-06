#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

OUTPUT_COLUMNS = RAW_COLUMNS + ["is_missing", "source_file"]
PERIOD_RE = re.compile(r"(?P<symbol>[A-Z]+USDT)-1m-(?P<period>\d{4}-\d{2}(?:-\d{2})?)\.zip$")


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def normalize_row(row):
    if not row or not row[0].strip().isdigit():
        return None
    row = row[: len(RAW_COLUMNS)]
    if len(row) < len(RAW_COLUMNS):
        row.extend([""] * (len(RAW_COLUMNS) - len(row)))
    row[0] = str(normalize_timestamp_ms(row[0]))
    row[6] = str(normalize_timestamp_ms(row[6]))
    return row


def normalize_timestamp_ms(value: str) -> int:
    timestamp = int(float(value))
    if timestamp > 10**17:
        return timestamp // 1_000_000
    if timestamp > 10**14:
        return timestamp // 1_000
    return timestamp


def period_sort_key(path: Path):
    match = PERIOD_RE.match(path.name)
    if not match:
        return (9, path.name)
    period = match.group("period")
    # Daily files must be read after their monthly file so daily rows can fill
    # missing monthly ranges without disturbing normal month-level archives.
    kind = 1 if len(period) == 10 else 0
    return (period[:7], kind, period, path.name)


def iter_zip_rows(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if not names:
            return
        with zf.open(names[0]) as handle:
            text = (line.decode("utf-8", errors="replace") for line in handle)
            for row in csv.reader(text):
                normalized = normalize_row(row)
                if normalized is not None:
                    yield normalized


def load_rows(input_dir: Path):
    rows = {}
    duplicates = 0
    for zip_path in sorted(input_dir.glob("*.zip"), key=period_sort_key):
        for row in iter_zip_rows(zip_path):
            open_time = int(row[0])
            if open_time in rows:
                duplicates += 1
                continue
            rows[open_time] = row + ["0", zip_path.name]
    return rows, duplicates


def write_aligned(input_dir: Path, output_path: Path, start_ms: int, end_ms: int):
    rows, duplicates = load_rows(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    present = 0
    missing = 0
    first_missing = None
    missing_ranges = []
    open_missing_start = None
    previous_missing = None

    with gzip.open(output_path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)

        for open_time in range(start_ms, end_ms + 1, 60_000):
            row = rows.get(open_time)
            if row is None:
                missing += 1
                if first_missing is None:
                    first_missing = open_time
                if open_missing_start is None:
                    open_missing_start = open_time
                previous_missing = open_time
                close_time = open_time + 59_999
                writer.writerow([open_time, "", "", "", "", "", close_time, "", "", "", "", "", "1", ""])
            else:
                present += 1
                if open_missing_start is not None:
                    missing_ranges.append([open_missing_start, previous_missing])
                    open_missing_start = None
                    previous_missing = None
                writer.writerow(row)

    if open_missing_start is not None:
        missing_ranges.append([open_missing_start, previous_missing])

    outside_range = sum(1 for open_time in rows if open_time < start_ms or open_time > end_ms)
    return {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "rows": (end_ms - start_ms) // 60_000 + 1,
        "present_rows": present,
        "missing_rows": missing,
        "duplicate_rows_ignored": duplicates,
        "outside_range_rows": outside_range,
        "first_missing_open_time": first_missing,
        "missing_ranges": missing_ranges,
    }


def ms_to_iso(value):
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def add_readable_times(record):
    record = dict(record)
    record["first_missing_utc"] = ms_to_iso(record["first_missing_open_time"])
    record["missing_ranges_utc"] = [
        [ms_to_iso(start), ms_to_iso(end)] for start, end in record["missing_ranges"]
    ]
    return record


def main():
    parser = argparse.ArgumentParser(description="Align Binance 1m kline zip archives to one timeline.")
    parser.add_argument("--raw-root", default="raw_data", type=Path)
    parser.add_argument("--out-root", default="aligned_data", type=Path)
    parser.add_argument("--start", default="2022-01-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-04-23T23:59:00+00:00")
    parser.add_argument("--markets", default=None, help="Optional comma-separated market directory names.")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated symbols, e.g. BTCUSDT,ETHUSDT.")
    args = parser.parse_args()

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    market_filter = None if args.markets is None else {item.strip() for item in args.markets.split(",") if item.strip()}
    symbol_filter = None if args.symbols is None else {item.strip().upper() for item in args.symbols.split(",") if item.strip()}

    report = {
        "start_utc": start.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "end_utc": end.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "expected_rows_per_series": (end_ms - start_ms) // 60_000 + 1,
        "series": [],
    }

    for market_dir in sorted(path for path in args.raw_root.iterdir() if path.is_dir()):
        if market_filter is not None and market_dir.name not in market_filter:
            continue
        for symbol_dir in sorted(path for path in market_dir.iterdir() if path.is_dir()):
            if symbol_filter is not None and symbol_dir.name.upper() not in symbol_filter:
                continue
            input_dir = symbol_dir / "1m"
            if not input_dir.exists():
                continue
            output_path = args.out_root / market_dir.name / symbol_dir.name / "1m.csv.gz"
            record = write_aligned(input_dir, output_path, start_ms, end_ms)
            record["market"] = market_dir.name
            record["symbol"] = symbol_dir.name
            report["series"].append(add_readable_times(record))
            print(
                f"{market_dir.name}/{symbol_dir.name}: rows={record['rows']} "
                f"present={record['present_rows']} missing={record['missing_rows']} "
                f"duplicates={record['duplicate_rows_ignored']}"
            )

    args.out_root.mkdir(parents=True, exist_ok=True)
    report_name = "alignment_report.json" if market_filter is None and symbol_filter is None else "alignment_report_filtered.json"
    with (args.out_root / report_name).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
