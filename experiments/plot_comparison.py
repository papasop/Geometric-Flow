"""Plot Adam vs Hybrid comparison charts from benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


def as_float(row: Dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return 0.0


def read_rows(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source"] = str(path)
                rows.append(row)
    return rows


def select_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    selected = []
    adam = [row for row in rows if row.get("optimizer") == "adam"]
    if adam:
        selected.append(adam[-1])

    geometric = [row for row in rows if row.get("optimizer") == "geometric"]
    if geometric:
        selected.append(geometric[-1])

    hybrids = [row for row in rows if row.get("optimizer", "").startswith("hybrid")]
    if hybrids:
        best_hybrid = max(hybrids, key=lambda row: as_float(row, "mean_accuracy", "final_accuracy"))
        selected.append(best_hybrid)
    return selected


def scale(value: float, max_value: float, height: float) -> float:
    if max_value <= 0:
        return 0.0
    return value / max_value * height


def render_svg(rows: List[Dict[str, str]], out: Path) -> None:
    metrics = [
        ("Accuracy", "mean_accuracy", "final_accuracy"),
        ("Loss", "mean_loss", "final_loss"),
        ("Seconds", "mean_seconds", "train_seconds"),
    ]
    width = 860
    height = 420
    margin_left = 70
    chart_top = 45
    chart_height = 250
    group_width = 250
    bar_width = 45
    colors = {
        "adam": "#4f7cff",
        "geometric": "#b05cff",
        "hybrid": "#12a37f",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="28" font-family="Arial" font-size="20" font-weight="700">GeoFlow Adam vs Hybrid Comparison</text>',
    ]

    for metric_index, metric in enumerate(metrics):
        label, primary, fallback = metric
        x0 = margin_left + metric_index * group_width
        values = [(row, as_float(row, primary, fallback)) for row in rows]
        max_value = max([value for _, value in values] + [1e-9])
        parts.append(
            f'<text x="{x0}" y="{chart_top + chart_height + 52}" font-family="Arial" '
            f'font-size="14" font-weight="700">{label}</text>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{chart_top}" x2="{x0}" y2="{chart_top + chart_height}" stroke="#dddddd"/>'
        )
        parts.append(
            f'<line x1="{x0}" y1="{chart_top + chart_height}" x2="{x0 + 190}" y2="{chart_top + chart_height}" stroke="#dddddd"/>'
        )
        for row_index, (row, value) in enumerate(values):
            optimizer = row.get("optimizer", "unknown")
            color_key = "hybrid" if optimizer.startswith("hybrid") else optimizer
            bar_height = scale(value, max_value, chart_height)
            x = x0 + 20 + row_index * (bar_width + 15)
            y = chart_top + chart_height - bar_height
            parts.append(
                f'<rect x="{x}" y="{y:.2f}" width="{bar_width}" height="{bar_height:.2f}" '
                f'fill="{colors.get(color_key, "#777777")}" rx="3"/>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2}" y="{y - 7:.2f}" font-family="Arial" '
                f'font-size="11" text-anchor="middle">{value:.3f}</text>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2}" y="{chart_top + chart_height + 18}" '
                f'font-family="Arial" font-size="10" text-anchor="middle">{optimizer}</text>'
            )

    parts.append('<text x="24" y="395" font-family="Arial" font-size="12" fill="#555555">')
    parts.append("Hybrid row is selected as the highest-accuracy hybrid configuration in the input CSV.")
    parts.append("</text>")
    parts.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="Benchmark CSV files to plot")
    parser.add_argument("--out", type=Path, default=Path("artifacts/comparison.svg"))
    args = parser.parse_args()

    rows = select_rows(read_rows(args.csv))
    if not rows:
        raise RuntimeError("no Adam, geometric, or hybrid rows found")
    render_svg(rows, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
