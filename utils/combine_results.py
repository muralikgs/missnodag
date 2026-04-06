#!/usr/bin/env python
"""Utility to concatenate parquet results files in a directory.

Given a directory containing one or more `.parquet` files with identical
schemas, this script concatenates them into a single parquet file and moves
the original files into a `results.indiv` subdirectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the parquet files to combine.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.parquet"),
        help="Name of the combined parquet file to write (default: results.parquet).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return parser.parse_args(argv)


def combine_parquet_files(directory: Path, output: Path, overwrite: bool) -> None:
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory.")

    parquet_files = sorted(directory.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {directory}")

    dataframes = [pd.read_parquet(parquet_path) for parquet_path in parquet_files]
    combined = pd.concat(dataframes, ignore_index=True)

    output_path = output if output.is_absolute() else directory / output
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {output_path} already exists. "
            "Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    results_dir = directory / "results.indiv"
    results_dir.mkdir(exist_ok=True)

    for original in parquet_files:
        destination = results_dir / original.name
        counter = 1
        while destination.exists():
            destination = results_dir / f"{original.stem}_{counter}{original.suffix}"
            counter += 1
        original.rename(destination)

    print(f"Combined {len(parquet_files)} files into {output_path}")
    print(f"Moved original files to {results_dir}")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        combine_parquet_files(args.directory, args.output, args.overwrite)
    except Exception as exc:  # noqa: BLE001 - surface error details to CLI
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())