#!/usr/bin/env python3
"""
Compare the suggested_filename columns from Ollama and Anthropic CSV outputs.

This script creates a row-by-row comparison showing:
1. The original PDF filename
2. Ollama's suggested filename
3. Anthropic's suggested filename
4. Whether they match or differ
"""

import csv
import sys
from pathlib import Path


def load_csv_data(csv_path: Path) -> dict:
    """Load CSV and create a dictionary keyed by filename."""
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['filename']] = row
    return data


def compare_csvs(ollama_csv: Path, anthropic_csv: Path):
    """Compare two CSV files and show differences in suggested filenames."""
    ollama_data = load_csv_data(ollama_csv)
    anthropic_data = load_csv_data(anthropic_csv)

    # Get all unique filenames
    all_files = sorted(set(ollama_data.keys()) | set(anthropic_data.keys()))

    # Statistics
    total = len(all_files)
    matches = 0
    differs = 0
    missing = 0

    print("=" * 120)
    print("COMPARISON: Ollama (llama3:8b) vs Anthropic (Claude Haiku)")
    print("=" * 120)
    print()

    for filename in all_files:
        ollama_row = ollama_data.get(filename)
        anthropic_row = anthropic_data.get(filename)

        if not ollama_row or not anthropic_row:
            missing += 1
            print(f"⚠ MISSING: {filename}")
            if not ollama_row:
                print("   Missing from Ollama CSV")
            if not anthropic_row:
                print("   Missing from Anthropic CSV")
            print()
            continue

        ollama_suggested = ollama_row['suggested_filename']
        anthropic_suggested = anthropic_row['suggested_filename']

        if ollama_suggested == anthropic_suggested:
            matches += 1
            status = "✓ MATCH"
        else:
            differs += 1
            status = "✗ DIFFER"

        print(f"{status}: {filename}")
        print(f"  Ollama:     {ollama_suggested}")
        print(f"  Anthropic:  {anthropic_suggested}")

        # Show field-by-field differences
        if ollama_suggested != anthropic_suggested:
            print("  Details:")
            print(f"    Originator: Ollama='{ollama_row['originator']}' vs "
                  f"Anthropic='{anthropic_row['originator']}'")
            print(f"    Date:       Ollama='{ollama_row['date']}' vs "
                  f"Anthropic='{anthropic_row['date']}'")
            print(f"    Summary:    Ollama='{ollama_row['summary']}' vs "
                  f"Anthropic='{anthropic_row['summary']}'")
            print(f"    Category:   Ollama='{ollama_row['category']}' vs "
                  f"Anthropic='{anthropic_row['category']}'")

        print()

    # Summary statistics
    print("=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print(f"Total files:        {total}")
    print(f"Exact matches:      {matches} ({matches/total*100:.1f}%)")
    print(f"Differences:        {differs} ({differs/total*100:.1f}%)")
    print(f"Missing entries:    {missing}")
    print()


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    ollama_csv = script_dir / "scan_summary_ollama.csv"
    anthropic_csv = script_dir / "scan_summary_anthropic.csv"

    if not ollama_csv.exists():
        print(f"Error: {ollama_csv} not found")
        return 1

    if not anthropic_csv.exists():
        print(f"Error: {anthropic_csv} not found")
        return 1

    compare_csvs(ollama_csv, anthropic_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
