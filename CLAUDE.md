# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ScanSnap document processing system. The repository manages scanned PDFs from a ScanSnap scanner, with documents stored in date-stamped incoming folders (format: `MmmDD-MmmDD-Incoming/`). The PDF files use timestamp-based naming (format: `YYYYMMDDHHMMSS.pdf`).

## Repository Structure

### Core Files
- **process_scans.py**: Main processing script with hybrid mode support
- **prompts.py**: Model-specific LLM prompt templates
- **quality_validators.py**: Quality scoring and result merging logic
- **categories.md**: Allowed category list and categorization rules
- **CLAUDE.md**: This file - project documentation

### Directories
- **Incoming/**: PDF files to be processed
- **support/**: Helper scripts (tests, comparisons, utilities)
- **Claude-DOC/**: Documentation files (strategy, summaries, analysis)

### PDF Naming Convention
- Files are named with scan timestamps: `YYYYMMDDHHMMSS.pdf`
- Some files use descriptive names: `CompanyName_Date_Description.pdf`

## Setup - DONE

1. Set up a virtual environment:
   ```bash
   uv venv
   ```
2. Install Python dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Configure Anthropic API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your API key from https://console.anthropic.com/settings/keys
   ```

## Processing Scanned Documents

The `process_scans.py` script analyzes PDF files and extracts structured information using LLM APIs.

### Processing Modes

**1. Ollama Mode (Default - Free, Local)**
```bash
uv run process_scans.py --input Incoming --output scan_summary.csv
```
Uses local llama3:8b model via Ollama. Fast, free, conservative categorization.

**2. Anthropic Mode (High Quality)**
```bash
uv run process_scans.py --anthropic --input Incoming --output scan_summary.csv
```
Uses Claude Haiku via Anthropic API. Best quality, but costs API credits.

**3. Hybrid Mode (Recommended - Cost-Effective)**
```bash
uv run process_scans.py --hybrid --input Incoming --output scan_summary.csv
```
Smart combination: Ollama first, then Anthropic refinement for low-quality results.
- 50-70% cost reduction vs pure Anthropic
- Near-Anthropic quality with Ollama efficiency
- Configurable quality threshold: `--threshold 0.6` (default)

### What It Extracts

For each PDF document:
- **Originator**: Company/organization that created the document
- **Date**: The document date (not the scan date) in YYYY-MM-DD format
- **Summary**: Concise description (max 60 characters)
- **Category**: 1-4 categories from `categories.md` (e.g., "banking-home-sandiego")
- **Suggested Filename**: YYYYMMDDTHHMMSS--description__category format

**Output CSV format:**
- `filename`: Original PDF filename
- `originator`: Company or organization name
- `date`: Document date
- `summary`: Brief description
- `category`: Categories (dash-separated, sorted)
- `suggested_filename`: Recommended new filename

## Linting, Testing and Revision Control

Each time you make a change, run the following:

```bash
uv run flake8
uv run pytest support/test_process_scans.py
```

**IMPORTANT**: Only commit changes if there are no lint or test issues. All commits should have meaningful commit messages.

**Note**: CSV output files (*.csv) are gitignored and should never be committed.
