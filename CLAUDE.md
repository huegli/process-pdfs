# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ScanSnap document processing system. The repository manages scanned PDFs from a ScanSnap scanner, with documents stored in date-stamped incoming folders (format: `MmmDD-MmmDD-Incoming/`). The PDF files use timestamp-based naming (format: `YYYYMMDDHHMMSS.pdf`).

## Repository Structure

### Package Structure
- **src/process_pdfs/**: Python package containing all modules
  - **cli.py**: Main CLI implementation with processing logic
  - **prompts.py**: Model-specific LLM prompt templates
  - **quality_validators.py**: Quality scoring and result merging logic
  - **categories.md**: Allowed category list and categorization rules
  - **__init__.py**: Package initialization
  - **__main__.py**: CLI entry point

### Legacy Files (for reference)
- **process_scans.py**: Original script (replaced by package)
- **prompts.py**, **quality_validators.py**, **categories.md**: Root-level copies

### Directories
- **Incoming/**: PDF files to be processed
- **support/**: Helper scripts (tests, comparisons, utilities)
- **Claude-DOC/**: Documentation files (strategy, summaries, analysis)
- **dist/**: Built packages (wheel and source distribution)

### PDF Naming Convention
- Files are named with scan timestamps: `YYYYMMDDHHMMSS.pdf`
- Some files use descriptive names: `CompanyName_Date_Description.pdf`

## Installation

### Option 1: Install as a CLI tool (Recommended)

Install the package globally as a command-line tool:

```bash
uv tool install .
```

This makes the `process-pdfs` command available system-wide.

### Option 2: Development Installation

For local development with the original scripts:

1. Set up a virtual environment:
   ```bash
   uv venv
   ```

2. Install dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

### Configuration

Configure Anthropic API key (required for `--anthropic` and `--hybrid` modes):

```bash
cp .env.example .env
# Edit .env and add your API key from https://console.anthropic.com/settings/keys
```

## Processing Scanned Documents

The `process-pdfs` command (or `uv run process_scans.py` for legacy usage) analyzes PDF files and extracts structured information using LLM APIs.

### Processing Modes

**1. Ollama Mode (Default - Free, Local)**
```bash
process-pdfs --input Incoming --output scan_summary.csv
# Or legacy: uv run process_scans.py --input Incoming --output scan_summary.csv
```
Uses local **Qwen2.5:7b** model via Ollama. Fast, free, excellent for structured data extraction.
- Optimized for document understanding and data extraction
- Runs comfortably on 8GB Mac Mini
- Better date parsing and categorization than Llama3:8b

**2. Anthropic Mode (High Quality)**
```bash
process-pdfs --anthropic --input Incoming --output scan_summary.csv
```
Uses **Claude Sonnet 4.5** via Anthropic API. Highest quality document extraction.
- Best-in-class for complex document understanding
- Superior accuracy for ambiguous or poorly-scanned documents
- $3 input / $15 output per million tokens

**2a. Anthropic Mode - Low Cost**
```bash
process-pdfs --anthropic --lowcost --input Incoming --output scan_summary.csv
```
Uses **Claude Haiku 4.5** via Anthropic API. Near-frontier performance at lower cost.
- Matches Sonnet 4 quality on many tasks
- 3x cheaper than Sonnet 4.5 ($1 input / $5 output per million tokens)
- 2x faster than Sonnet 4.5

**3. Hybrid Mode (Recommended - Best Balance)**
```bash
process-pdfs --hybrid --input Incoming --output scan_summary.csv
```
Smart combination: Qwen2.5:7b first, then Claude Sonnet 4.5 refinement for low-quality results.
- 50-70% cost reduction vs pure Anthropic
- Near-Anthropic quality with Ollama efficiency
- Configurable quality threshold: `--threshold 0.6` (default)
- Use `--lowcost` to refine with Haiku 4.5 instead of Sonnet 4.5

### Model Selection

Override default models with command-line flags:

**Ollama Models:**
```bash
# Use a different Ollama model (e.g., llama3:8b, mistral:7b)
process-pdfs --ollama-model llama3:8b
```

**Anthropic Models:**
```bash
# Use Claude Sonnet 4.5 (default for --anthropic)
process-pdfs --anthropic --anthropic-model claude-sonnet-4-5-20250929

# Use Claude Haiku 4.5 (lower cost option)
process-pdfs --anthropic --lowcost
# or explicitly:
process-pdfs --anthropic --anthropic-model claude-haiku-4-5-20251001
```

**Default Models:**
- **Ollama**: `qwen2.5:7b` (excellent document extraction, 8GB compatible)
- **Anthropic**: `claude-sonnet-4-5-20250929` (highest quality, default)
- **Anthropic (--lowcost)**: `claude-haiku-4-5-20251001` (3x cheaper, near-frontier performance)

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
