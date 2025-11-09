# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ScanSnap document processing system. The repository manages scanned PDFs from a ScanSnap scanner, with documents stored in date-stamped incoming folders (format: `MmmDD-MmmDD-Incoming/`). The PDF files use timestamp-based naming (format: `YYYYMMDDHHMMSS.pdf`).

## Repository Structure

- **Incoming folders**: Named with date ranges (e.g., `Sept07-Nov09-Incoming/`) contain scanned PDF documents
- **PDF naming**: Files are named with scan timestamps in format `YYYYMMDDHHMMSS.pdf`
- The repository is configured with two working directories:
  - Primary: `/Users/nikolai/Library/CloudStorage/Dropbox/Source/Claude/ScanSnap`
  - Secondary: `/Users/nikolai/Source/Scratch/Claude/ScanSnap`

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

The `process_scans.py` script analyzes PDF files and extracts structured information using Claude AI.

**Run the script:**
```bash
uv run process_scans.py
```

This will:
- Read all PDFs from `Sept07-Nov09-Incoming/`
- Extract text from the first 2 pages of each document
- Use Claude API to identify:
  - **Originator**: Company/organization that created the document
  - **Date**: The document date (not the scan date)
  - **Summary**: One-sentence description of the content
- Save results to `scan_summary.csv`

**Output CSV format:**
- `filename`: PDF filename
- `originator`: Company or organization name
- `date`: Document date in YYYY-MM-DD format (when possible)
- `summary`: One-sentence summary

## Development Environment

The user has LispWorks expertise available via the `lispworks` skill. When working with Common Lisp code, LispWorks CAPI, or LispWorks-specific features, invoke the skill using the Skill tool.
