# Process PDFs

A command-line tool for processing scanned PDF files and extracting structured information using LLMs (Ollama and Anthropic Claude).

## Features

- Extract document metadata (originator, date, summary, categories) from PDFs
- Support for multiple LLM backends:
  - **Ollama** (free, local): Qwen2.5:7b for fast local processing
  - **Anthropic Claude**: Sonnet 4.5 for highest quality, Haiku 4.5 for cost efficiency
  - **Hybrid mode**: Ollama first, Claude refinement for uncertain results
- Automatic filename suggestions based on document content
- Batch processing with progress bars
- Configurable category taxonomy

## Installation

### As a tool (recommended)

```bash
uv tool install process-pdfs
```

### For development

```bash
git clone <repository-url>
cd process-pdfs
uv pip install -e ".[dev]"
```

## Usage

### Basic usage

```bash
# Process PDFs using Ollama (default, free)
process-pdfs --input Incoming --output scan_summary.csv

# Use Anthropic Claude Sonnet 4.5 (highest quality)
process-pdfs --anthropic --input Incoming --output scan_summary.csv

# Use Anthropic Claude Haiku 4.5 (cost-effective)
process-pdfs --anthropic --lowcost --input Incoming --output scan_summary.csv

# Hybrid mode (recommended)
process-pdfs --hybrid --input Incoming --output scan_summary.csv
```

### Operation modes

The tool supports three operation modes:

- **csv** (default): Generate CSV with metadata
- **script**: Generate bash script to rename files
- **rename**: Directly rename files based on extracted metadata

```bash
# Generate rename script
process-pdfs script --input Incoming

# Rename files directly
process-pdfs rename --input Incoming
```

## Configuration

Create a `.env` file for Anthropic API key:

```bash
ANTHROPIC_API_KEY=your-api-key-here
```

For detailed documentation, see [CLAUDE.md](CLAUDE.md).

## Requirements

- Python 3.10+
- Ollama (for local processing)
- Anthropic API key (for Claude models)

## License

MIT
