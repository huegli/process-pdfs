#!/usr/bin/env python3
"""
Process scanned PDF files and extract structured information using LLM API.

This script reads PDF files from directories, extracts text content, and uses
LLMs (Ollama and/or Anthropic) to identify the originator, document date,
and generate a summary for each document.

Supports three modes:
- Ollama only (--local): Fast, free, local processing
- Anthropic only (--anthropic): High quality, API-based processing
- Hybrid (--hybrid): Ollama first, Anthropic refinement for uncertain results
"""

import os
import csv
import json
import re
import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import pdfplumber
from anthropic import Anthropic
import ollama
from dotenv import load_dotenv
from tqdm import tqdm

# Import new modules
from .prompts import get_analysis_prompt, get_categorization_prompt
from .quality_validators import (
    should_refine_with_anthropic,
    merge_results
)

# Load environment variables
load_dotenv()


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 2) -> str:
    """
    Extract text from the first few pages of a PDF.

    Args:
        pdf_path: Path to the PDF file
        max_pages: Maximum number of pages to extract (default: 2)

    Returns:
        Extracted text content
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_parts = []
            for i, page in enumerate(pdf.pages[:max_pages]):
                text = page.extract_text()
                if text:
                    text_parts.append(f"--- Page {i+1} ---\n{text}")

            return "\n\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting text from {pdf_path.name}: {e}")
        return ""


def call_llm(
    client: Any,
    prompt: str,
    use_ollama: bool,
    max_tokens: int = 500,
    ollama_model: str = 'qwen2.5:7b',
    anthropic_model: str = 'claude-sonnet-4-5-20250929'
) -> str:
    """
    Call either Ollama or Anthropic API based on configuration.

    Args:
        client: Either Anthropic client or None for Ollama
        prompt: The prompt to send to the LLM
        use_ollama: If True, use Ollama API; otherwise use Anthropic
        max_tokens: Maximum tokens for response (Anthropic only)
        ollama_model: Ollama model to use (default: qwen2.5:7b)
        anthropic_model: Anthropic model to use (default: claude-sonnet-4-5-20250929)

    Returns:
        The LLM response text
    """
    if use_ollama:
        response = ollama.chat(
            model=ollama_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    else:
        message = client.messages.create(
            model=anthropic_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()


def analyze_document(
    client: Any,
    text: str,
    filename: str,
    use_ollama: bool = True,
    model_type: str = 'ollama',
    ollama_model: str = 'qwen2.5:7b',
    anthropic_model: str = 'claude-sonnet-4-5-20250929'
) -> Optional[Dict[str, str]]:
    """
    Use LLM to analyze document text and extract structured information.

    Args:
        client: LLM API client (Anthropic) or None for Ollama
        text: Document text content
        filename: Name of the PDF file
        use_ollama: If True, use Ollama; otherwise use Anthropic
        model_type: 'ollama' or 'anthropic' for model-specific prompts
        ollama_model: Ollama model to use (default: qwen2.5:7b)
        anthropic_model: Anthropic model to use (default: claude-sonnet-4-5-20250929)

    Returns:
        Dictionary with originator, date, and summary, or None if analysis fails
    """
    if not text or len(text.strip()) < 50:
        print(f"  ⚠ Insufficient text extracted from {filename}")
        return None

    # Use the new prompt system
    prompt = get_analysis_prompt(text, model_type=model_type)

    try:
        response_text = call_llm(
            client, prompt, use_ollama, max_tokens=500,
            ollama_model=ollama_model, anthropic_model=anthropic_model
        )

        # Parse JSON response
        # Handle cases where LLM might wrap JSON in markdown code blocks or add text
        if response_text.startswith("```"):
            # Extract JSON from code block
            lines = response_text.split("\n")
            json_lines = [line for line in lines if not line.startswith("```")]
            response_text = "\n".join(json_lines).strip()

        # Try to find JSON object in the response (handles extra text before/after)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            response_text = response_text[json_start:json_end]

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        print(f"  ⚠ Failed to parse JSON response for {filename}: {e}")
        print(f"  Response was: {response_text[:200]}")
        return None
    except Exception as e:
        print(f"  ⚠ Error analyzing {filename}: {e}")
        return None


def categorize(
    client: Any,
    text: str,
    filename: str,
    category_rules: str,
    allowed_categories: str,
    use_ollama: bool = True,
    model_type: str = 'ollama',
    extra_category: Optional[str] = None,
    ollama_model: str = 'qwen2.5:7b',
    anthropic_model: str = 'claude-sonnet-4-5-20250929'
) -> Optional[str]:
    """
    Use LLM to categorize document text based on allowed categories.

    Args:
        client: LLM API client (Anthropic) or None for Ollama
        text: Document text content
        filename: Name of the PDF file
        category_rules: Content of the category_rules.md file
        allowed_categories: Content of the allowed_categories.md file
        use_ollama: If True, use Ollama; otherwise use Anthropic
        model_type: 'ollama' or 'anthropic' for model-specific prompts
        extra_category: Optional extra category to append to all results
        ollama_model: Ollama model to use (default: qwen2.5:7b)
        anthropic_model: Anthropic model to use
            (default: claude-sonnet-4-5-20250929)

    Returns:
        Category string (e.g., "banking" or "medical")
    """
    if not text or len(text.strip()) < 50:
        print(f"  ⚠ Insufficient text for categorization of {filename}")
        base_category = "reviewcategory"
        return base_category

    # Use the new prompt system
    prompt = get_categorization_prompt(
        text, category_rules, allowed_categories, model_type=model_type
    )

    try:
        response_text = call_llm(
            client, prompt, use_ollama, max_tokens=100,
            ollama_model=ollama_model, anthropic_model=anthropic_model
        )

        # Remove any potential markdown formatting or extra whitespace
        response_text = response_text.replace('`', '').strip()

        # Take only the first line if multiple lines are returned
        response_text = response_text.split('\n')[0].strip()

        # Remove trailing dashes
        response_text = response_text.rstrip('-')

        # Split categories, remove duplicates, enforce max 4 rule
        categories = response_text.split('-')
        # Remove empty strings and duplicates while preserving order
        seen = set()
        unique_categories = []
        for cat in categories:
            cat = cat.strip()
            if cat and cat not in seen:
                seen.add(cat)
                unique_categories.append(cat)

        # Enforce maximum 4 categories
        if len(unique_categories) > 4:
            unique_categories = unique_categories[:4]

        # Add extra category if provided
        if extra_category:
            if extra_category not in unique_categories:
                unique_categories.append(extra_category)

        # Sort and rejoin
        response_text = "_".join(sorted(unique_categories))

        return response_text

    except Exception as e:
        print(f"  ⚠ Error categorizing {filename}: {e}")
        base_category = "reviewcategory"
        return base_category


def parse_date_to_yyyymmdd(date_str: str) -> str:
    """
    Parse a date string to YYYYMMDD format.

    Args:
        date_str: Date string in various formats

    Returns:
        Date in YYYYMMDD format or "REVIEWDATE" if unknown/unparseable
    """
    if not date_str or date_str.lower() in ["unknown", "n/a", ""]:
        return "REVIEWDATE"

    # Try common date formats
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y%m%d")
        except ValueError:
            continue

    # Try to extract YYYY-MM-DD pattern anywhere in the string
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"

    return "REVIEWDATE"


def create_suggested_filename(originator: str, date_str: str, summary: str,
                              file_creation_time: datetime, category: str = "scansnap") -> str:
    """
    Create a suggested filename based on document metadata.

    Format: YYYYMMDDTHHMMSS--OriginatorDescription__category

    Args:
        originator: Document originator
        date_str: Document date (will be parsed to YYYYMMDD)
        summary: Document summary (should be 60 characters or less from LLM)
        file_creation_time: File creation timestamp
        category: Category string (default: "scansnap")

    Returns:
        Suggested filename string
    """
    # Parse date to YYYYMMDD format
    date_part = parse_date_to_yyyymmdd(date_str)

    # Get timestamp from file creation time
    time_part = file_creation_time.strftime("%H%M%S")

    # Create description from originator + summary
    description = f"{originator} {summary}"

    # Replace spaces with dashes and remove special characters
    description = re.sub(r'[^\w\s-]', '', description)  # Remove special chars except dash
    description = re.sub(r'\s+', '-', description)       # Replace spaces with dashes
    description = re.sub(r'-+', '-', description)        # Collapse multiple dashes
    description = description.strip('-')                 # Remove leading/trailing dashes
    description = description.lower()                    # Convert to lowercase

    # Build the final filename
    return f"{date_part}T{time_part}--{description}__{category}"


def process_document_hybrid(
    text: str,
    filename: str,
    category_rules: str,
    allowed_categories: str,
    ollama_client: Any,
    anthropic_client: Any,
    extra_category: Optional[str] = None,
    quality_threshold: float = 0.6,
    ollama_model: str = 'qwen2.5:7b',
    anthropic_model: str = 'claude-sonnet-4-5-20250929'
) -> Dict[str, Any]:
    """
    Process a document using hybrid mode: Ollama first, Anthropic refinement.

    Args:
        text: Document text content
        filename: Name of the PDF file
        category_rules: Content of category_rules.md file
        allowed_categories: Content of allowed_categories.md file
        ollama_client: None (Ollama uses ollama module directly)
        anthropic_client: Anthropic client instance
        extra_category: Optional extra category to append
        quality_threshold: Threshold for Anthropic refinement (0.0-1.0)
        ollama_model: Ollama model to use (default: qwen2.5:7b)
        anthropic_model: Anthropic model to use
            (default: claude-sonnet-4-5-20250929)

    Returns:
        Dictionary with analysis results and metadata
    """
    # Phase 1: Process with Ollama (fast, conservative)
    print("  Phase 1: Ollama analysis...")
    ollama_analysis = analyze_document(
        ollama_client, text, filename, use_ollama=True, model_type='ollama',
        ollama_model=ollama_model, anthropic_model=anthropic_model
    )
    ollama_category = categorize(
        ollama_client, text, filename, category_rules, allowed_categories,
        use_ollama=True, model_type='ollama', extra_category=extra_category,
        ollama_model=ollama_model, anthropic_model=anthropic_model
    )

    if not ollama_analysis:
        ollama_analysis = {
            'originator': 'Unknown (no text)',
            'date': 'Unknown',
            'summary': 'Could not extract text from PDF'
        }

    if not ollama_category:
        ollama_category = 'reviewcategory'

    # Combine Ollama results
    ollama_result = {
        **ollama_analysis,
        'category': ollama_category
    }

    # Phase 2: Determine if Anthropic refinement is needed
    needs_refinement = should_refine_with_anthropic(ollama_result, quality_threshold)

    if not needs_refinement:
        print("  ✓ High quality result, skipping Anthropic refinement")
        return {
            'result': ollama_result,
            'used_anthropic': False,
            'source': 'ollama'
        }

    # Phase 3: Refine with Anthropic
    print("  Phase 2: Anthropic refinement needed...")
    anthropic_analysis = analyze_document(
        anthropic_client, text, filename, use_ollama=False,
        model_type='anthropic',
        ollama_model=ollama_model, anthropic_model=anthropic_model
    )
    anthropic_category = categorize(
        anthropic_client, text, filename, category_rules, allowed_categories,
        use_ollama=False, model_type='anthropic', extra_category=extra_category,
        ollama_model=ollama_model, anthropic_model=anthropic_model
    )

    if not anthropic_analysis:
        anthropic_analysis = {}
    if not anthropic_category:
        anthropic_category = ollama_category

    # Combine Anthropic results
    anthropic_result = {
        **anthropic_analysis,
        'category': anthropic_category
    } if anthropic_analysis else None

    # Phase 4: Merge results (best of both)
    final_result = merge_results(ollama_result, anthropic_result)

    return {
        'result': final_result,
        'used_anthropic': True,
        'source': 'hybrid',
        'ollama_result': ollama_result,
        'anthropic_result': anthropic_result
    }


def process_pdfs(
    incoming_dir: Path,
    output_csv: Path,
    use_ollama: bool = True,
    use_hybrid: bool = False,
    extra_category: Optional[str] = None,
    quality_threshold: float = 0.6,
    ollama_model: str = 'qwen2.5:7b',
    anthropic_model: str = 'claude-sonnet-4-5-20250929',
    use_recursive_search: bool = False
):
    """
    Process all PDFs in the incoming directory and create a CSV summary.

    Args:
        incoming_dir: Path to directory containing PDF files
        output_csv: Path where the output CSV should be saved
        use_ollama: If True, use Ollama API; otherwise use Anthropic
        use_hybrid: If True, use hybrid mode (Ollama + selective Anthropic)
        extra_category: Optional extra category to append to all documents
        quality_threshold: Quality threshold for hybrid mode (0.0-1.0)
        ollama_model: Ollama model to use (default: qwen2.5:7b)
        anthropic_model: Anthropic model to use (default: claude-sonnet-4-5-20250929)
        use_recursive_search: If True, search subdirectories recursively (default: False)
    """
    # Initialize API clients
    ollama_client = None  # Ollama uses the ollama module directly
    anthropic_client = None

    if use_hybrid:
        print(f"Using HYBRID mode: {ollama_model} + selective {anthropic_model} refinement")
        print(f"Quality threshold: {quality_threshold}")
        # Need Anthropic client for hybrid mode
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY required for hybrid mode. "
                "Please create a .env file with your API key."
            )
        anthropic_client = Anthropic(api_key=api_key)
        model_type = 'ollama'  # Start with Ollama
    elif use_ollama:
        print(f"Using Ollama API with {ollama_model} model")
        model_type = 'ollama'
    else:
        if "sonnet" in anthropic_model.lower():
            model_name = "Claude Sonnet 4.5"
        else:
            model_name = "Claude Haiku 4.5"
        print(f"Using Anthropic API with {model_name} ({anthropic_model})")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please create a .env file with your API key."
            )
        anthropic_client = Anthropic(api_key=api_key)
        model_type = 'anthropic'

    # For backward compatibility
    client = anthropic_client if not use_ollama else ollama_client

    # Read category files
    script_dir = Path(__file__).parent
    category_rules_file = script_dir / "category_rules.md"
    allowed_categories_file = script_dir / "allowed_categories.md"

    if not category_rules_file.exists():
        raise ValueError(
            f"category_rules.md file not found at {category_rules_file}"
        )
    if not allowed_categories_file.exists():
        raise ValueError(
            f"allowed_categories.md file not found at "
            f"{allowed_categories_file}"
        )

    with open(category_rules_file, 'r', encoding='utf-8') as f:
        category_rules = f.read()

    with open(allowed_categories_file, 'r', encoding='utf-8') as f:
        allowed_categories = f.read()

    # Get all PDF files - recursive or non-recursive based on flag
    if use_recursive_search:
        pdf_files = sorted(incoming_dir.glob("**/*.pdf"))
        search_mode = "recursively"
    else:
        pdf_files = sorted(incoming_dir.glob("*.pdf"))
        search_mode = "non-recursively"

    if not pdf_files:
        print(f"No PDF files found in {incoming_dir} ({search_mode})")
        return

    print(f"Found {len(pdf_files)} PDF files to process ({search_mode})\n")

    # Process each PDF and collect results
    results = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        print(f"Processing {pdf_path.name}...")

        # Extract timestamp from filename (format: YYYYMMDDHHMMSS.pdf)
        # If filename doesn't match pattern, fall back to file creation time
        filename_stem = pdf_path.stem  # Remove .pdf extension
        try:
            if len(filename_stem) == 14 and filename_stem.isdigit():
                # Parse YYYYMMDDHHMMSS format from filename
                file_creation_time = datetime.strptime(
                    filename_stem, "%Y%m%d%H%M%S"
                )
            else:
                # TODO: Try out using the earlier of ctime and mtime
                # Fall back to file system modification time
                file_creation_time = datetime.fromtimestamp(
                    pdf_path.stat().st_mtime
                )
        except ValueError:
            # Fall back to file system modification time if parsing fails
            file_creation_time = datetime.fromtimestamp(
                pdf_path.stat().st_time
            )

        # Extract text
        text = extract_text_from_pdf(pdf_path)

        if not text:
            print("  ⚠ No text extracted, skipping")
            originator = "Unknown (no text)"
            date = "Unknown"
            summary = "Could not extract text from PDF"
            category = "reviewcategory"
            suggested_filename = create_suggested_filename(
                originator, date, summary, file_creation_time, category
            )
            results.append({
                "filename": pdf_path.name,
                "originator": originator,
                "date": date,
                "summary": summary,
                "category": category,
                "suggested_filename": suggested_filename
            })
            continue

        # Process based on mode
        if use_hybrid:
            # Hybrid mode: Ollama + selective Anthropic refinement
            hybrid_result = process_document_hybrid(
                text,
                pdf_path.name,
                category_rules,
                allowed_categories,
                ollama_client,
                anthropic_client,
                extra_category,
                quality_threshold,
                ollama_model,
                anthropic_model
            )
            result = hybrid_result['result']
            originator = result.get("originator", "Unknown")
            date = result.get("date", "Unknown")
            summary = result.get("summary", "Unknown")
            category = result.get("category", "reviewcategory")

            # Show which mode was used
            source_marker = " [Anthropic]" if hybrid_result['used_anthropic'] else " [Ollama]"
            print(f"  ✓ {originator} - {date} - {category}{source_marker}")
        else:
            # Single-model mode (Ollama or Anthropic)
            analysis = analyze_document(
                client, text, pdf_path.name, use_ollama, model_type,
                ollama_model, anthropic_model
            )
            category = categorize(
                client, text, pdf_path.name, category_rules,
                allowed_categories, use_ollama, model_type, extra_category,
                ollama_model, anthropic_model
            )

            if not category:
                category = "reviewcategory"

            if analysis:
                originator = analysis.get("originator", "Unknown")
                date = analysis.get("date", "Unknown")
                summary = analysis.get("summary", "Unknown")
                print(f"  ✓ {originator} - {date} - {category}")
            else:
                originator = "Unknown (analysis failed)"
                date = "Unknown"
                summary = "Failed to analyze document"

        # Create suggested filename and append result
        suggested_filename = create_suggested_filename(
            originator, date, summary, file_creation_time, category
        )
        results.append({
            "filename": pdf_path.name,
            "originator": originator,
            "date": date,
            "summary": summary,
            "category": category,
            "suggested_filename": suggested_filename
        })

        print()

    # Write results to CSV
    print(f"Writing results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'filename', 'originator', 'date', 'summary',
            'category', 'suggested_filename'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Successfully processed {len(results)} files")
    print(f"✓ Results saved to {output_csv}")


def create_rename_script(
    results: List[Dict[str, str]],
    incoming_dir: Path,
    output_script: Path,
    output_dir: Optional[Path] = None
):
    """
    Create a bash script to rename PDFs based on suggested filenames.

    Args:
        results: List of processing results with filename and suggested_filename
        incoming_dir: Directory containing the original PDF files
        output_script: Path where the bash script should be saved
        output_dir: Optional directory to copy renamed files to (instead of in-place rename)
    """
    script_lines = ["#!/bin/bash", ""]

    for result in results:
        original_filename = result['filename']
        suggested_filename = result['suggested_filename']

        # Add .pdf extension if not present
        if not suggested_filename.endswith('.pdf'):
            suggested_filename += '.pdf'

        original_path = incoming_dir / original_filename

        if output_dir:
            # Copy to output directory with new name
            new_path = output_dir / suggested_filename
            script_lines.append(f'cp "{original_path}" "{new_path}"')
        else:
            # Rename in place
            new_path = incoming_dir / suggested_filename
            script_lines.append(f'mv "{original_path}" "{new_path}"')

        script_lines.append('')

    # Write the script
    with open(output_script, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))

    # Make script executable
    os.chmod(output_script, 0o755)

    print(f"✓ Rename script saved to {output_script}")
    print(f"  Run with: {output_script}")


def rename_files_directly(
    results: List[Dict[str, str]],
    incoming_dir: Path,
    output_dir: Optional[Path] = None
):
    """
    Directly rename PDFs based on suggested filenames.

    Args:
        results: List of processing results with filename and suggested_filename
        incoming_dir: Directory containing the original PDF files
        output_dir: Optional directory to copy renamed files to (instead of in-place rename)
    """

    for result in results:
        original_filename = result['filename']
        suggested_filename = result['suggested_filename']

        # Add .pdf extension if not present
        if not suggested_filename.endswith('.pdf'):
            suggested_filename += '.pdf'

        original_path = incoming_dir / original_filename

        if not original_path.exists():
            print(f"  ⚠ Warning: {original_filename} not found, skipping")
            continue

        if output_dir:
            # Copy to output directory with new name
            new_path = output_dir / suggested_filename
            shutil.copy2(original_path, new_path)
            print(f"  ✓ Copied {original_filename} -> {new_path}")
        else:
            # Rename in place
            new_path = incoming_dir / suggested_filename
            original_path.rename(new_path)
            print(f"  ✓ Renamed {original_filename} -> {suggested_filename}")

    print(f"\n✓ Successfully renamed {len(results)} files")


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process scanned PDF files and extract structured "
                    "information using LLM API."
    )
    parser.add_argument(
        'mode',
        nargs='?',
        choices=['csv', 'script', 'rename'],
        default='csv',
        help='Operation mode: csv (create CSV, default), script (create rename script), '
             'rename (rename files directly)'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local Ollama API with qwen2.5:7b model (default)'
    )
    parser.add_argument(
        '--anthropic',
        action='store_true',
        help='Use Anthropic API with Claude Sonnet 4.5'
    )
    parser.add_argument(
        '--ollama-model',
        type=str,
        default='qwen2.5:7b',
        help='Ollama model to use (default: qwen2.5:7b)'
    )
    parser.add_argument(
        '--anthropic-model',
        type=str,
        default='claude-sonnet-4-5-20250929',
        help='Anthropic model to use (default: claude-sonnet-4-5-20250929)'
    )
    parser.add_argument(
        '--lowcost',
        action='store_true',
        help='Use Claude Haiku 4.5 instead of Sonnet 4.5 (lower cost, slightly reduced quality)'
    )
    parser.add_argument(
        '--hybrid',
        action='store_true',
        help='Use hybrid mode: Ollama first, Anthropic refinement for low-quality results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Quality threshold for hybrid mode (0.0-1.0, default: 0.6)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Extra category tag to append to all documents'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Directory from which to read PDF files recursively '
             '(default: current directory, non-recursive)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output destination: CSV filename for csv mode (default: scan_summary.csv), '
             'script filename for script mode (default: rename_pdfs.sh), '
             'or output directory for rename/script mode (optional)'
    )
    args = parser.parse_args()

    # Handle --lowcost flag: override anthropic_model if specified
    if args.lowcost:
        # Override to use Haiku 4.5 unless user explicitly specified a different model
        if args.anthropic_model == 'claude-sonnet-4-5-20250929':
            args.anthropic_model = 'claude-haiku-4-5-20251001'

    # Determine which mode to use
    use_hybrid = args.hybrid
    use_ollama = not args.anthropic and not use_hybrid

    # Define paths
    script_dir = Path(__file__).parent

    # Handle input directory - determine if recursive search is needed
    # If --input is not specified, use current directory (non-recursive)
    # If --input is specified, use that directory (recursive)
    use_recursive_search = args.input is not None

    if args.input is None:
        # Default: current directory, non-recursive
        incoming_dir = Path.cwd()
    else:
        # Specified directory, recursive
        input_path = Path(args.input)
        if input_path.is_absolute():
            incoming_dir = input_path
        else:
            incoming_dir = script_dir / args.input

    # Determine output path based on mode
    mode = args.mode
    output_dir = None

    if args.output:
        output_path = Path(args.output)
        if output_path.is_absolute():
            output_destination = output_path
        else:
            output_destination = script_dir / args.output
    else:
        # Set defaults based on mode
        if mode == 'csv':
            output_destination = script_dir / 'scan_summary.csv'
        elif mode == 'script':
            output_destination = script_dir / 'rename_pdfs.sh'
        else:  # rename mode
            output_destination = None  # Rename in place

    # If output_destination is a directory (for rename/script mode), set output_dir
    if output_destination and output_destination.is_dir():
        output_dir = output_destination
        if mode == 'script':
            output_destination = script_dir / 'rename_pdfs.sh'
        else:
            output_destination = None  # For rename mode, we'll use output_dir

    # For csv mode, we always need a CSV file path
    if mode == 'csv':
        output_csv = output_destination
    else:
        # For script/rename modes, create a temporary CSV to collect results
        output_csv = script_dir / '.temp_scan_summary.csv'

    # Verify incoming directory exists
    if not incoming_dir.exists():
        print(f"Error: Directory not found: {incoming_dir}")
        return 1

    # Create output directory if needed
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Process PDFs
    try:
        process_pdfs(
            incoming_dir,
            output_csv,
            use_ollama,
            use_hybrid,
            args.category,
            args.threshold,
            args.ollama_model,
            args.anthropic_model,
            use_recursive_search
        )

        # Handle post-processing based on mode
        if mode in ['script', 'rename']:
            # Read the results from the CSV
            with open(output_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                results = list(reader)

            if mode == 'script':
                # Create rename script
                create_rename_script(results, incoming_dir, output_destination, output_dir)
            else:  # rename
                # Rename files directly
                print("\nRenaming files...")
                rename_files_directly(results, incoming_dir, output_dir)

            # Clean up temporary CSV
            if output_csv.name == '.temp_scan_summary.csv':
                output_csv.unlink()

    except Exception as e:
        print(f"Error: {e}")
        # Clean up temporary CSV on error
        is_temp_csv = (mode in ['script', 'rename'] and
                       output_csv.exists() and
                       output_csv.name == '.temp_scan_summary.csv')
        if is_temp_csv:
            output_csv.unlink()
        raise


if __name__ == "__main__":
    main()
