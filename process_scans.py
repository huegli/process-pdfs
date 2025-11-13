#!/usr/bin/env python3
"""
Process scanned PDF files and extract structured information using LLM API.

This script reads PDF files from the Sept07-Nov09-Incoming/ directory, extracts
text content, and uses an LLM (Ollama or Anthropic) to identify the originator,
document date, and generate a summary for each document.
"""

import os
import csv
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import pdfplumber
from anthropic import Anthropic
import ollama
from dotenv import load_dotenv

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
    client: Any, prompt: str, use_ollama: bool, max_tokens: int = 500
) -> str:
    """
    Call either Ollama or Anthropic API based on configuration.

    Args:
        client: Either Anthropic client or None for Ollama
        prompt: The prompt to send to the LLM
        use_ollama: If True, use Ollama API; otherwise use Anthropic
        max_tokens: Maximum tokens for response (Anthropic only)

    Returns:
        The LLM response text
    """
    if use_ollama:
        response = ollama.chat(
            model='llama3:8b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    else:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()


def analyze_document(
    client: Any, text: str, filename: str, use_ollama: bool = True
) -> Optional[Dict[str, str]]:
    """
    Use LLM to analyze document text and extract structured information.

    Args:
        client: LLM API client (Anthropic) or None for Ollama
        text: Document text content
        filename: Name of the PDF file
        use_ollama: If True, use Ollama; otherwise use Anthropic

    Returns:
        Dictionary with originator, date, and summary, or None if analysis fails
    """
    if not text or len(text.strip()) < 50:
        print(f"  ⚠ Insufficient text extracted from {filename}")
        return None

    prompt = f"""Analyze this document and extract the following information in \
JSON format:

1. "originator": The PRIMARY company, organization, or entity that created/sent \
this document.
   - Extract ONLY the main company name (e.g., "Chase", "Vanguard", "PG&E")
   - Do NOT include full legal names, addresses, or plan names
   - Look for letterheads, logos, and company names at the top of the document
   - For medical offices, use just the doctor's name (e.g., "Dr. Smith")

2. "date": The primary document date (NOT the scan date or filename date).
   - This could be: document date, due date, statement date, or billing date
   - Format as YYYY-MM-DD if possible
   - IMPORTANT: Only use dates that appear IN THE DOCUMENT TEXT
   - Do NOT make up dates or use dates from the future
   - Verify the year makes sense (should be between 2015-2025)
   - If no clear date is visible, use "Unknown"

3. "summary": A concise summary of what this document is about.
   - MAXIMUM 60 characters including spaces
   - Focus on document type (e.g., "Credit card statement", "Medical bill")
   - Do NOT include names, amounts, or excessive detail

If you cannot determine any field with confidence, use "Unknown" for that field.

Document text:
{text[:4000]}

Respond with ONLY a JSON object in this exact format:
{{"originator": "...", "date": "...", "summary": "..."}}"""

    try:
        response_text = call_llm(client, prompt, use_ollama, max_tokens=500)

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
    categories_content: str,
    use_ollama: bool = True,
    extra_category: Optional[str] = None
) -> Optional[str]:
    """
    Use LLM to categorize document text based on allowed categories.

    Args:
        client: LLM API client (Anthropic) or None for Ollama
        text: Document text content
        filename: Name of the PDF file
        categories_content: Content of the categories.md file
        use_ollama: If True, use Ollama; otherwise use Anthropic
        extra_category: Optional extra category to append to all results

    Returns:
        Category string (e.g., "banking" or "medical")
    """
    if not text or len(text.strip()) < 50:
        print(f"  ⚠ Insufficient text for categorization of {filename}")
        base_category = "reviewcategory"
        if extra_category:
            return "-".join(sorted([base_category, extra_category]))
        return base_category

    prompt = f"""You are a document categorization system. Your task is to \
categorize the following document based on the rules and allowed categories \
provided.

{categories_content}

Document text to categorize:
{text[:4000]}

CATEGORIZATION STRATEGY:
1. Identify the PRIMARY purpose of this document (choose ONE main category)
2. Add 1-2 secondary categories ONLY if they are directly relevant
3. Prefer FEWER categories over MORE categories
4. Aim for 1-3 categories total (MAXIMUM 4 if absolutely necessary)

CATEGORY SELECTION RULES:
- "medical" - Only for medical bills, doctor visits, prescriptions, medical records
- "banking" - Only for bank statements, checks, deposits (NOT credit cards)
- "creditcard" - Only for actual credit card statements (NOT other invoices)
- "insurance" - Only for insurance policies, claims, EOBs
- "home" or location tags - Only if document is about property/residence
- "education" - Only for school-related documents (tuition, grades, etc.)
- Special names (lucy, mikhaila, stephanie, vincent, kahlea) - ONLY if the \
document is specifically ABOUT or FOR that person

IMPORTANT RESTRICTIONS:
- Do NOT add "creditcard" to every invoice or bill
- Do NOT add "education" to financial documents unless they're school-related
- Do NOT add location tags unless the document relates to a specific property
- Do NOT add person names unless the document is specifically for that person

CRITICAL RULES:
1. MAXIMUM 4 categories - prefer 2-3 categories
2. NO duplicate categories - each category should appear only once
3. NO trailing dashes - "education-" is WRONG, "education" is CORRECT
4. Use ONLY words from the allowed categories list above
5. If no applicable category is found, use "reviewcategory"
6. Sort alphabetically and concatenate with '-'

IMPORTANT: Respond with ONLY a single line containing the category string \
(e.g., "banking" or "medical" or "home-sandiego").
Do NOT include any explanation, additional text, or multiple lines. Just the \
category string on one line."""

    try:
        response_text = call_llm(client, prompt, use_ollama, max_tokens=100)

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
                # Re-enforce max 4 after adding extra category
                if len(unique_categories) > 4:
                    unique_categories = unique_categories[:4]

        # Sort and rejoin
        response_text = "-".join(sorted(unique_categories))

        return response_text

    except Exception as e:
        print(f"  ⚠ Error categorizing {filename}: {e}")
        base_category = "reviewcategory"
        if extra_category:
            return "-".join(sorted([base_category, extra_category]))
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


def process_pdfs(
    incoming_dir: Path,
    output_csv: Path,
    use_ollama: bool = True,
    extra_category: Optional[str] = None
):
    """
    Process all PDFs in the incoming directory and create a CSV summary.

    Args:
        incoming_dir: Path to directory containing PDF files
        output_csv: Path where the output CSV should be saved
        use_ollama: If True, use Ollama API; otherwise use Anthropic
        extra_category: Optional extra category to append to all documents
    """
    # Initialize API client
    client = None
    if use_ollama:
        print("Using Ollama API with llama3:8b model")
        # No client needed for Ollama, it uses the ollama module directly
    else:
        print("Using Anthropic API with Claude")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment variables. "
                "Please create a .env file with your API key."
            )
        client = Anthropic(api_key=api_key)

    # Read categories.md file
    script_dir = Path(__file__).parent
    categories_file = script_dir / "categories.md"

    if not categories_file.exists():
        raise ValueError(f"categories.md file not found at {categories_file}")

    with open(categories_file, 'r', encoding='utf-8') as f:
        categories_content = f.read()

    # Get all PDF files
    pdf_files = sorted(incoming_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {incoming_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files to process\n")

    # Process each PDF and collect results
    results = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")

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
                # Fall back to file system creation time
                file_creation_time = datetime.fromtimestamp(
                    pdf_path.stat().st_ctime
                )
        except ValueError:
            # Fall back to file system creation time if parsing fails
            file_creation_time = datetime.fromtimestamp(
                pdf_path.stat().st_ctime
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

        # Analyze with LLM
        analysis = analyze_document(client, text, pdf_path.name, use_ollama)

        # Categorize the document
        category = categorize(
            client, text, pdf_path.name, categories_content,
            use_ollama, extra_category
        )
        if not category:
            category = "reviewcategory"
            if extra_category:
                category = "-".join(sorted([category, extra_category]))

        if analysis:
            originator = analysis.get("originator", "Unknown")
            date = analysis.get("date", "Unknown")
            summary = analysis.get("summary", "Unknown")
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
            print(f"  ✓ {originator} - {date} - {category}")
        else:
            originator = "Unknown (analysis failed)"
            date = "Unknown"
            summary = "Failed to analyze document"
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


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process scanned PDF files and extract structured "
                    "information using LLM API."
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local Ollama API with llama3:8b model (default)'
    )
    parser.add_argument(
        '--anthropic',
        action='store_true',
        help='Use Anthropic API with Claude'
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
        default='Incoming',
        help='Directory from which to read PDF files (default: Incoming)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scan_summary.csv',
        help='Output CSV filename (default: scan_summary.csv)'
    )
    args = parser.parse_args()

    # Determine which API to use
    # If --anthropic is specified, use Anthropic; otherwise use Ollama (default)
    use_ollama = not args.anthropic

    # Define paths
    script_dir = Path(__file__).parent

    # Handle input directory - can be absolute or relative to script directory
    input_path = Path(args.input)
    if input_path.is_absolute():
        incoming_dir = input_path
    else:
        incoming_dir = script_dir / args.input

    # Handle output file - can be absolute or relative to script directory
    output_path = Path(args.output)
    if output_path.is_absolute():
        output_csv = output_path
    else:
        output_csv = script_dir / args.output

    # Verify incoming directory exists
    if not incoming_dir.exists():
        print(f"Error: Directory not found: {incoming_dir}")
        return 1

    # Process PDFs
    try:
        process_pdfs(incoming_dir, output_csv, use_ollama, args.category)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
