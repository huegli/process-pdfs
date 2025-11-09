#!/usr/bin/env python3
"""
Process scanned PDF files and extract structured information using Claude API.

This script reads PDF files from the Sept07-Nov09-Incoming/ directory, extracts
text content, and uses Claude to identify the originator, document date, and
generate a summary for each document.
"""

import os
import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import pdfplumber
from anthropic import Anthropic
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


def analyze_document(client: Anthropic, text: str, filename: str) -> Optional[Dict[str, str]]:
    """
    Use Claude to analyze document text and extract structured information.

    Args:
        client: Anthropic API client
        text: Document text content
        filename: Name of the PDF file

    Returns:
        Dictionary with originator, date, and summary, or None if analysis fails
    """
    if not text or len(text.strip()) < 50:
        print(f"  ⚠ Insufficient text extracted from {filename}")
        return None

    prompt = f"""Analyze this document and extract the following information in JSON format:

1. "originator": The company, organization, or entity that created/sent this document (look for letterheads, logos, company names)
2. "date": The primary document date (NOT the scan date or filename date). This could be:
   - The document creation date
   - A due date (for bills/invoices)
   - A statement date
   - Any other relevant date shown on the document
   Format as YYYY-MM-DD if possible, or the format shown in the document
3. "summary": A concise summary of what this document is about. IMPORTANT: Keep this to 60 characters or less.

If you cannot determine any field with confidence, use "Unknown" for that field.

Document text:
{text[:4000]}

Respond with ONLY a JSON object in this exact format:
{{"originator": "...", "date": "...", "summary": "..."}}"""

    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        # Handle cases where Claude might wrap JSON in markdown code blocks
        if response_text.startswith("```"):
            # Extract JSON from code block
            lines = response_text.split("\n")
            json_lines = [line for line in lines if not line.startswith("```")]
            response_text = "\n".join(json_lines).strip()

        result = json.loads(response_text)
        return result

    except json.JSONDecodeError as e:
        print(f"  ⚠ Failed to parse JSON response for {filename}: {e}")
        print(f"  Response was: {response_text[:200]}")
        return None
    except Exception as e:
        print(f"  ⚠ Error analyzing {filename}: {e}")
        return None


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
                              file_creation_time: datetime) -> str:
    """
    Create a suggested filename based on document metadata.

    Format: YYYYMMDDTHHMMSS--OriginatorDescription__scansnap

    Args:
        originator: Document originator
        date_str: Document date (will be parsed to YYYYMMDD)
        summary: Document summary (should be 60 characters or less from LLM)
        file_creation_time: File creation timestamp

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

    # Build the final filename
    return f"{date_part}T{time_part}--{description}__scansnap"


def process_pdfs(incoming_dir: Path, output_csv: Path):
    """
    Process all PDFs in the incoming directory and create a CSV summary.

    Args:
        incoming_dir: Path to directory containing PDF files
        output_csv: Path where the output CSV should be saved
    """
    # Initialize Anthropic client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables. "
                        "Please create a .env file with your API key.")

    client = Anthropic(api_key=api_key)

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

        # Get file creation time
        file_creation_time = datetime.fromtimestamp(pdf_path.stat().st_ctime)

        # Extract text
        text = extract_text_from_pdf(pdf_path)

        if not text:
            print(f"  ⚠ No text extracted, skipping")
            originator = "Unknown (no text)"
            date = "Unknown"
            summary = "Could not extract text from PDF"
            suggested_filename = create_suggested_filename(
                originator, date, summary, file_creation_time
            )
            results.append({
                "filename": pdf_path.name,
                "originator": originator,
                "date": date,
                "summary": summary,
                "suggested_filename": suggested_filename
            })
            continue

        # Analyze with Claude
        analysis = analyze_document(client, text, pdf_path.name)

        if analysis:
            originator = analysis.get("originator", "Unknown")
            date = analysis.get("date", "Unknown")
            summary = analysis.get("summary", "Unknown")
            suggested_filename = create_suggested_filename(
                originator, date, summary, file_creation_time
            )
            results.append({
                "filename": pdf_path.name,
                "originator": originator,
                "date": date,
                "summary": summary,
                "suggested_filename": suggested_filename
            })
            print(f"  ✓ {originator} - {date}")
        else:
            originator = "Unknown (analysis failed)"
            date = "Unknown"
            summary = "Failed to analyze document"
            suggested_filename = create_suggested_filename(
                originator, date, summary, file_creation_time
            )
            results.append({
                "filename": pdf_path.name,
                "originator": originator,
                "date": date,
                "summary": summary,
                "suggested_filename": suggested_filename
            })

        print()

    # Write results to CSV
    print(f"Writing results to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'originator', 'date', 'summary', 'suggested_filename']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Successfully processed {len(results)} files")
    print(f"✓ Results saved to {output_csv}")


def main():
    """Main entry point for the script."""
    # Define paths
    script_dir = Path(__file__).parent
    incoming_dir = script_dir / "Sept07-Nov09-Incoming"
    output_csv = script_dir / "scan_summary.csv"

    # Verify incoming directory exists
    if not incoming_dir.exists():
        print(f"Error: Directory not found: {incoming_dir}")
        return

    # Process PDFs
    try:
        process_pdfs(incoming_dir, output_csv)
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
