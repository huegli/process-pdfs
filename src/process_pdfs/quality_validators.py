"""
Quality validation functions for document analysis results.

These validators help determine if results need refinement with a higher-quality
LLM or if they are acceptable as-is.
"""

import re
from typing import Dict, Optional


def validate_date_format(date_str: str) -> float:
    """
    Validate date format and return quality score.

    Args:
        date_str: Date string to validate

    Returns:
        Quality score from 0.0 (poor) to 1.0 (excellent)
    """
    if not date_str or date_str.lower() in ['unknown', 'n/a', '']:
        return 0.0

    # Perfect: YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return 1.0

    # Good: Other standard formats
    if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):  # MM/DD/YYYY
        return 0.7
    if re.match(r'^\d{4}/\d{2}/\d{2}$', date_str):  # YYYY/MM/DD
        return 0.8

    # Acceptable: Month name formats
    if re.search(r'(January|February|March|April|May|June|July|August|'
                 r'September|October|November|December)', date_str):
        return 0.6
    if re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',
                 date_str):
        return 0.6

    # Poor: Date ranges or unclear formats
    if ' to ' in date_str or '-' in date_str:
        return 0.3

    # Unknown format
    return 0.2


def validate_category(category_str: str) -> bool:
    """
    Check if category needs review.

    Args:
        category_str: Category string to validate

    Returns:
        True if category needs review, False if acceptable
    """
    if not category_str:
        return True

    # Needs review if explicitly marked
    if 'reviewcategory' in category_str:
        return True

    # Check for trailing dashes (formatting issue)
    if category_str.endswith('-'):
        return True

    # Check for empty categories after splitting
    categories = category_str.split('-')
    if any(not cat.strip() for cat in categories):
        return True

    # Too many categories (likely over-tagged)
    if len(categories) > 3:
        return True

    return False


def validate_summary(summary_str: str) -> float:
    """
    Validate summary quality and return quality score.

    Args:
        summary_str: Summary string to validate

    Returns:
        Quality score from 0.0 (poor) to 1.0 (excellent)
    """
    if not summary_str or summary_str.lower() in ['unknown', 'n/a', '']:
        return 0.0

    # Check length (should be under 60 chars)
    length = len(summary_str)
    if length > 80:
        return 0.3  # Too long
    if length < 10:
        return 0.4  # Too short/generic

    # Penalize very generic summaries
    generic_terms = [
        'unknown', 'document', 'statement', 'n/a', 'could not',
        'failed to', 'unable to'
    ]
    lower_summary = summary_str.lower()
    if any(term in lower_summary for term in generic_terms):
        if length < 20:
            return 0.3  # Short and generic

    # Good summary: reasonable length, specific content
    if 20 <= length <= 60:
        return 1.0
    elif 60 < length <= 70:
        return 0.9
    else:
        return 0.7


def validate_originator(originator_str: str) -> float:
    """
    Validate originator quality and return quality score.

    Args:
        originator_str: Originator string to validate

    Returns:
        Quality score from 0.0 (poor) to 1.0 (excellent)
    """
    if not originator_str or originator_str.lower() in ['unknown', 'n/a']:
        return 0.0

    # Check for address-like content (poor quality)
    if any(indicator in originator_str.lower() for indicator in
           ['street', 'blvd', 'avenue', 'road', 'drive', 'california', 'ca ']):
        return 0.3

    # Check for overly long names (likely full legal entity)
    if len(originator_str) > 50:
        return 0.5

    # Good: Short, clean company name
    if len(originator_str) <= 30:
        return 1.0

    return 0.7


def should_refine_with_anthropic(
    result: Dict[str, str],
    threshold: float = 0.6
) -> bool:
    """
    Determine if a result should be refined with Anthropic API.

    Args:
        result: Dictionary with originator, date, summary, category
        threshold: Quality threshold (0.0-1.0). Below this = needs refinement

    Returns:
        True if result should be refined, False otherwise
    """
    # Always refine if category needs review
    if validate_category(result.get('category', '')):
        return True

    # Calculate overall quality score
    date_quality = validate_date_format(result.get('date', ''))
    summary_quality = validate_summary(result.get('summary', ''))
    originator_quality = validate_originator(result.get('originator', ''))

    # Weight the scores (date and summary are most important)
    overall_quality = (
        date_quality * 0.4 +
        summary_quality * 0.4 +
        originator_quality * 0.2
    )

    return overall_quality < threshold


def merge_results(
    ollama_result: Optional[Dict[str, str]],
    anthropic_result: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """
    Merge results from Ollama and Anthropic, taking best of both.

    Strategy:
    - Originator: Prefer Ollama (cleaner, shorter names)
    - Date: Prefer Anthropic if YYYY-MM-DD format, else Ollama
    - Summary: Prefer Anthropic (more detailed and specific)
    - Category: Use Ollama base, but validate against Anthropic

    Args:
        ollama_result: Result from Ollama analysis
        anthropic_result: Result from Anthropic analysis (may be None)

    Returns:
        Merged result dictionary
    """
    if not anthropic_result:
        return ollama_result or {}

    if not ollama_result:
        return anthropic_result

    merged = {}

    # Originator: Prefer Ollama (cleaner names)
    ollama_orig_quality = validate_originator(
        ollama_result.get('originator', '')
    )
    anthropic_orig_quality = validate_originator(
        anthropic_result.get('originator', '')
    )

    if ollama_orig_quality >= anthropic_orig_quality:
        merged['originator'] = ollama_result.get('originator', 'Unknown')
    else:
        merged['originator'] = anthropic_result.get('originator', 'Unknown')

    # Date: Prefer Anthropic if good format, otherwise Ollama
    anthropic_date_quality = validate_date_format(
        anthropic_result.get('date', '')
    )
    if anthropic_date_quality >= 0.8:  # YYYY-MM-DD or YYYY/MM/DD
        merged['date'] = anthropic_result.get('date', 'Unknown')
    else:
        # Use whichever has better quality
        ollama_date_quality = validate_date_format(
            ollama_result.get('date', '')
        )
        if ollama_date_quality >= anthropic_date_quality:
            merged['date'] = ollama_result.get('date', 'Unknown')
        else:
            merged['date'] = anthropic_result.get('date', 'Unknown')

    # Summary: Prefer Anthropic (more detailed)
    anthropic_summary_quality = validate_summary(
        anthropic_result.get('summary', '')
    )
    ollama_summary_quality = validate_summary(
        ollama_result.get('summary', '')
    )

    if anthropic_summary_quality >= ollama_summary_quality:
        merged['summary'] = anthropic_result.get('summary', 'Unknown')
    else:
        merged['summary'] = ollama_result.get('summary', 'Unknown')

    # Category: Prefer Ollama (more conservative)
    # But if Ollama has reviewcategory, try Anthropic
    ollama_cat = ollama_result.get('category', 'reviewcategory')
    anthropic_cat = anthropic_result.get('category', 'reviewcategory')

    if 'reviewcategory' in ollama_cat and 'reviewcategory' not in anthropic_cat:
        merged['category'] = anthropic_cat
    else:
        merged['category'] = ollama_cat

    return merged
