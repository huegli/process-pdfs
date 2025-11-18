"""
LLM Prompt templates for document analysis and categorization.

This module provides model-specific prompts optimized for different LLM backends.
"""


def get_analysis_prompt(text: str, model_type: str = 'ollama') -> str:
    """
    Get analysis prompt optimized for the specified model type.

    Args:
        text: Document text to analyze
        model_type: 'ollama' for fast/conservative or 'anthropic' for detailed

    Returns:
        Formatted prompt string
    """
    if model_type == 'ollama':
        return _get_ollama_analysis_prompt(text)
    else:  # anthropic
        return _get_anthropic_analysis_prompt(text)


def get_categorization_prompt(
    text: str,
    category_rules: str,
    allowed_categories: str,
    model_type: str = 'ollama'
) -> str:
    """
    Get categorization prompt optimized for the specified model type.

    Args:
        text: Document text to categorize
        category_rules: Content of category_rules.md file
        allowed_categories: Content of allowed_categories.md file
        model_type: 'ollama' for conservative or 'anthropic' for detailed

    Returns:
        Formatted prompt string
    """
    if model_type == 'ollama':
        return _get_ollama_categorization_prompt(
            text, category_rules, allowed_categories
        )
    else:  # anthropic
        return _get_anthropic_categorization_prompt(
            text, category_rules, allowed_categories
        )


def _get_ollama_analysis_prompt(text: str) -> str:
    """Fast, conservative analysis prompt for Ollama."""
    return f"""Analyze this document and extract the following information in \
JSON format:

1. "originator": The PRIMARY company or organization name.
   - Use SHORT form (e.g., "Chase", "Vanguard", "PG&E")
   - For doctors: "Dr. LastName" (e.g., "Dr. Smith")
   - If unclear, use "Unknown"

2. "date": The main document date.
   - Look for: document date, due date, statement date
   - Use the format shown in the document
   - If a date range is given, use the last date in the range
   - Do not allow dates after 2025
   - If no clear date, use "Unknown"

3. "summary": Brief description of document type.
   - Keep under 60 characters
   - Examples: "Credit card statement", "Medical bill", "Utility bill"
   - If unclear, use "Unknown"

Document text:
{text[:4000]}

Respond with ONLY a JSON object:
{{"originator": "...", "date": "...", "summary": "..."}}"""


def _get_anthropic_analysis_prompt(text: str) -> str:
    """Detailed, precise analysis prompt for Anthropic."""
    return f"""Analyze this document and extract the following information in \
JSON format:

1. "originator": The PRIMARY company, organization, or entity.
   - Extract the main company name (e.g., "Chase", "Vanguard", "SDG&E")
   - For medical: Include full doctor name (e.g., "Dr. John Smith")
   - Look for letterheads, logos, company names at document top
   - Do NOT include addresses or full legal entity names

2. "date": The primary document date in YYYY-MM-DD format.
   CRITICAL REQUIREMENTS:
   - Find the main date IN THE DOCUMENT TEXT (statement date, due date, etc.)
   - Convert to YYYY-MM-DD format (e.g., "June 14, 2023" → "2023-06-14")
   - If a date range is given use the last date in the range
   - Verify year is between 2015-2025
   - Do NOT make up dates or use future dates
   - If no clear date exists in document, use "Unknown"

3. "summary": Concise, specific description (MAXIMUM 60 characters).
   - Be specific about document type and purpose
   - Examples: "Credit card statement", "Medical bill for office visit"
   - Do NOT include names, amounts, or excessive detail
   - Focus on WHAT the document is

Document text:
{text[:4000]}

Respond with ONLY a JSON object in this exact format:
{{"originator": "...", "date": "...", "summary": "..."}}"""


def _get_ollama_categorization_prompt(
    text: str,
    category_rules: str,
    allowed_categories: str
) -> str:
    """Conservative categorization prompt for Ollama."""
    return f"""Categorize this document based on the rules and allowed \
categories.

{category_rules}

{allowed_categories}

Document text:
{text[:4000]}

Respond with ONLY the category string on one line."""


def _get_anthropic_categorization_prompt(
    text: str,
    category_rules: str,
    allowed_categories: str
) -> str:
    """Detailed categorization prompt for Anthropic with validation."""
    return f"""Categorize this document based on the rules and allowed \
categories provided.

{category_rules}

{allowed_categories}

Document text:
{text[:4000]}

Respond with ONLY a single line containing the category string.
Do NOT include explanations."""
