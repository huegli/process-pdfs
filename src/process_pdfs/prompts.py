"""
LLM Prompt templates for document analysis and categorization.

This module provides model-specific prompts optimized for different LLM backends.
"""

import random


def generate_category_examples(allowed_categories: str, num_examples: int = 5) -> str:
    """
    Generate random category examples from the allowed categories list.

    Args:
        allowed_categories: Content of allowed_categories.md file
        num_examples: Number of examples to generate (default: 5)

    Returns:
        Formatted string with example category combinations
    """
    # Parse allowed categories
    categories = [line.strip() for line in allowed_categories.split('\n')
                  if line.strip() and not line.startswith('#')]

    if len(categories) < 2:
        return ""

    examples = []

    # Generate single category examples
    single_cats = random.sample(categories, min(2, len(categories)))
    for cat in single_cats:
        examples.append(f'- "{cat}"')

    # Generate multi-category examples (2-4 categories)
    for _ in range(num_examples - 2):
        num_cats = random.randint(2, min(4, len(categories)))
        selected = sorted(random.sample(categories, num_cats))
        examples.append(f'- "{"_".join(selected)}"')

    # Always include reviewcategory as an option
    examples.append('- "reviewcategory" (if no category fits)')

    return "Examples of valid category responses:\n" + "\n".join(examples)


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


def get_analysis_prompt_with_filename(
    text: str,
    filename: str,
    model_type: str = 'ollama'
) -> str:
    """
    Get analysis prompt that uses filename as a starting point.

    Args:
        text: Document text to analyze
        filename: Name of the PDF file
        model_type: 'ollama' for fast/conservative or 'anthropic' for detailed

    Returns:
        Formatted prompt string
    """
    if model_type == 'ollama':
        return _get_ollama_analysis_prompt_with_filename(text, filename)
    else:  # anthropic
        return _get_anthropic_analysis_prompt_with_filename(text, filename)


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
    # Generate random examples
    examples = generate_category_examples(allowed_categories, num_examples=5)

    return f"""Categorize this document based on the rules and allowed \
categories.

CRITICAL INSTRUCTIONS:
- You MUST use ONLY categories from the "Allowed Categories" list below
- DO NOT create new categories or use variations
- If no category fits, use "reviewcategory"
- Categories must be joined with '_' (underscore)
- Maximum 4 categories per document

{category_rules}

{allowed_categories}

{examples}

Document text:
{text[:4000]}

Respond with ONLY the category string (e.g., "banking_home_sandiego" or \
"medical" or "reviewcategory"), nothing else."""


def _get_anthropic_categorization_prompt(
    text: str,
    category_rules: str,
    allowed_categories: str
) -> str:
    """Detailed categorization prompt for Anthropic with validation."""
    # Generate random examples
    examples = generate_category_examples(allowed_categories, num_examples=5)

    return f"""Categorize this document based on the rules and allowed \
categories provided.

IMPORTANT:
- Use ONLY categories from the "Allowed Categories" list below
- Categories must be joined with '_' (underscore)
- If no category fits, use "reviewcategory"
- Maximum 4 categories per document

{category_rules}

{allowed_categories}

{examples}

Document text:
{text[:4000]}

Respond with ONLY a single line containing the category string.
Do NOT include explanations."""


def _get_ollama_analysis_prompt_with_filename(
    text: str,
    filename: str
) -> str:
    """Fast, conservative analysis prompt for Ollama using filename."""
    return f"""Analyze this document and extract the following information in \
JSON format.

IMPORTANT: Use the filename as a starting point to help identify the \
originator, date, summary, and categories.

Filename: {filename}

1. "originator": The PRIMARY company or organization name.
   - First, examine the filename for company/organization clues
   - Then verify/refine based on document content
   - Use SHORT form (e.g., "Chase", "Vanguard", "PG&E")
   - For doctors: "Dr. LastName" (e.g., "Dr. Smith")
   - If unclear, use "Unknown"

2. "date": The main document date.
   - First, look for dates in the filename (YYYYMMDD or similar patterns)
   - Then verify/refine with dates from document content
   - Look for: document date, due date, statement date
   - Use the format shown in the document
   - If a date range is given, use the last date in the range
   - Do not allow dates after 2025
   - If no clear date, use "Unknown"

3. "summary": Brief description of document type.
   - Consider filename for document type hints
   - Keep under 60 characters
   - Examples: "Credit card statement", "Medical bill", "Utility bill"
   - If unclear, use "Unknown"

Document text:
{text[:4000]}

Respond with ONLY a JSON object:
{{"originator": "...", "date": "...", "summary": "..."}}"""


def _get_anthropic_analysis_prompt_with_filename(
    text: str,
    filename: str
) -> str:
    """Detailed, precise analysis prompt for Anthropic using filename."""
    return f"""Analyze this document and extract the following information in \
JSON format.

IMPORTANT: Use the filename as a helpful starting point to identify the \
originator, date, and document type. However, always verify and prioritize \
information from the actual document content.

Filename: {filename}

1. "originator": The PRIMARY company, organization, or entity.
   - First examine the filename for organization/company name clues
   - Then verify and refine based on document content (letterheads, logos)
   - Extract the main company name (e.g., "Chase", "Vanguard", "SDG&E")
   - For medical: Include full doctor name (e.g., "Dr. John Smith")
   - Do NOT include addresses or full legal entity names

2. "date": The primary document date in YYYY-MM-DD format.
   CRITICAL REQUIREMENTS:
   - First check the filename for date patterns (YYYYMMDD, YYYY-MM-DD, etc.)
   - Then verify with the main date IN THE DOCUMENT TEXT
   - Use the document date if it differs from filename date
   - Convert to YYYY-MM-DD format (e.g., "June 14, 2023" → "2023-06-14")
   - If a date range is given use the last date in the range
   - Verify year is between 2015-2025
   - Do NOT make up dates or use future dates
   - If no clear date exists, use "Unknown"

3. "summary": Concise, specific description (MAXIMUM 60 characters).
   - Consider filename for document type hints
   - Be specific about document type and purpose
   - Examples: "Credit card statement", "Medical bill for office visit"
   - Do NOT include names, amounts, or excessive detail
   - Focus on WHAT the document is

Document text:
{text[:4000]}

Respond with ONLY a JSON object in this exact format:
{{"originator": "...", "date": "...", "summary": "..."}}"""
