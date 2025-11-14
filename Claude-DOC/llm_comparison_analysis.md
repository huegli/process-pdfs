# LLM Comparison Analysis: Ollama (llama3:8b) vs Anthropic (Claude Haiku)

## Overview

Processed 105 PDF files from the `Incoming/` directory using two different LLM backends:
- **Ollama**: Local llama3:8b model
- **Anthropic**: Claude Haiku via API

## Summary Statistics

- **Total files**: 105
- **Exact matches**: 2 (1.9%)
- **Differences**: 103 (98.1%)
- **Missing entries**: 0

## Files with Exact Matches

Only 2 files produced identical suggested filenames:
1. `20240116.pdf` - Both returned: `REVIEWDATET105159--unknown-unknown__reviewcategory`
2. `RHCC_Statement_July_2023.pdf` - Both returned: `20230731T105159--round-hill-country-club-monthly-statement__danville-home`

## Key Differences Observed

### 1. Date Extraction
- **Anthropic**: More consistent YYYY-MM-DD format
- **Ollama**: Sometimes uses varied formats (e.g., "06/14/23", "10/1/22 to 12/31/22")
- **Example**: File `20210528_STATEMENT OF SERV.CES RENDERED.pdf`
  - Ollama: `05/28/2021`
  - Anthropic: `2021-05-28`

### 2. Originator Extraction
- **Anthropic**: Often extracts full legal names
- **Ollama**: Tends toward shorter company names
- **Example**: File `20220224_OFFICE RECEIPT.pdf`
  - Ollama: `Dr. Tsai`
  - Anthropic: `CLARK S. TSAI, MD`

### 3. Category Assignment
- **Major difference**: Both models assign different categories frequently
- **Ollama patterns**:
  - More conservative with categories (2-3 categories)
  - Uses "reviewcategory" more often when uncertain
  - Less likely to add location tags
- **Anthropic patterns**:
  - More aggressive with category assignment (3-4 categories)
  - More frequently adds location tags (danville, sandiego)
  - More likely to add person names as categories
  
**Example**: File `07230517_(DocTitle).pdf` (Chase credit card)
- Ollama: `banking-creditcard`
- Anthropic: `creditcard-home-sandiego`

### 4. Summary Generation
- **Anthropic**: More detailed, specific summaries
- **Ollama**: More concise, generic summaries
- **Example**: File `20221028_٠ا,اا٠ا«ا,,,٠٠,ا,٠اااا٠٠اااا٠ا٠ا!ا٠اااااااا٠٠,ااا٠ا٠اا٠.pdf`
  - Ollama: `Medical bill`
  - Anthropic: `Medical bill for follow-up visit`

### 5. Date Accuracy Issues
- **Example**: File `20220224_OFFICE RECEIPT.pdf`
  - Ollama extracted: `2022-02-24` (matches filename timestamp)
  - Anthropic extracted: `2023-02-08` (different date from document)
  
- **Example**: File `20221028_STATEMENT.pdf`
  - Ollama: `2022-01-29`
  - Anthropic: `2022-11-29` (closer to filename date)

## Categorization Differences

### Common Patterns:
- **Property-related documents**: 
  - Ollama prefers: `home`, `rental`, `reviewcategory`
  - Anthropic prefers: `property`, `rental`, `home` with location tags

- **Medical documents**:
  - Ollama: `medical` alone or with 1-2 categories
  - Anthropic: `medical` + location + person name

- **Financial documents**:
  - Ollama: Clearer separation between `banking` and `creditcard`
  - Anthropic: More overlap and additional tags

## Quality Assessment

### Ollama (llama3:8b) Strengths:
- More conservative categorization (fewer false positives)
- Shorter, cleaner originator names
- Less aggressive with category assignment
- Faster processing (local)

### Ollama Weaknesses:
- Inconsistent date formats
- Sometimes too generic in summaries
- Uses "reviewcategory" frequently when uncertain

### Anthropic (Claude Haiku) Strengths:
- More consistent YYYY-MM-DD date formatting
- More detailed summaries
- Better at extracting full legal entity names

### Anthropic Weaknesses:
- Over-categorization (too many tags)
- Some date extraction errors
- More expensive (API costs)
- Adds location/person tags even when not relevant

## Recommendations

1. **For production use**: Consider using Anthropic for date extraction reliability, but post-process to reduce over-categorization
2. **For cost-effective processing**: Ollama works well but may need date format normalization
3. **Hybrid approach**: Use Ollama for initial processing, flag uncertain items (reviewcategory) for Anthropic re-processing
4. **Prompt improvements needed for both**:
   - Stricter category limits (max 2-3 categories)
   - More guidance on when to add location/person tags
   - Clearer date format requirements

## Files Generated

- `scan_summary_ollama.csv` - Results from Ollama processing
- `scan_summary_anthropic.csv` - Results from Anthropic processing  
- `llm_comparison.txt` - Full row-by-row comparison (detailed)
- `compare_results.py` - Python script for generating comparisons
- `llm_comparison_analysis.md` - This analysis document
