# Hybrid LLM Strategy Design

## Strategy Overview

Leverage the complementary strengths of both LLM models:
- **Ollama (llama3:8b)**: Fast, free, conservative categorization, clean company names
- **Anthropic (Claude Haiku)**: Better date extraction, detailed summaries, consistent formatting

## Proposed Hybrid Approach

### Phase 1: Initial Processing (Ollama - Fast & Free)
Process all documents with Ollama first to get:
- Quick originator extraction (shorter, cleaner names preferred)
- Initial categorization (conservative, fewer false positives)
- Basic summary
- Initial date extraction

### Phase 2: Quality Assessment & Selective Reprocessing
Identify documents that need Anthropic's strengths:
1. **Date issues**: Ollama returned non-standard format or "Unknown"
2. **Uncertainty markers**: Category includes "reviewcategory"
3. **Summary quality**: Summary is too generic or "Unknown"
4. **Complex documents**: Multiple originators, unclear document type

### Phase 3: Anthropic Refinement (Selective)
For flagged documents, use Anthropic to:
- Extract properly formatted dates (YYYY-MM-DD)
- Generate more detailed, specific summaries
- Validate uncertain categorizations

### Phase 4: Merge & Validate
Combine results using best-of-both:
- **Originator**: Prefer Ollama (cleaner names)
- **Date**: Prefer Anthropic if Ollama had issues (better formatting)
- **Summary**: Prefer Anthropic if available (more detailed)
- **Category**: Use Ollama base + validate with Anthropic, limit to 2-3 tags

## Implementation Plan

### Refactored Architecture

```
1. Prompt Templates (Separate Module)
   - get_analysis_prompt(model_type, focus_areas)
   - get_categorization_prompt(model_type, strictness_level)

2. Quality Validators
   - validate_date_format(date_str) -> quality_score
   - validate_category(category_str) -> needs_review
   - validate_summary(summary_str) -> quality_score

3. Hybrid Processor
   - process_with_ollama(doc) -> initial_result
   - should_refine_with_anthropic(result) -> bool
   - refine_with_anthropic(doc, initial_result) -> refined_result
   - merge_results(ollama_result, anthropic_result) -> final_result
```

### Prompt Refactoring

Create focused, model-specific prompts:

**For Ollama (Speed & Breadth):**
- Simpler instructions
- Focus on basic extraction
- Allow "Unknown" for uncertain fields
- Conservative categorization

**For Anthropic (Precision & Detail):**
- More detailed instructions
- Focus on date normalization and validation
- Require specific evidence from document
- Detailed summary generation

### Cost-Benefit Analysis

Assuming 105 documents:
- **Pure Ollama**: 0 API calls, fast, 98% need review
- **Pure Anthropic**: 210 API calls (analyze + categorize), expensive, higher quality
- **Hybrid**: ~30-50 API calls (30-50% need refinement), balanced cost/quality

Estimated API cost reduction: **75-85%** vs pure Anthropic
Quality improvement: **Significant** vs pure Ollama

## Quality Metrics to Track

1. **Date quality**: % in YYYY-MM-DD format
2. **Category confidence**: % without "reviewcategory"
3. **Summary specificity**: Average length, keyword diversity
4. **Originator clarity**: % with clean company names
5. **API efficiency**: Anthropic calls / total documents

## Configuration Options

```python
--hybrid               # Enable hybrid mode
--hybrid-threshold     # Quality threshold for Anthropic refinement (0.0-1.0)
--max-anthropic-calls  # Budget cap for API calls
```

## Expected Outcomes

1. **Cost**: 75-85% reduction in API costs vs pure Anthropic
2. **Speed**: 2-3x faster than pure Anthropic
3. **Quality**: Near Anthropic quality with Ollama efficiency
4. **Confidence**: Clear flagging of uncertain results for manual review
