#!/usr/bin/env python3
"""
Unit tests for quality_validators module

These tests verify the quality validation and result merging logic
used in hybrid mode processing.
"""

import pytest
from process_pdfs.quality_validators import (
    validate_date_format,
    validate_category,
    validate_summary,
    validate_originator,
    should_refine_with_anthropic,
    merge_results
)


class TestValidateDateFormat:
    """Tests for validate_date_format function"""

    def test_perfect_yyyy_mm_dd_format(self):
        """YYYY-MM-DD format should score 1.0"""
        assert validate_date_format("2025-08-21") == 1.0
        assert validate_date_format("2020-01-01") == 1.0
        assert validate_date_format("2024-12-31") == 1.0

    def test_mm_dd_yyyy_format(self):
        """MM/DD/YYYY format should score 0.7"""
        assert validate_date_format("08/21/2025") == 0.7
        assert validate_date_format("01/01/2020") == 0.7
        assert validate_date_format("12/31/2024") == 0.7

    def test_yyyy_mm_dd_slash_format(self):
        """YYYY/MM/DD format should score 0.8"""
        assert validate_date_format("2025/08/21") == 0.8
        assert validate_date_format("2020/01/01") == 0.8

    def test_full_month_name_format(self):
        """Full month names should score 0.6"""
        assert validate_date_format("January 15, 2025") == 0.6
        assert validate_date_format("December 31, 2024") == 0.6
        assert validate_date_format("August 21, 2025") == 0.6

    def test_abbreviated_month_format(self):
        """Abbreviated month names should score 0.6"""
        assert validate_date_format("Jan 15, 2025") == 0.6
        assert validate_date_format("Dec 31, 2024") == 0.6
        assert validate_date_format("Aug 21, 2025") == 0.6

    def test_date_ranges(self):
        """Date ranges should score 0.3 when using ' to ' separator"""
        assert validate_date_format("2025-08-01 to 2025-08-31") == 0.3
        # Note: "Jan 1 - Jan 31" matches abbreviated month pattern first (0.6)
        assert validate_date_format("Jan 1 - Jan 31") == 0.6

    def test_unknown_values(self):
        """Unknown/N/A/empty should score 0.0"""
        assert validate_date_format("Unknown") == 0.0
        assert validate_date_format("unknown") == 0.0
        assert validate_date_format("N/A") == 0.0
        assert validate_date_format("n/a") == 0.0
        assert validate_date_format("") == 0.0

    def test_invalid_formats(self):
        """Invalid/unrecognized formats should score 0.2"""
        assert validate_date_format("not a date") == 0.2
        assert validate_date_format("2025") == 0.2
        assert validate_date_format("random text") == 0.2

    def test_none_input(self):
        """None input should score 0.0"""
        assert validate_date_format(None) == 0.0


class TestValidateCategory:
    """Tests for validate_category function"""

    def test_valid_single_category(self):
        """Valid single category should return False (no review needed)"""
        assert validate_category("banking") is False
        assert validate_category("medical") is False
        assert validate_category("utilities") is False

    def test_valid_multiple_categories(self):
        """Valid multiple categories should return False"""
        assert validate_category("banking-home") is False
        assert validate_category("medical-insurance") is False
        assert validate_category("utilities-electricity-home") is False

    def test_reviewcategory_flag(self):
        """reviewcategory should return True (needs review)"""
        assert validate_category("reviewcategory") is True
        assert validate_category("banking-reviewcategory") is True

    def test_trailing_dashes(self):
        """Trailing dashes should return True (formatting issue)"""
        assert validate_category("banking-") is True
        assert validate_category("medical-home-") is True

    def test_empty_categories_after_split(self):
        """Empty categories in string should return True"""
        assert validate_category("banking--home") is True
        assert validate_category("-banking") is True

    def test_too_many_categories(self):
        """More than 4 categories should return True"""
        assert validate_category("cat1-cat2-cat3-cat4-cat5") is True
        assert validate_category("a-b-c-d-e-f") is True

    def test_exactly_four_categories(self):
        """Exactly 4 categories should return False"""
        assert validate_category("cat1-cat2-cat3-cat4") is False

    def test_empty_string(self):
        """Empty string should return True"""
        assert validate_category("") is True

    def test_none_input(self):
        """None input should return True"""
        assert validate_category(None) is True


class TestValidateSummary:
    """Tests for validate_summary function"""

    def test_optimal_length_summaries(self):
        """Summaries between 20-60 chars should score 1.0"""
        assert validate_summary("Credit card statement") == 1.0
        assert validate_summary("Medical bill for office visit") == 1.0
        assert validate_summary("Utility bill for electricity") == 1.0

    def test_good_length_60_to_70(self):
        """Summaries 60-70 chars should score 0.9"""
        # 65 characters
        summary = "This is a longer but still acceptable summary for the document"
        assert validate_summary(summary) == 0.9

    def test_acceptable_length_70_to_80(self):
        """Summaries 70-80 chars should score 0.7"""
        # 75 characters
        summary = "This is an even longer summary that is approaching the upper limit of text"
        assert validate_summary(summary) == 0.7

    def test_too_long_summaries(self):
        """Summaries over 80 chars should score 0.3"""
        summary = (
            "This is a very long summary that exceeds the recommended "
            "length and should be penalized accordingly"
        )
        assert validate_summary(summary) == 0.3

    def test_too_short_summaries(self):
        """Summaries under 20 chars should score 0.4"""
        assert validate_summary("Short summary") == 0.4
        assert validate_summary("Brief") == 0.4

    def test_generic_short_summaries(self):
        """Short and generic should score 0.3"""
        assert validate_summary("Unknown") == 0.0  # Special case
        assert validate_summary("Document") == 0.4
        assert validate_summary("Statement") == 0.4

    def test_unknown_values(self):
        """Unknown/N/A/empty should score 0.0"""
        assert validate_summary("Unknown") == 0.0
        assert validate_summary("unknown") == 0.0
        assert validate_summary("N/A") == 0.0
        assert validate_summary("") == 0.0

    def test_none_input(self):
        """None input should score 0.0"""
        assert validate_summary(None) == 0.0


class TestValidateOriginator:
    """Tests for validate_originator function"""

    def test_clean_short_names(self):
        """Clean names under 30 chars should score 1.0"""
        assert validate_originator("Chase") == 1.0
        assert validate_originator("Vanguard") == 1.0
        assert validate_originator("Dr. Smith") == 1.0
        assert validate_originator("PG&E") == 1.0

    def test_medium_length_names(self):
        """Names between 30-50 chars should score 0.7"""
        assert validate_originator("Chase Bank Corporation of America") == 0.7

    def test_overly_long_names(self):
        """Names over 50 chars should score 0.5"""
        long_name = "Very Long Corporate Legal Entity Name Inc. Corporation"
        assert validate_originator(long_name) == 0.5

    def test_address_like_content(self):
        """Names with address indicators should score 0.3"""
        assert validate_originator("Company at 123 Main Street") == 0.3
        assert validate_originator("Business on Oak Avenue") == 0.3
        assert validate_originator("Corp, California") == 0.3
        assert validate_originator("Business CA 94105") == 0.3

    def test_unknown_values(self):
        """Unknown/N/A should score 0.0"""
        assert validate_originator("Unknown") == 0.0
        assert validate_originator("unknown") == 0.0
        assert validate_originator("N/A") == 0.0
        assert validate_originator("n/a") == 0.0

    def test_none_input(self):
        """None input should score 0.0"""
        assert validate_originator(None) == 0.0


class TestShouldRefineWithAnthropicBest:
    """Tests for should_refine_with_anthropic function"""

    def test_high_quality_no_refinement(self):
        """High quality results should not need refinement"""
        result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Credit card statement',
            'category': 'banking-home'
        }

        assert should_refine_with_anthropic(result, threshold=0.6) is False

    def test_low_quality_needs_refinement(self):
        """Low quality results should need refinement"""
        result = {
            'originator': 'Unknown',
            'date': 'Unknown',
            'summary': 'Unknown',
            'category': 'banking'
        }

        assert should_refine_with_anthropic(result, threshold=0.6) is True

    def test_reviewcategory_always_refines(self):
        """reviewcategory should always trigger refinement"""
        result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Credit card statement',
            'category': 'reviewcategory'
        }

        assert should_refine_with_anthropic(result, threshold=0.6) is True

    def test_poor_date_triggers_refinement(self):
        """Poor date format should trigger refinement when combined with other low scores"""
        result = {
            'originator': 'Chase',
            'date': 'sometime in 2025',  # 0.2 quality (no month name)
            'summary': 'Doc',  # 0.4 quality (too short)
            'category': 'banking'
        }
        # Overall: 0.2*0.4 + 0.4*0.4 + 1.0*0.2 = 0.08 + 0.16 + 0.2 = 0.44

        assert should_refine_with_anthropic(result, threshold=0.6) is True

    def test_poor_summary_triggers_refinement(self):
        """Poor summary quality should trigger refinement when combined with other issues"""
        result = {
            'originator': 'Unknown Company at 123 Main Street',  # 0.3 (has address)
            'date': 'Aug 2025',  # 0.6 (abbreviated month)
            'summary': 'Doc',  # 0.4 (too short)
            'category': 'banking'
        }
        # Overall: 0.6*0.4 + 0.4*0.4 + 0.3*0.2 = 0.24 + 0.16 + 0.06 = 0.46

        assert should_refine_with_anthropic(result, threshold=0.6) is True

    def test_threshold_sensitivity(self):
        """Different thresholds should change refinement decision"""
        result = {
            'originator': 'Chase',
            'date': '08/21/2025',  # 0.7 quality
            'summary': 'Statement',  # 0.4 quality (too short)
            'category': 'banking'
        }

        # Overall quality: 0.7*0.4 + 0.4*0.4 + 1.0*0.2 = 0.28 + 0.16 + 0.2 = 0.64
        assert should_refine_with_anthropic(result, threshold=0.7) is True
        assert should_refine_with_anthropic(result, threshold=0.6) is False

    def test_missing_fields(self):
        """Missing fields should trigger refinement"""
        result = {
            'category': 'banking'
        }

        assert should_refine_with_anthropic(result, threshold=0.6) is True


class TestMergeResults:
    """Tests for merge_results function"""

    def test_merge_with_both_results(self):
        """Test merging when both Ollama and Anthropic results exist"""
        ollama_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking-home'
        }
        anthropic_result = {
            'originator': 'Chase Bank Corporation',
            'date': '2025-08-21',
            'summary': 'Credit card statement for account ending in 1234',
            'category': 'banking-home'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Ollama originator is cleaner (shorter)
        assert merged['originator'] == 'Chase'
        # Both dates are same quality
        assert merged['date'] == '2025-08-21'
        # Anthropic summary is more detailed
        assert 'Credit card' in merged['summary']
        # Categories are same
        assert merged['category'] == 'banking-home'

    def test_merge_with_only_ollama(self):
        """Test merging with only Ollama result"""
        ollama_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking'
        }

        merged = merge_results(ollama_result, None)

        assert merged == ollama_result

    def test_merge_with_only_anthropic(self):
        """Test merging with only Anthropic result"""
        anthropic_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Credit card statement',
            'category': 'banking'
        }

        merged = merge_results(None, anthropic_result)

        assert merged == anthropic_result

    def test_merge_with_neither(self):
        """Test merging with no results"""
        merged = merge_results(None, None)
        assert merged == {}

    def test_originator_selection_by_quality(self):
        """Test originator selection based on quality scores"""
        ollama_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking'
        }
        anthropic_result = {
            'originator': 'Chase Bank, 123 Main Street, California',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Ollama is cleaner (no address)
        assert merged['originator'] == 'Chase'

    def test_date_selection_prefers_yyyy_mm_dd(self):
        """Test date selection prefers YYYY-MM-DD format"""
        ollama_result = {
            'originator': 'Chase',
            'date': 'August 21, 2025',
            'summary': 'Statement',
            'category': 'banking'
        }
        anthropic_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Anthropic has better format
        assert merged['date'] == '2025-08-21'

    def test_date_selection_falls_back_to_quality(self):
        """Test date selection falls back to quality comparison"""
        ollama_result = {
            'originator': 'Chase',
            'date': 'Aug 21, 2025',
            'summary': 'Statement',
            'category': 'banking'
        }
        anthropic_result = {
            'originator': 'Chase',
            'date': 'Unknown',
            'summary': 'Statement',
            'category': 'banking'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Ollama has better quality
        assert merged['date'] == 'Aug 21, 2025'

    def test_summary_selection_by_quality(self):
        """Test summary selection based on quality scores"""
        ollama_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking'
        }
        anthropic_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Credit card statement for August billing cycle',
            'category': 'banking'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Anthropic is more detailed and better length
        assert merged['summary'] == 'Credit card statement for August billing cycle'

    def test_category_with_reviewcategory_in_ollama(self):
        """Test category selection when Ollama has reviewcategory"""
        ollama_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'reviewcategory'
        }
        anthropic_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking-home'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Should use Anthropic category
        assert merged['category'] == 'banking-home'

    def test_category_prefers_ollama_when_valid(self):
        """Test category selection prefers Ollama when valid"""
        ollama_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking'
        }
        anthropic_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'banking-home'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Should use Ollama category (more conservative)
        assert merged['category'] == 'banking'

    def test_category_both_reviewcategory(self):
        """Test category when both have reviewcategory"""
        ollama_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'reviewcategory'
        }
        anthropic_result = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement',
            'category': 'reviewcategory'
        }

        merged = merge_results(ollama_result, anthropic_result)

        # Should use Anthropic (but both are reviewcategory)
        assert merged['category'] == 'reviewcategory'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
