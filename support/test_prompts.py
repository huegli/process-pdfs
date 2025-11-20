#!/usr/bin/env python3
"""
Unit tests for prompts module

These tests verify that prompt generation functions create properly
formatted prompts for both Ollama and Anthropic models.
"""

import pytest
from process_pdfs.prompts import (
    get_analysis_prompt,
    get_categorization_prompt
)


class TestGetAnalysisPrompt:
    """Tests for get_analysis_prompt function"""

    def test_ollama_prompt_generation(self):
        """Test Ollama analysis prompt generation"""
        text = "This is a sample document text for testing."
        prompt = get_analysis_prompt(text, model_type='ollama')

        # Check that prompt contains expected components
        assert "JSON format" in prompt
        assert "originator" in prompt
        assert "date" in prompt
        assert "summary" in prompt
        assert text in prompt
        assert "SHORT form" in prompt
        assert "under 60 characters" in prompt

    def test_anthropic_prompt_generation(self):
        """Test Anthropic analysis prompt generation"""
        text = "This is a sample document text for testing."
        prompt = get_analysis_prompt(text, model_type='anthropic')

        # Check that prompt contains expected components
        assert "JSON format" in prompt
        assert "originator" in prompt
        assert "date" in prompt
        assert "summary" in prompt
        assert text in prompt
        assert "YYYY-MM-DD" in prompt
        assert "MAXIMUM 60 characters" in prompt
        assert "CRITICAL REQUIREMENTS" in prompt

    def test_prompt_includes_text(self):
        """Test that prompt includes the document text"""
        text = "Unique document identifier XYZ123"
        prompt = get_analysis_prompt(text, model_type='ollama')

        assert "XYZ123" in prompt

    def test_prompt_truncates_long_text(self):
        """Test that very long text is truncated to 4000 chars"""
        # Create text longer than 4000 characters
        text = "A" * 5000
        prompt = get_analysis_prompt(text, model_type='ollama')

        # The text in the prompt should be truncated to [:4000]
        # The prompt contains additional text, so check that full 5000 char text isn't there
        assert text not in prompt  # Full 5000 char string should not be in prompt
        assert text[:4000] in prompt  # But first 4000 chars should be

    def test_ollama_vs_anthropic_differences(self):
        """Test that Ollama and Anthropic prompts have distinct characteristics"""
        text = "Sample document"

        ollama_prompt = get_analysis_prompt(text, model_type='ollama')
        anthropic_prompt = get_analysis_prompt(text, model_type='anthropic')

        # Ollama prompt is simpler/shorter
        assert "CRITICAL REQUIREMENTS" not in ollama_prompt
        assert "CRITICAL REQUIREMENTS" in anthropic_prompt

        # Anthropic has more detailed date instructions
        assert "YYYY-MM-DD format" in anthropic_prompt
        assert "between 2015-2025" in anthropic_prompt

    def test_json_format_instructions(self):
        """Test that both prompts include JSON format instructions"""
        text = "Sample"

        ollama = get_analysis_prompt(text, model_type='ollama')
        anthropic = get_analysis_prompt(text, model_type='anthropic')

        # Both should show JSON format
        assert '{"originator"' in ollama or '{{"originator"' in ollama
        assert '{"originator"' in anthropic or '{{"originator"' in anthropic

    def test_default_model_type(self):
        """Test default model type is ollama"""
        text = "Sample"
        default_prompt = get_analysis_prompt(text)
        ollama_prompt = get_analysis_prompt(text, model_type='ollama')

        assert default_prompt == ollama_prompt


class TestGetCategorizationPrompt:
    """Tests for get_categorization_prompt function"""

    def test_ollama_categorization_prompt(self):
        """Test Ollama categorization prompt generation"""
        text = "Sample document"
        category_rules = "# Category Rules\n1. Use banking for bank statements"
        allowed_categories = "# Allowed Categories\n- banking\n- medical"

        prompt = get_categorization_prompt(
            text, category_rules, allowed_categories, model_type='ollama'
        )

        # Check components are included
        assert text in prompt
        assert category_rules in prompt
        assert allowed_categories in prompt
        assert "Categorize this document" in prompt

    def test_anthropic_categorization_prompt(self):
        """Test Anthropic categorization prompt generation"""
        text = "Sample document"
        category_rules = "# Category Rules\n1. Use banking for bank statements"
        allowed_categories = "# Allowed Categories\n- banking\n- medical"

        prompt = get_categorization_prompt(
            text, category_rules, allowed_categories, model_type='anthropic'
        )

        # Check components are included
        assert text in prompt
        assert category_rules in prompt
        assert allowed_categories in prompt
        assert "Categorize this document" in prompt
        assert "Do NOT include explanations" in prompt

    def test_includes_category_rules(self):
        """Test that category rules are included in prompt"""
        text = "Sample"
        category_rules = "UNIQUE_RULE_IDENTIFIER_12345"
        allowed_categories = "Categories list"

        prompt = get_categorization_prompt(
            text, category_rules, allowed_categories, model_type='ollama'
        )

        assert "UNIQUE_RULE_IDENTIFIER_12345" in prompt

    def test_includes_allowed_categories(self):
        """Test that allowed categories are included in prompt"""
        text = "Sample"
        category_rules = "Rules"
        allowed_categories = "UNIQUE_CATEGORY_LIST_67890"

        prompt = get_categorization_prompt(
            text, category_rules, allowed_categories, model_type='ollama'
        )

        assert "UNIQUE_CATEGORY_LIST_67890" in prompt

    def test_text_truncation(self):
        """Test that long document text is truncated to 4000 chars"""
        text = "B" * 5000
        category_rules = "Rules"
        allowed_categories = "Categories"

        prompt = get_categorization_prompt(
            text, category_rules, allowed_categories, model_type='ollama'
        )

        # Text should be truncated
        b_count = prompt.count("B")
        assert b_count <= 4000

    def test_ollama_vs_anthropic_categorization(self):
        """Test differences between Ollama and Anthropic categorization prompts"""
        text = "Sample"
        rules = "Rules"
        categories = "Categories"

        ollama = get_categorization_prompt(text, rules, categories, model_type='ollama')
        anthropic = get_categorization_prompt(text, rules, categories, model_type='anthropic')

        # Anthropic has additional instructions
        assert "Do NOT include explanations" in anthropic
        # Ollama is more concise
        assert len(ollama) <= len(anthropic)

    def test_response_format_instructions(self):
        """Test that prompts include clear response format instructions"""
        text = "Sample"
        rules = "Rules"
        categories = "Categories"

        ollama = get_categorization_prompt(text, rules, categories, model_type='ollama')
        anthropic = get_categorization_prompt(text, rules, categories, model_type='anthropic')

        # Both should specify single line output
        assert "one line" in ollama or "single line" in anthropic

    def test_default_model_type_categorization(self):
        """Test default model type is ollama for categorization"""
        text = "Sample"
        rules = "Rules"
        categories = "Categories"

        default = get_categorization_prompt(text, rules, categories)
        ollama = get_categorization_prompt(text, rules, categories, model_type='ollama')

        assert default == ollama


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
