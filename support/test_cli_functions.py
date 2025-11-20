#!/usr/bin/env python3
"""
Unit tests for cli.py functions

These tests cover individual functions from the CLI module including
LLM calling, document analysis, categorization, and hybrid processing.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from process_pdfs.cli import (
    call_llm,
    analyze_document,
    categorize,
    process_document_hybrid
)


class TestCallLLM:
    """Tests for call_llm function"""

    @patch('process_pdfs.cli.ollama')
    def test_ollama_api_call(self, mock_ollama):
        """Test calling Ollama API"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Test response from Ollama'}
        }

        prompt = "Test prompt"
        response = call_llm(None, prompt, use_ollama=True, ollama_model='qwen2.5:7b')

        assert response == 'Test response from Ollama'
        mock_ollama.chat.assert_called_once()
        call_args = mock_ollama.chat.call_args
        assert call_args[1]['model'] == 'qwen2.5:7b'
        assert call_args[1]['messages'][0]['content'] == prompt

    def test_anthropic_api_call(self):
        """Test calling Anthropic API"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='Test response from Anthropic')]
        mock_client.messages.create.return_value = mock_message

        prompt = "Test prompt"
        response = call_llm(
            mock_client, prompt, use_ollama=False,
            max_tokens=500, anthropic_model='claude-sonnet-4-5-20250929'
        )

        assert response == 'Test response from Anthropic'
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['model'] == 'claude-sonnet-4-5-20250929'
        assert call_args[1]['max_tokens'] == 500
        assert call_args[1]['messages'][0]['content'] == prompt

    def test_max_tokens_parameter(self):
        """Test max_tokens parameter is passed correctly"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='Response')]
        mock_client.messages.create.return_value = mock_message

        call_llm(mock_client, "Test", use_ollama=False, max_tokens=1000)

        call_args = mock_client.messages.create.call_args
        assert call_args[1]['max_tokens'] == 1000

    @patch('process_pdfs.cli.ollama')
    def test_custom_ollama_model(self, mock_ollama):
        """Test using custom Ollama model"""
        mock_ollama.chat.return_value = {
            'message': {'content': 'Response'}
        }

        call_llm(None, "Test", use_ollama=True, ollama_model='llama3:8b')

        call_args = mock_ollama.chat.call_args
        assert call_args[1]['model'] == 'llama3:8b'

    def test_custom_anthropic_model(self):
        """Test using custom Anthropic model"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='Response')]
        mock_client.messages.create.return_value = mock_message

        call_llm(
            mock_client, "Test", use_ollama=False,
            anthropic_model='claude-haiku-4-5-20251001'
        )

        call_args = mock_client.messages.create.call_args
        assert call_args[1]['model'] == 'claude-haiku-4-5-20251001'


class TestAnalyzeDocumentExpanded:
    """Expanded tests for analyze_document function"""

    def test_json_extraction_with_surrounding_text(self):
        """Test extracting JSON when surrounded by other text"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(
            text='Here is the analysis:\n{"originator": "Test", "date": "2025-01-01", '
                 '"summary": "Test doc"}\nHope this helps!'
        )]
        mock_client.messages.create.return_value = mock_message

        text = "Sample document text " * 10
        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is not None
        assert result['originator'] == 'Test'
        assert result['date'] == '2025-01-01'

    def test_json_in_code_block_without_language(self):
        """Test extracting JSON from code block without language specifier"""
        mock_client = Mock()
        mock_message = Mock()
        json_text = '{"originator": "Test", "date": "2025-01-01", '
        json_text += '"summary": "Doc"}'
        mock_message.content = [Mock(text=f'```\n{json_text}\n```')]
        mock_client.messages.create.return_value = mock_message

        text = "Sample document text " * 10
        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is not None
        assert result['originator'] == 'Test'

    def test_malformed_json_response(self):
        """Test handling of malformed JSON"""
        mock_client = Mock()
        mock_message = Mock()
        bad_json = '{"originator": "Test", "date": "2025-01-01" MISSING_BRACE'
        mock_message.content = [Mock(text=bad_json)]
        mock_client.messages.create.return_value = mock_message

        text = "Sample document text " * 10
        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is None

    def test_empty_response(self):
        """Test handling of empty response"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='')]
        mock_client.messages.create.return_value = mock_message

        text = "Sample document text " * 10
        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is None

    def test_non_json_response(self):
        """Test handling of non-JSON response"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='This is just plain text without JSON')]
        mock_client.messages.create.return_value = mock_message

        text = "Sample document text " * 10
        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is None

    def test_insufficient_text_length(self):
        """Test that very short text returns None"""
        mock_client = Mock()
        text = "Short"  # Less than 50 characters

        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is None
        # API should not be called for insufficient text
        mock_client.messages.create.assert_not_called()

    def test_exactly_50_chars_processes(self):
        """Test that exactly 50 chars of text is processed"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(
            text='{"originator": "Test", "date": "2025-01-01", "summary": "Doc"}'
        )]
        mock_client.messages.create.return_value = mock_message

        text = "X" * 50  # Exactly 50 characters

        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is not None
        mock_client.messages.create.assert_called_once()

    def test_api_exception_handling(self):
        """Test handling of API exceptions"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")

        text = "Sample document text " * 10
        result = analyze_document(mock_client, text, "test.pdf", use_ollama=False)

        assert result is None

    def test_model_type_parameter_propagation(self):
        """Test that model_type is used for prompt generation"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(
            text='{"originator": "Test", "date": "2025-01-01", "summary": "Doc"}'
        )]
        mock_client.messages.create.return_value = mock_message

        text = "Sample document text " * 10

        # Test with different model types
        with patch('process_pdfs.cli.get_analysis_prompt') as mock_prompt:
            mock_prompt.return_value = "Test prompt"

            analyze_document(
                mock_client, text, "test.pdf",
                use_ollama=False, model_type='anthropic'
            )

            mock_prompt.assert_called_once()
            assert mock_prompt.call_args[1]['model_type'] == 'anthropic'


class TestCategorizeExpanded:
    """Expanded tests for categorize function"""

    @pytest.fixture
    def category_files(self):
        """Load category rules and allowed categories"""
        base = Path(__file__).parent.parent / "src" / "process_pdfs"
        rules_file = base / "category_rules.md"
        allowed_file = base / "allowed_categories.md"

        with open(rules_file, 'r') as f:
            rules = f.read()
        with open(allowed_file, 'r') as f:
            allowed = f.read()

        return rules, allowed

    def test_duplicate_category_removal(self, category_files):
        """Test that duplicate categories are removed"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='banking-banking-home')]
        mock_client.messages.create.return_value = mock_message

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        # Should remove duplicate 'banking'
        assert result.count('banking') == 1

    def test_category_sorting(self, category_files):
        """Test that categories are sorted alphabetically"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='medical-banking-home')]
        mock_client.messages.create.return_value = mock_message

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        # Should be sorted: banking_home_medical (underscore separator in output)
        categories = result.split('_')
        assert categories == sorted(categories)

    def test_maximum_4_categories_enforcement(self, category_files):
        """Test that maximum 4 categories are enforced"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='cat1-cat2-cat3-cat4-cat5-cat6')]
        mock_client.messages.create.return_value = mock_message

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        # Should have max 4 categories
        category_count = len(result.split('_'))
        assert category_count <= 4

    def test_trailing_dash_removal(self, category_files):
        """Test that trailing dashes are removed"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='banking-home-')]
        mock_client.messages.create.return_value = mock_message

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        # Should not end with separator
        assert not result.endswith('_')
        assert not result.endswith('-')

    def test_underscore_separator_in_output(self, category_files):
        """Test that output uses underscore separator"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='banking-home')]
        mock_client.messages.create.return_value = mock_message

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        # Output should use underscore
        assert '_' in result or len(result.split('_')) == 1
        assert '-' not in result

    def test_empty_category_handling(self, category_files):
        """Test handling of empty categories after splitting"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='banking--home')]
        mock_client.messages.create.return_value = mock_message

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        # Should not have empty categories
        categories = result.split('_')
        assert all(cat for cat in categories)

    def test_extra_category_with_max_categories(self, category_files):
        """Test extra_category when already at 4 categories"""
        mock_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text='cat1-cat2-cat3-cat4')]
        mock_client.messages.create.return_value = mock_message

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False,
            extra_category='extra'
        )

        # Should still add extra category (may exceed 4 when extra is added)
        assert 'extra' in result

    def test_insufficient_text_returns_reviewcategory(self, category_files):
        """Test that insufficient text returns reviewcategory"""
        mock_client = Mock()
        rules, allowed = category_files
        text = "Short"  # Less than 50 characters

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        assert result == "reviewcategory"
        mock_client.messages.create.assert_not_called()

    def test_api_exception_returns_reviewcategory(self, category_files):
        """Test that API exceptions return reviewcategory"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")

        rules, allowed = category_files
        text = "Sample text " * 10

        result = categorize(
            mock_client, text, "test.pdf",
            rules, allowed, use_ollama=False
        )

        assert result == "reviewcategory"


class TestProcessDocumentHybrid:
    """Tests for process_document_hybrid function"""

    @pytest.fixture
    def category_files(self):
        """Load category rules and allowed categories"""
        base = Path(__file__).parent.parent / "src" / "process_pdfs"
        rules_file = base / "category_rules.md"
        allowed_file = base / "allowed_categories.md"

        with open(rules_file, 'r') as f:
            rules = f.read()
        with open(allowed_file, 'r') as f:
            allowed = f.read()

        return rules, allowed

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_high_quality_result_skips_anthropic(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test that high quality Ollama result skips Anthropic"""
        rules, allowed = category_files

        # Mock high-quality Ollama response
        mock_analyze.return_value = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Credit card statement for August'
        }
        mock_categorize.return_value = 'banking-home'

        text = "Sample document text " * 10
        result = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.6
        )

        # Should not use Anthropic
        assert result['used_anthropic'] is False
        assert result['source'] == 'ollama'
        assert result['result']['originator'] == 'Chase'

        # analyze_document should be called only once (Ollama)
        assert mock_analyze.call_count == 1

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_low_quality_triggers_anthropic(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test that low quality Ollama result triggers Anthropic"""
        rules, allowed = category_files

        # Mock low-quality Ollama response, high-quality Anthropic
        mock_analyze.side_effect = [
            {
                'originator': 'Unknown',
                'date': 'Unknown',
                'summary': 'Unknown'
            },
            {
                'originator': 'Chase Bank',
                'date': '2025-08-21',
                'summary': 'Credit card statement'
            }
        ]
        mock_categorize.side_effect = ['reviewcategory', 'banking-home']

        text = "Sample document text " * 10
        result = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.6
        )

        # Should use Anthropic
        assert result['used_anthropic'] is True
        assert result['source'] == 'hybrid'

        # analyze_document should be called twice (Ollama + Anthropic)
        assert mock_analyze.call_count == 2

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_reviewcategory_triggers_refinement(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test that reviewcategory always triggers Anthropic refinement"""
        rules, allowed = category_files

        # Good analysis but reviewcategory
        mock_analyze.side_effect = [
            {
                'originator': 'Chase',
                'date': '2025-08-21',
                'summary': 'Credit card statement'
            },
            {
                'originator': 'Chase',
                'date': '2025-08-21',
                'summary': 'Credit card statement'
            }
        ]
        mock_categorize.side_effect = ['reviewcategory', 'banking-home']

        text = "Sample document text " * 10
        result = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.6
        )

        # Should use Anthropic due to reviewcategory
        assert result['used_anthropic'] is True
        assert mock_analyze.call_count == 2

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_quality_threshold_parameter(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test that quality threshold affects refinement decision"""
        rules, allowed = category_files

        # Medium quality result
        mock_analyze.return_value = {
            'originator': 'Chase',
            'date': '08/21/2025',  # 0.7 quality
            'summary': 'Statement'  # 0.4 quality (too short)
        }
        mock_categorize.return_value = 'banking'

        text = "Sample document text " * 10

        # With high threshold (0.7), should trigger refinement
        result_high_threshold = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.7
        )

        # Reset mocks
        mock_analyze.reset_mock()
        mock_categorize.reset_mock()
        mock_analyze.return_value = {
            'originator': 'Chase',
            'date': '08/21/2025',
            'summary': 'Statement'
        }
        mock_categorize.return_value = 'banking'

        # With low threshold (0.5), should not trigger refinement
        result_low_threshold = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.5
        )

        assert result_high_threshold['used_anthropic'] is True
        assert result_low_threshold['used_anthropic'] is False

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_result_merging(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test that results are properly merged"""
        rules, allowed = category_files

        # Mock low-quality Ollama response to trigger refinement
        mock_analyze.side_effect = [
            {
                'originator': 'Chase',
                'date': 'Unknown',  # Low quality to trigger refinement
                'summary': 'Doc'  # Too short
            },
            {
                'originator': 'Chase Bank Corporation',
                'date': '2025-08-21',
                'summary': 'Credit card statement for August billing'
            }
        ]
        mock_categorize.side_effect = ['reviewcategory', 'banking-home']

        text = "Sample document text " * 10
        result = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.6
        )

        # Verify that Anthropic was used
        assert result['used_anthropic'] is True

        merged_result = result['result']

        # Should take best of both after merging
        assert merged_result['originator'] == 'Chase'  # Cleaner (Ollama)
        # Better format (from Anthropic)
        assert merged_result['date'] == '2025-08-21'
        # More detailed (from Anthropic)
        assert 'Credit card' in merged_result['summary']
        # From Anthropic (Ollama had reviewcategory)
        assert merged_result['category'] == 'banking-home'

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_insufficient_text_handling(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test handling of insufficient text"""
        rules, allowed = category_files

        # Mock analyze_document to return None (insufficient text)
        mock_analyze.return_value = None
        mock_categorize.return_value = 'reviewcategory'

        text = "Sample document text " * 10
        result = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.6
        )

        # Should have fallback values
        assert result['result']['originator'] == 'Unknown (no text)'
        assert result['result']['date'] == 'Unknown'

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_extra_category_parameter(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test that extra_category is passed through"""
        rules, allowed = category_files

        mock_analyze.return_value = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Statement'
        }
        mock_categorize.return_value = 'banking'

        text = "Sample document text " * 10
        process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            extra_category='testcat',
            quality_threshold=0.6
        )

        # Categorize should be called with extra_category
        assert any(
            call[1].get('extra_category') == 'testcat'
            for call in mock_categorize.call_args_list
        )

    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.analyze_document')
    def test_return_structure(
        self, mock_analyze, mock_categorize, category_files
    ):
        """Test that return structure is correct"""
        rules, allowed = category_files

        mock_analyze.return_value = {
            'originator': 'Chase',
            'date': '2025-08-21',
            'summary': 'Credit card statement'
        }
        mock_categorize.return_value = 'banking-home'

        text = "Sample document text " * 10
        result = process_document_hybrid(
            text, "test.pdf", rules, allowed,
            None, Mock(),
            quality_threshold=0.6
        )

        # Check required fields in return
        assert 'result' in result
        assert 'used_anthropic' in result
        assert 'source' in result

        # result should contain all document fields
        assert 'originator' in result['result']
        assert 'date' in result['result']
        assert 'summary' in result['result']
        assert 'category' in result['result']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
