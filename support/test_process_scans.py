#!/usr/bin/env python3
"""
Unit tests for process_pdfs package

These tests use mocked LLM calls with test data from test_data.json
to verify the package's functionality without making actual API calls.
"""

import pytest
import json
import csv
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import tempfile
import shutil

from process_pdfs import (
    extract_text_from_pdf,
    analyze_document,
    categorize,
    parse_date_to_yyyymmdd,
    create_suggested_filename,
    process_pdfs
)


# Load test data fixture
@pytest.fixture
def test_data():
    """Load test data from test_data.json"""
    test_data_file = Path(__file__).parent.parent / "test_data.json"
    with open(test_data_file, 'r', encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    return Mock()


@pytest.fixture
def category_rules():
    """Load category_rules.md content"""
    rules_file = (
        Path(__file__).parent.parent / "src" / "process_pdfs" /
        "category_rules.md"
    )
    with open(rules_file, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def allowed_categories():
    """Load allowed_categories.md content"""
    categories_file = (
        Path(__file__).parent.parent / "src" / "process_pdfs" /
        "allowed_categories.md"
    )
    with open(categories_file, 'r', encoding='utf-8') as f:
        return f.read()


@pytest.fixture
def temp_incoming_dir(test_data):
    """Create a temporary directory with dummy test PDFs"""
    temp_dir = tempfile.mkdtemp()
    incoming_dir = Path(temp_dir) / "test-incoming"
    incoming_dir.mkdir()

    # Create dummy PDF files for each entry in test_data
    # We don't need real PDFs since we mock the extraction and analysis
    for pdf_name in test_data.keys():
        pdf_path = incoming_dir / pdf_name
        # Create a minimal valid PDF file
        pdf_path.write_bytes(
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj "
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj "
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000056 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )

    yield incoming_dir

    # Cleanup
    shutil.rmtree(temp_dir)


class TestParseDateToYYYYMMDD:
    """Tests for parse_date_to_yyyymmdd function"""

    def test_parse_iso_format(self):
        assert parse_date_to_yyyymmdd("2025-08-21") == "20250821"

    def test_parse_slash_format(self):
        assert parse_date_to_yyyymmdd("08/21/2025") == "20250821"

    def test_parse_month_name(self):
        assert parse_date_to_yyyymmdd("August 21, 2025") == "20250821"

    def test_parse_abbreviated_month(self):
        assert parse_date_to_yyyymmdd("Aug 21, 2025") == "20250821"

    def test_parse_unknown(self):
        assert parse_date_to_yyyymmdd("Unknown") == "REVIEWDATE"

    def test_parse_invalid(self):
        assert parse_date_to_yyyymmdd("not a date") == "REVIEWDATE"

    def test_parse_empty(self):
        assert parse_date_to_yyyymmdd("") == "REVIEWDATE"


class TestCreateSuggestedFilename:
    """Tests for create_suggested_filename function"""

    def test_basic_filename(self):
        file_time = datetime(2025, 9, 7, 9, 52, 21)
        result = create_suggested_filename(
            "Comerica Bank",
            "2025-08-21",
            "Statement of Account",
            file_time,
            "banking-home"
        )
        assert result == "20250821T095221--comerica-bank-statement-of-account__banking-home"

    def test_lowercase_conversion(self):
        file_time = datetime(2025, 9, 7, 9, 52, 21)
        result = create_suggested_filename(
            "TEST COMPANY",
            "2025-08-21",
            "TEST SUMMARY",
            file_time,
            "banking"
        )
        assert result == "20250821T095221--test-company-test-summary__banking"

    def test_special_chars_removed(self):
        file_time = datetime(2025, 9, 7, 9, 52, 21)
        result = create_suggested_filename(
            "Company & Co.",
            "2025-08-21",
            "Test! Summary?",
            file_time,
            "banking"
        )
        assert result == "20250821T095221--company-co-test-summary__banking"

    def test_spaces_to_dashes(self):
        file_time = datetime(2025, 9, 7, 9, 52, 21)
        result = create_suggested_filename(
            "Multiple Word Company",
            "2025-08-21",
            "Long test summary here",
            file_time,
            "banking"
        )
        assert "multiple-word-company" in result
        assert "long-test-summary-here" in result

    def test_unknown_date(self):
        file_time = datetime(2025, 9, 7, 9, 52, 21)
        result = create_suggested_filename(
            "Test Company",
            "Unknown",
            "Test summary",
            file_time,
            "reviewcategory"
        )
        assert result.startswith("REVIEWDATET")


class TestAnalyzeDocumentMocked:
    """Tests for analyze_document function with mocked LLM"""

    def test_analyze_with_valid_response(self, mock_anthropic_client):
        # Mock the LLM response
        mock_message = Mock()
        mock_message.content = [
            Mock(text='{"originator": "Test Company", "date": "2025-08-21", '
                      '"summary": "Test document"}')
        ]
        mock_anthropic_client.messages.create.return_value = mock_message

        text = "This is a test document with enough text to analyze. " * 10
        result = analyze_document(
            mock_anthropic_client, text, "test.pdf", use_ollama=False
        )

        assert result is not None
        assert result["originator"] == "Test Company"
        assert result["date"] == "2025-08-21"
        assert result["summary"] == "Test document"

    def test_analyze_with_insufficient_text(self, mock_anthropic_client):
        text = "Too short"
        result = analyze_document(mock_anthropic_client, text, "test.pdf")
        assert result is None

    def test_analyze_with_markdown_wrapped_json(self, mock_anthropic_client):
        # Mock response wrapped in markdown code block
        mock_message = Mock()
        mock_message.content = [
            Mock(text='```json\n{"originator": "Test", "date": "2025-08-21", '
                      '"summary": "Test"}\n```')
        ]
        mock_anthropic_client.messages.create.return_value = mock_message

        text = "This is a test document with enough text to analyze. " * 10
        result = analyze_document(
            mock_anthropic_client, text, "test.pdf", use_ollama=False
        )

        assert result is not None
        assert result["originator"] == "Test"


class TestCategorize:
    """Tests for categorize function"""

    def test_categorize_with_valid_response(
        self, mock_anthropic_client, category_rules, allowed_categories
    ):
        # Mock the LLM response
        mock_message = Mock()
        mock_message.content = [Mock(text='banking_home_sandiego')]
        mock_anthropic_client.messages.create.return_value = mock_message

        text = "This is a test document with enough text to categorize. " * 10
        result = categorize(
            mock_anthropic_client, text, "test.pdf",
            category_rules, allowed_categories, use_ollama=False
        )

        assert result == "banking_home_sandiego"

    def test_categorize_with_extra_category(
        self, mock_anthropic_client, category_rules, allowed_categories
    ):
        # Mock the LLM response
        mock_message = Mock()
        mock_message.content = [Mock(text='banking_home')]
        mock_anthropic_client.messages.create.return_value = mock_message

        text = "This is a test document with enough text to categorize. " * 10
        result = categorize(
            mock_anthropic_client, text, "test.pdf",
            category_rules, allowed_categories, use_ollama=False,
            extra_category="testcat"
        )

        # Should be sorted alphabetically
        assert result == "banking_home_testcat"

    def test_categorize_with_insufficient_text(
        self, mock_anthropic_client, category_rules, allowed_categories
    ):
        text = "Too short"
        result = categorize(
            mock_anthropic_client, text, "test.pdf",
            category_rules, allowed_categories
        )
        assert result == "reviewcategory"

    def test_categorize_removes_markdown(
        self, mock_anthropic_client, category_rules, allowed_categories
    ):
        # Mock response with backticks
        mock_message = Mock()
        mock_message.content = [Mock(text='`banking_home`')]
        mock_anthropic_client.messages.create.return_value = mock_message

        text = "This is a test document with enough text to categorize. " * 10
        result = categorize(
            mock_anthropic_client, text, "test.pdf",
            category_rules, allowed_categories, use_ollama=False
        )

        assert result == "banking_home"

    def test_categorize_takes_first_line_only(
        self, mock_anthropic_client, category_rules, allowed_categories
    ):
        # Mock response with multiple lines
        mock_message = Mock()
        mock_message.content = [
            Mock(text='banking_home\nSome extra text\nMore text')
        ]
        mock_anthropic_client.messages.create.return_value = mock_message

        text = "This is a test document with enough text to categorize. " * 10
        result = categorize(
            mock_anthropic_client, text, "test.pdf",
            category_rules, allowed_categories, use_ollama=False
        )

        assert result == "banking_home"


class TestExtractTextFromPDF:
    """Tests for extract_text_from_pdf function"""

    def test_extract_from_real_pdf(self):
        """Test extraction from a real PDF file"""
        pdf_path = Path(__file__).parent.parent / "Sept07-Nov09-Incoming" / "20250907095221.pdf"
        if pdf_path.exists():
            text = extract_text_from_pdf(pdf_path, max_pages=1)
            assert len(text) > 0
            assert "Page 1" in text or len(text) > 50

    def test_extract_from_nonexistent_pdf(self):
        """Test extraction from non-existent file"""
        pdf_path = Path("/tmp/nonexistent.pdf")
        text = extract_text_from_pdf(pdf_path)
        assert text == ""


class TestProcessPDFsIntegration:
    """Integration tests for process_pdfs function"""

    # Note: This test is complex due to mocking challenges.
    # Use TestProcessPDFsWithTestData instead.
    # def test_process_pdfs_with_mocked_llm(
    #     self, temp_incoming_dir, test_data,
    #     mock_anthropic_client, categories_content
    # ):
    #     """Test the full process_pdfs workflow with mocked LLM calls"""

    #     # This test is commented out due to complexities with mocking file I/O
    #     # The test_process_single_pdf_with_test_data test provides equivalent coverage
    #     pass


class TestProcessPDFsWithTestData:
    """Tests using the test_data.json fixture to simulate LLM responses"""

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_process_single_pdf_with_test_data(
        self, mock_categorize, mock_analyze, mock_extract_text,
        temp_incoming_dir, test_data
    ):
        """Test processing a single PDF with predefined test data"""

        # Mock extract_text_from_pdf to return dummy text
        mock_extract_text.return_value = "This is dummy PDF text for testing purposes."

        # Set up mock returns based on test_data
        def analyze_side_effect(
            client, text, filename, use_ollama=True, model_type='ollama',
            ollama_model='qwen2.5:7b', anthropic_model='claude-sonnet-4-5-20250929'
        ):
            if filename in test_data:
                data = test_data[filename]
                return {
                    "originator": data["originator"],
                    "date": data["date"],
                    "summary": data["summary"]
                }
            return None

        def categorize_side_effect(
            client, text, filename, category_rules, allowed_categories,
            use_ollama=True, model_type='ollama', extra_category=None,
            ollama_model='qwen2.5:7b',
            anthropic_model='claude-sonnet-4-5-20250929'
        ):
            if filename in test_data:
                category = test_data[filename]["category"]
                if extra_category:
                    categories = category.split('-')
                    categories.append(extra_category)
                    category = "-".join(sorted(categories))
                return category
            return "reviewcategory"

        mock_analyze.side_effect = analyze_side_effect
        mock_categorize.side_effect = categorize_side_effect

        # Create output CSV path
        output_csv = temp_incoming_dir.parent / "test_output.csv"

        # Mock the Anthropic initialization
        with patch('process_pdfs.cli.Anthropic'), \
             patch('process_pdfs.cli.os.getenv', return_value='fake-api-key'):

            # Process the PDFs
            process_pdfs(temp_incoming_dir, output_csv)

        # Verify the CSV was created
        assert output_csv.exists()

        # Verify CSV contents match test data
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            for row in rows:
                filename = row['filename']
                if filename in test_data:
                    expected = test_data[filename]
                    assert row['originator'] == expected['originator']
                    assert row['date'] == expected['date']
                    assert row['summary'] == expected['summary']
                    assert row['category'] == expected['category']

    # Note: extra_category feature not implemented in current version
    # @patch('process_scans.analyze_document')
    # @patch('process_scans.categorize')
    # def test_process_with_extra_category(
    #     self, mock_categorize, mock_analyze, temp_incoming_dir, test_data
    # ):
    #     """Test processing with an extra category"""

    #     # Set up mock returns
    #     def analyze_side_effect(client, text, filename):
    #         if filename in test_data:
    #             data = test_data[filename]
    #             return {
    #                 "originator": data["originator"],
    #                 "date": data["date"],
    #                 "summary": data["summary"]
    #             }
    #         return None

    #     def categorize_side_effect(client, text, filename, categories_content):
    #         if filename in test_data:
    #             category = test_data[filename]["category"]
    #             return category
    #         return "reviewcategory"

    #     mock_analyze.side_effect = analyze_side_effect
    #     mock_categorize.side_effect = categorize_side_effect

    #     # Create output CSV path
    #     output_csv = temp_incoming_dir.parent / "test_output.csv"

    #     # Mock the Anthropic initialization
    #     with patch('process_scans.Anthropic'), \
    #          patch('process_scans.os.getenv', return_value='fake-api-key'):

    #         # Process the PDFs with extra category
    #         process_pdfs(temp_incoming_dir, output_csv)

    #     # Verify categories
    #     with open(output_csv, 'r', encoding='utf-8') as f:
    #         reader = csv.DictReader(f)
    #         rows = list(reader)

    #         for row in rows:
    #             if row['filename'] in test_data:
    #                 assert row['category'] == test_data[row['filename']]['category']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
