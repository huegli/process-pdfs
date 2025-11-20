#!/usr/bin/env python3
"""
Integration tests for process_pdfs

These tests verify end-to-end workflows including full document
processing with mocked LLM calls.
"""

import pytest
import csv
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

from process_pdfs import process_pdfs


class TestProcessPDFsIntegration:
    """Integration tests for the full process_pdfs workflow"""

    @pytest.fixture
    def temp_pdf_dir(self):
        """Create temporary directory with test PDF files"""
        temp_dir = tempfile.mkdtemp()
        pdf_dir = Path(temp_dir) / "test-pdfs"
        pdf_dir.mkdir()

        # Create minimal valid PDF files
        pdf_content = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj "
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj "
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj "
            b"xref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000056 00000 n\n0000000115 00000 n\n"
            b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n190\n%%EOF"
        )

        # Create test PDFs with timestamp names
        test_files = [
            "20250801120000.pdf",
            "20250801130000.pdf",
            "20250801140000.pdf"
        ]

        for filename in test_files:
            (pdf_dir / filename).write_bytes(pdf_content)

        yield pdf_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.Anthropic')
    @patch('process_pdfs.cli.os.getenv')
    def test_ollama_mode_full_workflow(
        self, mock_getenv, mock_anthropic, mock_categorize,
        mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test full workflow in Ollama mode"""
        mock_getenv.return_value = 'fake-api-key'

        # Mock PDF text extraction
        mock_extract.return_value = "Sample bank statement text from Chase Bank"

        # Mock analysis results
        mock_analyze.return_value = {
            'originator': 'Chase',
            'date': '2025-08-01',
            'summary': 'Bank statement'
        }

        # Mock categorization
        mock_categorize.return_value = 'banking-home'

        # Run processing
        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(
            temp_pdf_dir,
            output_csv,
            use_ollama=True,
            use_hybrid=False
        )

        # Verify CSV was created
        assert output_csv.exists()

        # Verify CSV contents
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        # Check each row
        for row in rows:
            assert row['originator'] == 'Chase'
            assert row['date'] == '2025-08-01'
            assert row['summary'] == 'Bank statement'
            assert row['category'] == 'banking-home'
            assert row['suggested_filename'].startswith('20250801T')
            assert '__banking-home' in row['suggested_filename']

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    @patch('process_pdfs.cli.Anthropic')
    @patch('process_pdfs.cli.os.getenv')
    def test_anthropic_mode_full_workflow(
        self, mock_getenv, mock_anthropic_class, mock_categorize,
        mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test full workflow in Anthropic mode"""
        mock_getenv.return_value = 'fake-api-key'
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock PDF text extraction
        mock_extract.return_value = "Medical bill from Dr. Smith's office"

        # Mock analysis results
        mock_analyze.return_value = {
            'originator': 'Dr. Smith',
            'date': '2025-08-15',
            'summary': 'Medical consultation bill'
        }

        # Mock categorization
        mock_categorize.return_value = 'medical-insurance'

        # Run processing
        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(
            temp_pdf_dir,
            output_csv,
            use_ollama=False,
            use_hybrid=False
        )

        # Verify CSV was created
        assert output_csv.exists()

        # Verify CSV contents
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        for row in rows:
            assert row['originator'] == 'Dr. Smith'
            assert row['category'] == 'medical-insurance'

    @patch('process_pdfs.cli.process_document_hybrid')
    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.Anthropic')
    @patch('process_pdfs.cli.os.getenv')
    def test_hybrid_mode_full_workflow(
        self, mock_getenv, mock_anthropic_class, mock_extract,
        mock_hybrid, temp_pdf_dir
    ):
        """Test full workflow in hybrid mode"""
        mock_getenv.return_value = 'fake-api-key'
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Mock PDF text extraction
        mock_extract.return_value = "Utility bill from PG&E"

        # Mock hybrid processing results
        mock_hybrid.return_value = {
            'result': {
                'originator': 'PG&E',
                'date': '2025-08-10',
                'summary': 'Electricity bill',
                'category': 'utilities-home'
            },
            'used_anthropic': False,
            'source': 'ollama'
        }

        # Run processing
        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(
            temp_pdf_dir,
            output_csv,
            use_ollama=False,
            use_hybrid=True,
            quality_threshold=0.6
        )

        # Verify CSV was created
        assert output_csv.exists()

        # Verify CSV contents
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

        for row in rows:
            assert row['originator'] == 'PG&E'
            assert row['category'] == 'utilities-home'

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_empty_directory(self, mock_categorize, mock_analyze, mock_extract):
        """Test handling of directory with no PDFs"""
        temp_dir = tempfile.mkdtemp()
        empty_dir = Path(temp_dir) / "empty"
        empty_dir.mkdir()

        output_csv = Path(temp_dir) / "output.csv"

        # Should handle gracefully
        process_pdfs(empty_dir, output_csv, use_ollama=True, use_hybrid=False)

        # CSV should not be created for empty directory
        assert not output_csv.exists()

        # Cleanup
        shutil.rmtree(temp_dir)

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_pdf_with_no_text(
        self, mock_categorize, mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test handling of PDF with no extractable text"""
        # Mock no text extraction
        mock_extract.return_value = ""

        # Run processing
        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(temp_pdf_dir, output_csv, use_ollama=True, use_hybrid=False)

        # Verify CSV was created
        assert output_csv.exists()

        # Verify CSV has fallback values
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            assert row['originator'] == 'Unknown (no text)'
            assert row['date'] == 'Unknown'
            assert 'Could not extract' in row['summary']
            assert row['category'] == 'reviewcategory'

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_extra_category_parameter(
        self, mock_categorize, mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test extra_category parameter"""
        mock_extract.return_value = "Sample text"
        mock_analyze.return_value = {
            'originator': 'Test',
            'date': '2025-08-01',
            'summary': 'Test doc'
        }

        # Mock categorize to return categories without extra
        def categorize_side_effect(*args, **kwargs):
            extra = kwargs.get('extra_category')
            if extra:
                return f'banking_{extra}'
            return 'banking'

        mock_categorize.side_effect = categorize_side_effect

        # Run with extra category
        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(
            temp_pdf_dir,
            output_csv,
            use_ollama=True,
            use_hybrid=False,
            extra_category='testcat'
        )

        # Verify categorize was called with extra_category (as positional arg index 7)
        call_args = mock_categorize.call_args_list[0][0]
        # Args: client, text, filename, category_rules, allowed_categories,
        #       use_ollama, model_type, extra_category
        assert len(call_args) >= 8
        assert call_args[7] == 'testcat'  # extra_category is 8th positional arg (index 7)

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_csv_output_format(
        self, mock_categorize, mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test CSV output has correct format and headers"""
        mock_extract.return_value = "Sample text"
        mock_analyze.return_value = {
            'originator': 'Test',
            'date': '2025-08-01',
            'summary': 'Test document'
        }
        mock_categorize.return_value = 'banking'

        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(temp_pdf_dir, output_csv, use_ollama=True, use_hybrid=False)

        # Check CSV format
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        # Verify headers
        expected_headers = [
            'filename', 'originator', 'date', 'summary',
            'category', 'suggested_filename'
        ]
        assert headers == expected_headers

        # Verify all fields are populated
        for row in rows:
            for header in expected_headers:
                assert header in row
                assert row[header]  # Not empty

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_suggested_filename_format(
        self, mock_categorize, mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test suggested filename has correct format"""
        mock_extract.return_value = "Sample text"
        mock_analyze.return_value = {
            'originator': 'Chase Bank',
            'date': '2025-08-15',
            'summary': 'Credit Card Statement'
        }
        mock_categorize.return_value = 'banking-home'

        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(temp_pdf_dir, output_csv, use_ollama=True, use_hybrid=False)

        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            suggested = row['suggested_filename']

            # Format: YYYYMMDDTHHMMSS--description__category
            assert 'T' in suggested  # Time separator
            assert '--' in suggested  # Description separator
            assert '__' in suggested  # Category separator
            assert suggested.startswith('20250815T')  # Date from document
            assert 'chase-bank' in suggested.lower()
            assert 'banking' in suggested.lower()

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_unicode_handling(
        self, mock_categorize, mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test handling of unicode characters in results"""
        mock_extract.return_value = "Sample text"
        mock_analyze.return_value = {
            'originator': 'Café Françaisé',
            'date': '2025-08-01',
            'summary': 'Receipt for crème brûlée'
        }
        mock_categorize.return_value = 'food'

        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(temp_pdf_dir, output_csv, use_ollama=True, use_hybrid=False)

        # Should not crash with unicode
        assert output_csv.exists()

        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Verify unicode is preserved in CSV
        assert len(rows) == 3

    @patch('process_pdfs.cli.extract_text_from_pdf')
    @patch('process_pdfs.cli.analyze_document')
    @patch('process_pdfs.cli.categorize')
    def test_analysis_failure_handling(
        self, mock_categorize, mock_analyze, mock_extract, temp_pdf_dir
    ):
        """Test handling when analysis fails"""
        mock_extract.return_value = "Sample text"
        mock_analyze.return_value = None  # Analysis failed
        mock_categorize.return_value = 'reviewcategory'

        output_csv = temp_pdf_dir.parent / "test_output.csv"
        process_pdfs(temp_pdf_dir, output_csv, use_ollama=True, use_hybrid=False)

        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            assert 'Unknown' in row['originator'] or 'failed' in row['originator'].lower()
            assert row['category'] == 'reviewcategory'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
