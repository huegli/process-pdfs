# Test Suite for process_scans.py

## Overview

This test suite provides comprehensive testing for `process_scans.py` without making actual LLM API calls. Tests use mocked LLM responses and test data generated from actual PDFs.

## Files

- **test_process_scans.py**: Main test file with 22 tests covering all major functionality
- **test_data.json**: Fixture containing dummy LLM responses for 27 PDFs from Sept07-Nov09-Incoming/
- **requirements.txt**: Updated to include pytest>=7.4.0

## Running Tests

```bash
# Run all tests
uv run pytest test_process_scans.py -v

# Run specific test class
uv run pytest test_process_scans.py::TestParseDateToYYYYMMDD -v

# Run with coverage (if pytest-cov installed)
uv run pytest test_process_scans.py --cov=process_scans -v
```

## Test Coverage

### Unit Tests (22 tests total)

1. **TestParseDateToYYYYMMDD** (7 tests)
   - Tests date parsing in various formats (ISO, slash, month names)
   - Tests handling of unknown/invalid dates

2. **TestCreateSuggestedFilename** (5 tests)
   - Tests filename generation with proper formatting
   - Tests lowercase conversion, special character removal
   - Tests space-to-dash conversion
   - Tests unknown date handling

3. **TestAnalyzeDocumentMocked** (3 tests)
   - Tests analyze_document with mocked LLM responses
   - Tests handling of insufficient text
   - Tests parsing of markdown-wrapped JSON responses

4. **TestCategorize** (4 tests)
   - Tests categorize function with mocked LLM responses
   - Tests handling of insufficient text
   - Tests markdown removal and multi-line response handling

5. **TestExtractTextFromPDF** (2 tests)
   - Tests PDF text extraction from real files
   - Tests handling of non-existent files

6. **TestProcessPDFsWithTestData** (1 integration test)
   - Tests full workflow with mocked analyze_document and categorize
   - Uses test_data.json for realistic dummy responses
   - Verifies CSV output matches expected data

## Test Data

The `test_data.json` file contains pre-generated responses for all 27 PDFs in the Sept07-Nov09-Incoming/ directory. Each entry includes:

```json
{
  "filename.pdf": {
    "originator": "Company Name",
    "date": "2025-08-21",
    "summary": "Brief description",
    "category": "category-tags"
  }
}
```

## Key Testing Strategies

1. **Mocking LLM Calls**: All tests that would normally call the Anthropic API use mocks to return predefined responses
2. **Test Data Fixtures**: Real PDF filenames mapped to expected LLM responses
3. **Unit Testing**: Each function tested independently with various inputs
4. **Integration Testing**: End-to-end workflow tested with mocked dependencies

## Benefits

- **No API Costs**: Tests run without making actual LLM API calls
- **Fast Execution**: All 22 tests complete in < 0.5 seconds
- **Deterministic**: Same inputs always produce same outputs
- **No API Key Required**: Tests work without ANTHROPIC_API_KEY set
- **Comprehensive Coverage**: Tests cover edge cases, error handling, and normal flows

## Future Enhancements

Tests currently use the base version of process_scans.py without the --category flag. When that feature is added back, uncomment the relevant test cases in test_process_scans.py.
