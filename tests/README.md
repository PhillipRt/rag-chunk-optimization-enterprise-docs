# RAG Evaluation Framework Tests

This directory contains test files for the RAG evaluation framework. The tests are organized into subdirectories that mirror the project structure.

## Directory Structure

```
tests/
├── conftest.py            # Shared pytest fixtures
├── fixtures/              # Test data fixtures
│   └── sample_documents.py # Sample documents for testing
├── retrievers/            # Tests for retriever implementations
│   ├── test_base_retriever.py
│   ├── test_embedding_retriever.py
│   ├── test_graph_retriever.py
│   ├── test_hybrid_retriever.py
│   └── test_multihop_retriever.py
└── core/                  # Tests for core components
    └── test_evaluation.py
```

## Running Tests

You can use the provided `run_tests.py` script at the project root to run the tests:

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Run a specific test file
python run_tests.py --test master-thesis-rag/tests/retrievers/test_embedding_retriever.py

# Generate coverage report
python run_tests.py --coverage
```

### Using uv

The test runner script uses `uv` to manage dependencies. Make sure you have `uv` installed and activated:

```bash
# For Windows
.\setup-uv.ps1

# For Linux/Mac
./setup-uv.sh
```

## Test Fixtures

The main test fixtures are defined in `conftest.py`:

- `sample_documents`: A list of sample documents for testing retrievers
- `sample_queries`: A list of sample queries for testing retrieval
- `mock_embedding_api`: Mocks the remote embedding API calls
- `mock_llm`: Mocks LLM calls for query generation and answer generation

## Adding New Tests

When adding new tests:

1. Create a new test file in the appropriate subdirectory
2. Import the necessary modules and fixtures
3. Write test functions that start with `test_`
4. Use appropriate mocks to avoid actual API calls

Example:

```python
"""
Tests for a new component.
"""
import pytest
from unittest.mock import patch, MagicMock

from module.to.test import ComponentToTest

def test_component_initialization():
    """Test that the component can be initialized."""
    component = ComponentToTest()
    assert component is not None
```

## Mocking Strategy

The tests use `unittest.mock` to avoid making actual external API calls:

- API calls for embeddings are mocked in the `mock_embedding_api` fixture
- LLM calls are mocked in the `mock_llm` fixture
- File system operations are mocked using `patch` decorators

This ensures that the tests can run without requiring external services and makes them more deterministic. 