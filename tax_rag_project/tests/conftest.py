"""Test configuration for tax_rag_project tests.

This conftest provides:
- Centralized test fixtures to eliminate duplication
- Mock services (OpenAI, Qdrant, embeddings)
- Sample test data (documents, batches)
- Pytest markers (unit, integration, slow)
"""

import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

# Add the src directory to Python path so imports work
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ============================================================================
# SECTION 1: Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (no external dependencies, fast)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires real Qdrant)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# ============================================================================
# SECTION 2: Configuration Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_api_key() -> str:
    """Provide a mock OpenAI API key for testing."""
    return "test-openai-key-12345"


@pytest.fixture
def mock_qdrant_url() -> str:
    """Provide a mock Qdrant Cloud URL."""
    return "https://test-cluster.cloud.qdrant.io"


@pytest.fixture
def mock_qdrant_api_key() -> str:
    """Provide a mock Qdrant API key."""
    return "test-qdrant-key-12345"


# ============================================================================
# SECTION 3: Sample Document Fixtures
# ============================================================================

@pytest.fixture
def sample_tax_document() -> dict:
    """Provide a sample tax document for testing."""
    return {
        'title': 'IRS Publication 501 - Dependents',
        'content': 'Information about claiming dependents on your tax return. ' * 50,
        'url': 'https://www.canada.ca/en/revenue-agency/test-doc.html',
        'source': 'CRA',
        'doc_type': 'CRA_Guide',
        'scraped_at': '2024-01-15T10:30:00Z',
    }


@pytest.fixture
def sample_large_document() -> dict:
    """Provide a large document that requires chunking (>1200 words)."""
    return {
        'title': 'Comprehensive Tax Guide 2024',
        'content': 'Tax information. ' * 5000,  # Large enough to trigger chunking
        'url': 'https://www.canada.ca/en/revenue-agency/large-guide.html',
        'source': 'CRA',
        'doc_type': 'CRA_Guide',
        'scraped_at': '2024-01-15T10:30:00Z',
    }


@pytest.fixture
def sample_document_batch() -> list[dict]:
    """Provide a batch of documents for testing batch processing."""
    return [
        {
            'title': f'Tax Document {i}',
            'content': f'Content for document {i}. ' * 100,
            'url': f'https://www.canada.ca/en/revenue-agency/doc-{i}.html',
            'source': 'CRA',
            'doc_type': 'CRA_Guide',
            'scraped_at': '2024-01-15T10:30:00Z',
        }
        for i in range(5)
    ]


# ============================================================================
# SECTION 4: Embedding Fixtures
# ============================================================================

@pytest.fixture
def mock_dense_embedding() -> list[float]:
    """Provide a mock dense embedding (1536 dimensions for text-embedding-3-small)."""
    return [0.1] * 1536


@pytest.fixture
def mock_sparse_vector() -> SparseVector:
    """Provide a mock sparse vector."""
    return SparseVector(
        indices=[0, 5, 10, 15, 20],
        values=[0.8, 0.6, 0.4, 0.3, 0.2]
    )


@pytest.fixture
def mock_embedding_service(mock_openai_api_key, mock_dense_embedding):
    """Provide a mocked EmbeddingService."""
    from tax_rag_scraper.utils.embeddings import EmbeddingService

    service = EmbeddingService(api_key=mock_openai_api_key)

    # Mock the OpenAI client
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=mock_dense_embedding)]
    mock_response.usage.total_tokens = 100

    service.client.embeddings.create = AsyncMock(return_value=mock_response)

    return service


@pytest.fixture
def mock_sparse_embedding_service(mock_sparse_vector):
    """Provide a mocked SparseEmbeddingService."""
    from tax_rag_scraper.utils.embeddings import SparseEmbeddingService

    service = SparseEmbeddingService()

    # Mock the fastembed model
    service.model.embed = MagicMock(return_value=[mock_sparse_vector])

    return service


# ============================================================================
# SECTION 5: Qdrant Client Mocks
# ============================================================================

@pytest.fixture
def mock_qdrant_client():
    """Provide a fully mocked QdrantClient for unit tests."""
    mock_client = MagicMock(spec=QdrantClient)

    # Mock get_collections() for validation
    mock_collection = MagicMock()
    mock_collection.name = 'cra-collection'
    mock_client.get_collections.return_value = MagicMock(collections=[mock_collection])

    # Mock get_collection() for info
    mock_client.get_collection.return_value = MagicMock(points_count=100)

    # Mock upsert() for storage
    mock_client.upsert.return_value = None

    # Mock query_points() for search
    mock_result = MagicMock()
    mock_result.points = [
        MagicMock(id='1', score=0.95, payload={'chunk_text': 'Result 1'}),
        MagicMock(id='2', score=0.85, payload={'chunk_text': 'Result 2'}),
    ]
    mock_client.query_points.return_value = mock_result

    return mock_client


@pytest.fixture
def mock_qdrant_client_no_collection():
    """Provide a mocked QdrantClient with no collections (for validation testing)."""
    mock_client = MagicMock(spec=QdrantClient)
    mock_client.get_collections.return_value = MagicMock(collections=[])
    return mock_client


# ============================================================================
# SECTION 6: Integration Test Fixtures (Real Qdrant - Optional)
# ============================================================================

@pytest.fixture
def real_qdrant_available() -> bool:
    """Check if real Qdrant instance is available for integration tests."""
    import os
    return bool(os.getenv('QDRANT_URL') and os.getenv('QDRANT_API_KEY'))


@pytest.fixture
async def real_qdrant_client(real_qdrant_available) -> AsyncGenerator:
    """
    Provide a real QdrantClient for integration tests.

    Requires environment variables:
    - QDRANT_URL
    - QDRANT_API_KEY
    - QDRANT_TEST_COLLECTION (defaults to 'test-collection')

    Tests will be skipped if these variables are not set.
    """
    if not real_qdrant_available:
        pytest.skip("Real Qdrant not available (set QDRANT_URL and QDRANT_API_KEY)")

    import os
    from tax_rag_scraper.storage.qdrant_client import TaxDataQdrantClient

    client = TaxDataQdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
        collection_name=os.getenv('QDRANT_TEST_COLLECTION', 'test-collection'),
        source='test',
    )

    yield client

    # Cleanup: optionally delete test data
    # client.delete_collection()  # Uncomment if you want cleanup
