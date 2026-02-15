"""Test suite for TaxDataQdrantClient.

Tests the Qdrant Cloud client for storing tax documentation with
hybrid vector embeddings (dense + sparse).
"""

import pytest
from unittest.mock import MagicMock, patch
from qdrant_client.models import SparseVector, PointStruct
from tax_rag_scraper.storage.qdrant_client import TaxDataQdrantClient


@pytest.mark.unit
class TestTaxDataQdrantClientInit:
    """Test initialization and validation."""

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_init_success(self, mock_qdrant_class, mock_qdrant_client) -> None:
        """Test successful initialization with existing collection."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        assert client.collection_name == 'cra-collection'
        assert client.source == 'cra'
        assert client.dense_vector_name == 'cra-dense'
        assert client.sparse_vector_name == 'cra-sparse'

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_init_collection_not_exists(
        self, mock_qdrant_class, mock_qdrant_client_no_collection
    ) -> None:
        """Test initialization fails when collection doesn't exist."""
        mock_qdrant_class.return_value = mock_qdrant_client_no_collection

        with pytest.raises(RuntimeError, match="does not exist in Qdrant Cloud"):
            TaxDataQdrantClient(
                url='https://test.cloud.qdrant.io',
                api_key='test-key',
                collection_name='nonexistent-collection',
                source='cra',
            )

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_init_custom_vector_size(self, mock_qdrant_class, mock_qdrant_client) -> None:
        """Test initialization with custom vector size."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
            vector_size=3072,  # text-embedding-3-large
        )

        assert client.vector_size == 3072

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_named_vectors_correctly_formatted(
        self, mock_qdrant_class, mock_qdrant_client
    ) -> None:
        """Test that named vectors are correctly formatted: 'cra' → 'cra-dense', 'cra-sparse'."""
        # Setup mock to accept multiple collection names
        def get_collections_side_effect():
            # Return different collections based on which client was created
            mock_collection = MagicMock()
            # Return collection that matches whatever was requested
            if hasattr(get_collections_side_effect, 'call_count'):
                get_collections_side_effect.call_count += 1
                if get_collections_side_effect.call_count == 1:
                    mock_collection.name = 'cra-collection'
                else:
                    mock_collection.name = 'dof-collection'
            else:
                get_collections_side_effect.call_count = 1
                mock_collection.name = 'cra-collection'
            return MagicMock(collections=[mock_collection])

        mock_qdrant_client.get_collections = get_collections_side_effect
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        assert client.dense_vector_name == 'cra-dense'
        assert client.sparse_vector_name == 'cra-sparse'

        # Test with different source
        # Reset mock for second client
        mock_dof_collection = MagicMock()
        mock_dof_collection.name = 'dof-collection'
        mock_qdrant_client.get_collections.return_value = MagicMock(collections=[mock_dof_collection])

        client2 = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='dof-collection',
            source='dof',
        )

        assert client2.dense_vector_name == 'dof-dense'
        assert client2.sparse_vector_name == 'dof-sparse'


@pytest.mark.unit
class TestTaxDataQdrantClientStorage:
    """Test document storage with hybrid vectors."""

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    @pytest.mark.asyncio
    async def test_store_documents_single_chunk(
        self, mock_qdrant_class, mock_qdrant_client
    ) -> None:
        """Test storing a single document chunk with dual vectors."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        chunks = [
            (
                'Tax deduction information',
                [0.1] * 1536,  # Dense embedding
                SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2]),  # Sparse
                {
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'parent_title': 'Tax Guide',
                    'parent_url': 'https://example.com',
                    'parent_source': 'CRA',
                    'parent_doc_type': 'Guide',
                    'parent_scraped_at': '2024-01-15',
                }
            )
        ]

        await client.store_documents(chunks)

        # Verify upsert was called once
        assert mock_qdrant_client.upsert.call_count == 1

        # Verify point structure
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs['points']
        assert len(points) == 1

        point = points[0]
        assert isinstance(point, PointStruct)
        assert 'cra-dense' in point.vector
        assert 'cra-sparse' in point.vector
        assert point.payload['chunk_text'] == 'Tax deduction information'
        assert point.payload['title'] == 'Tax Guide'

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    @pytest.mark.asyncio
    async def test_store_documents_multiple_chunks(
        self, mock_qdrant_class, mock_qdrant_client
    ) -> None:
        """Test storing multiple document chunks."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        chunks = [
            (
                f'Chunk {i}',
                [float(i)] * 1536,
                SparseVector(indices=[i], values=[0.5]),
                {
                    'chunk_index': i,
                    'total_chunks': 3,
                    'parent_title': 'Doc',
                    'parent_url': 'http://example.com',
                    'parent_source': 'CRA',
                    'parent_doc_type': 'Guide',
                    'parent_scraped_at': '2024-01-15',
                },
            )
            for i in range(3)
        ]

        await client.store_documents(chunks)

        # Verify all chunks stored
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs['points']
        assert len(points) == 3

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    @pytest.mark.asyncio
    async def test_store_documents_metadata_preservation(
        self, mock_qdrant_class, mock_qdrant_client
    ) -> None:
        """Test that all payload fields are preserved."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        metadata = {
            'chunk_index': 2,
            'total_chunks': 5,
            'parent_title': 'Complete Tax Guide',
            'parent_url': 'https://canada.ca/tax-guide',
            'parent_source': 'CRA',
            'parent_doc_type': 'CRA_Guide',
            'parent_scraped_at': '2024-01-15T10:30:00Z',
        }

        chunks = [
            (
                'Content',
                [0.1] * 1536,
                SparseVector(indices=[1], values=[0.5]),
                metadata,
            )
        ]

        await client.store_documents(chunks)

        # Verify metadata is preserved in payload
        call_args = mock_qdrant_client.upsert.call_args
        point = call_args.kwargs['points'][0]

        assert point.payload['chunk_index'] == 2
        assert point.payload['total_chunks'] == 5
        assert point.payload['title'] == 'Complete Tax Guide'
        assert point.payload['url'] == 'https://canada.ca/tax-guide'
        assert point.payload['source'] == 'CRA'
        assert point.payload['doc_type'] == 'CRA_Guide'
        assert point.payload['scraped_at'] == '2024-01-15T10:30:00Z'

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    @pytest.mark.asyncio
    async def test_store_documents_uuid_generation(
        self, mock_qdrant_class, mock_qdrant_client
    ) -> None:
        """Test that unique IDs are generated for each point."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        chunks = [
            (
                f'Chunk {i}',
                [0.1] * 1536,
                SparseVector(indices=[1], values=[0.5]),
                {'chunk_index': i, 'total_chunks': 2},
            )
            for i in range(2)
        ]

        await client.store_documents(chunks)

        # Verify unique IDs
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args.kwargs['points']
        ids = [point.id for point in points]

        assert len(ids) == 2
        assert ids[0] != ids[1], "Each point should have a unique ID"


@pytest.mark.unit
class TestTaxDataQdrantClientSearch:
    """Test hybrid search functionality."""

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_search_dense_only(self, mock_qdrant_class, mock_qdrant_client) -> None:
        """Test search with dense vectors only (fallback when sparse_vector=None)."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        query_vector = [0.5] * 1536
        results = client.search(query_vector=query_vector, limit=5)

        # Verify query_points called with correct params
        mock_qdrant_client.query_points.assert_called_once()
        call_args = mock_qdrant_client.query_points.call_args
        assert call_args.kwargs['using'] == 'cra-dense'
        assert call_args.kwargs['limit'] == 5
        assert 'prefetch' not in call_args.kwargs, "Dense-only search should not use prefetch"

        # Verify results returned
        assert len(results) == 2

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_search_hybrid(self, mock_qdrant_class, mock_qdrant_client) -> None:
        """Test hybrid search with both dense and sparse vectors using Prefetch."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        query_vector = [0.5] * 1536
        sparse_vector = SparseVector(indices=[1, 2], values=[0.5, 0.3])

        results = client.search(
            query_vector=query_vector,
            sparse_vector=sparse_vector,
            limit=5
        )

        # Verify prefetch search was used
        call_args = mock_qdrant_client.query_points.call_args
        assert 'prefetch' in call_args.kwargs
        prefetch = call_args.kwargs['prefetch']
        assert len(prefetch) == 2, "Should prefetch from both dense and sparse vectors"

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_search_limit_parameter(self, mock_qdrant_class, mock_qdrant_client) -> None:
        """Test that limit parameter is correctly passed."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        query_vector = [0.5] * 1536

        # Test with different limits
        for limit in [1, 5, 10]:
            client.search(query_vector=query_vector, limit=limit)
            call_args = mock_qdrant_client.query_points.call_args
            assert call_args.kwargs['limit'] == limit


@pytest.mark.unit
class TestTaxDataQdrantClientUtilities:
    """Test utility methods."""

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_count_documents(self, mock_qdrant_class, mock_qdrant_client) -> None:
        """Test counting documents in collection."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        count = client.count_documents()
        assert count == 100

    @patch('tax_rag_scraper.storage.qdrant_client.QdrantClient')
    def test_get_collection_info(self, mock_qdrant_class, mock_qdrant_client) -> None:
        """Test retrieving collection information."""
        mock_qdrant_class.return_value = mock_qdrant_client

        client = TaxDataQdrantClient(
            url='https://test.cloud.qdrant.io',
            api_key='test-key',
            collection_name='cra-collection',
            source='cra',
        )

        info = client.get_collection_info()

        assert info['name'] == 'cra-collection'
        assert info['source'] == 'cra'
        assert info['dense_vector_name'] == 'cra-dense'
        assert info['sparse_vector_name'] == 'cra-sparse'
        assert info['points_count'] == 100
