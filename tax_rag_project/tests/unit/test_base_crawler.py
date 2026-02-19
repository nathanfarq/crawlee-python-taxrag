"""Test suite for TaxDataCrawler batch processing logic.

Tests document batching, flushing, and error handling in the base crawler.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from tax_rag_scraper.crawlers.base_crawler import TaxDataCrawler
from tax_rag_scraper.config.settings import Settings


@pytest.mark.unit
class TestTaxDataCrawlerBatching:
    """Test document batching functionality in TaxDataCrawler."""

    def test_batch_initialization(self) -> None:
        """Test that batch is initialized with correct size from settings."""
        # Create settings with custom batch size
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 10
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        # Mock Qdrant client to avoid initialization errors
        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient'):
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService'):
                crawler = TaxDataCrawler(
                    settings=settings,
                    use_qdrant=True,
                    qdrant_url='https://test.cloud.qdrant.io',
                    qdrant_api_key='test-key',
                )

                # Verify batch initialization
                assert crawler.batch_size == 10
                assert crawler.document_batch == []
                assert isinstance(crawler.document_batch, list)

    def test_batch_initialization_without_qdrant(self) -> None:
        """Test that batch_size is 0 when Qdrant is disabled."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 10

        crawler = TaxDataCrawler(settings=settings, use_qdrant=False)

        # Batch should be disabled
        assert crawler.batch_size == 0
        assert crawler.document_batch == []

    @pytest.mark.asyncio
    async def test_store_document_adds_to_batch(self) -> None:
        """Test that _store_document appends documents to batch."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 5
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient'):
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService'):
                crawler = TaxDataCrawler(
                    settings=settings,
                    use_qdrant=True,
                    qdrant_url='https://test.cloud.qdrant.io',
                    qdrant_api_key='test-key',
                )

                # Add documents to batch
                doc1 = {'title': 'Doc 1', 'content': 'Content 1'}
                doc2 = {'title': 'Doc 2', 'content': 'Content 2'}

                await crawler._store_document(doc1)
                assert len(crawler.document_batch) == 1
                assert crawler.document_batch[0] == doc1

                await crawler._store_document(doc2)
                assert len(crawler.document_batch) == 2
                assert crawler.document_batch[1] == doc2

    @pytest.mark.asyncio
    async def test_batch_flush_on_size(self) -> None:
        """Test that batch flushes automatically when it reaches batch_size."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 3  # Small batch for testing
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient') as mock_qdrant_class:
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService') as mock_embedding_class:
                # Setup mocks
                mock_qdrant = MagicMock()
                mock_qdrant.store_documents = AsyncMock()
                mock_qdrant_class.return_value = mock_qdrant

                mock_embedding = MagicMock()
                # Mock embed_documents to return 3-tuples: (text, embedding, metadata)
                mock_embedding.embed_documents = AsyncMock(return_value=[
                    ('Chunk 1', [0.1] * 1536, {'chunk_index': 0}),
                    ('Chunk 2', [0.2] * 1536, {'chunk_index': 1}),
                    ('Chunk 3', [0.3] * 1536, {'chunk_index': 2}),
                ])
                mock_embedding_class.return_value = mock_embedding

                crawler = TaxDataCrawler(
                    settings=settings,
                    use_qdrant=True,
                    qdrant_url='https://test.cloud.qdrant.io',
                    qdrant_api_key='test-key',
                )

                # Add documents one by one
                await crawler._store_document({'title': 'Doc 1', 'content': 'Content 1'})
                assert len(crawler.document_batch) == 1

                await crawler._store_document({'title': 'Doc 2', 'content': 'Content 2'})
                assert len(crawler.document_batch) == 2

                # Adding third document should trigger flush
                await crawler._store_document({'title': 'Doc 3', 'content': 'Content 3'})

                # Batch should be empty after flush
                assert len(crawler.document_batch) == 0

                # Verify embedding service was called
                mock_embedding.embed_documents.assert_called_once()
                mock_qdrant.store_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_flush_on_completion(self) -> None:
        """Test that remaining documents are flushed when crawler.run() completes."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 5
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient') as mock_qdrant_class:
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService') as mock_embedding_class:
                # Setup mocks
                mock_qdrant = MagicMock()
                mock_qdrant.store_documents = AsyncMock()
                mock_qdrant.count_documents = MagicMock(return_value=2)
                mock_qdrant_class.return_value = mock_qdrant

                mock_embedding = MagicMock()
                mock_embedding.embed_documents = AsyncMock(return_value=[
                    ('Chunk 1', [0.1] * 1536, {'chunk_index': 0}),
                    ('Chunk 2', [0.2] * 1536, {'chunk_index': 1}),
                ])
                mock_embedding_class.return_value = mock_embedding

                crawler = TaxDataCrawler(
                    settings=settings,
                    use_qdrant=True,
                    qdrant_url='https://test.cloud.qdrant.io',
                    qdrant_api_key='test-key',
                )

                # Add 2 documents (less than batch_size)
                crawler.document_batch = [
                    {'title': 'Doc 1', 'content': 'Content 1'},
                    {'title': 'Doc 2', 'content': 'Content 2'},
                ]

                # Mock the crawler.run() method to simulate completion
                crawler.crawler.run = AsyncMock()

                # Run the crawler
                await crawler.run(['https://example.com'])

                # Verify batch was flushed despite not reaching batch_size
                assert len(crawler.document_batch) == 0
                mock_embedding.embed_documents.assert_called_once()
                mock_qdrant.store_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_flush_batch_calls_embedding_service(self) -> None:
        """Test that _flush_batch calls the dense embedding service."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 5
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient') as mock_qdrant_class:
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService') as mock_embedding_class:
                # Setup mocks
                mock_qdrant = MagicMock()
                mock_qdrant.store_documents = AsyncMock()
                mock_qdrant_class.return_value = mock_qdrant

                mock_embedding = MagicMock()
                dense_chunks = [
                    ('Chunk 1', [0.1] * 1536, {'chunk_index': 0, 'parent_title': 'Doc 1'}),
                    ('Chunk 2', [0.2] * 1536, {'chunk_index': 1, 'parent_title': 'Doc 2'}),
                ]
                mock_embedding.embed_documents = AsyncMock(return_value=dense_chunks)
                mock_embedding_class.return_value = mock_embedding

                crawler = TaxDataCrawler(
                    settings=settings,
                    use_qdrant=True,
                    qdrant_url='https://test.cloud.qdrant.io',
                    qdrant_api_key='test-key',
                )

                # Add documents to batch
                documents = [
                    {'title': 'Doc 1', 'content': 'Content 1'},
                    {'title': 'Doc 2', 'content': 'Content 2'},
                ]
                crawler.document_batch = documents

                # Flush batch
                await crawler._flush_batch()

                # Verify dense embedding service was called with original documents
                mock_embedding.embed_documents.assert_called_once_with(documents)

                # Verify Qdrant was called with the dense chunks directly (BM25 computed by Qdrant)
                mock_qdrant.store_documents.assert_called_once_with(dense_chunks)

    @pytest.mark.asyncio
    async def test_flush_batch_stores_in_qdrant(self) -> None:
        """Test that _flush_batch calls qdrant_client.store_documents() with 3-tuple chunks."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 5
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient') as mock_qdrant_class:
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService') as mock_embedding_class:
                # Setup mocks
                mock_qdrant = MagicMock()
                mock_qdrant.store_documents = AsyncMock()
                mock_qdrant_class.return_value = mock_qdrant

                mock_embedding = MagicMock()
                dense_chunks = [
                    ('Chunk 1', [0.1] * 1536, {'chunk_index': 0}),
                ]
                mock_embedding.embed_documents = AsyncMock(return_value=dense_chunks)
                mock_embedding_class.return_value = mock_embedding

                crawler = TaxDataCrawler(
                    settings=settings,
                    use_qdrant=True,
                    qdrant_url='https://test.cloud.qdrant.io',
                    qdrant_api_key='test-key',
                )

                # Add document to batch
                crawler.document_batch = [{'title': 'Doc 1', 'content': 'Content 1'}]

                # Flush batch
                await crawler._flush_batch()

                # Verify Qdrant store_documents was called
                mock_qdrant.store_documents.assert_called_once()

                # Verify 3-tuple chunk structure (text, dense, metadata)
                call_args = mock_qdrant.store_documents.call_args[0][0]
                assert len(call_args) == 1
                chunk = call_args[0]
                assert len(chunk) == 3  # (text, dense, metadata)

                text, dense, metadata = chunk
                assert text == 'Chunk 1'
                assert dense == [0.1] * 1536
                assert metadata == {'chunk_index': 0}

    @pytest.mark.asyncio
    async def test_flush_batch_error_handling(self) -> None:
        """Test that _flush_batch logs errors but doesn't re-raise (prevents crawler crash)."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 5
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient') as mock_qdrant_class:
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService') as mock_embedding_class:
                with patch('tax_rag_scraper.crawlers.base_crawler.logger') as mock_logger:
                    # Setup mocks to raise error
                    mock_qdrant = MagicMock()
                    mock_qdrant_class.return_value = mock_qdrant

                    mock_embedding = MagicMock()
                    mock_embedding.embed_documents = AsyncMock(side_effect=Exception('Embedding failed'))
                    mock_embedding_class.return_value = mock_embedding

                    crawler = TaxDataCrawler(
                        settings=settings,
                        use_qdrant=True,
                        qdrant_url='https://test.cloud.qdrant.io',
                        qdrant_api_key='test-key',
                    )

                    # Add document to batch
                    crawler.document_batch = [{'title': 'Doc 1', 'content': 'Content 1'}]

                    # Flush batch - should not raise
                    await crawler._flush_batch()

                    # Verify error was logged
                    mock_logger.exception.assert_called_once_with('Error flushing batch')

                    # Batch should NOT be cleared on error (cleared inside try block)
                    assert len(crawler.document_batch) == 1

    @pytest.mark.asyncio
    async def test_empty_batch_no_flush(self) -> None:
        """Test that _flush_batch is a no-op when batch is empty."""
        settings = Settings()
        settings.EMBEDDING_BATCH_SIZE = 5
        settings.QDRANT_COLLECTION = 'test-collection'
        settings.QDRANT_SOURCE = 'test'

        with patch('tax_rag_scraper.crawlers.base_crawler.TaxDataQdrantClient') as mock_qdrant_class:
            with patch('tax_rag_scraper.crawlers.base_crawler.EmbeddingService') as mock_embedding_class:
                # Setup mocks
                mock_qdrant = MagicMock()
                mock_qdrant.store_documents = AsyncMock()
                mock_qdrant_class.return_value = mock_qdrant

                mock_embedding = MagicMock()
                mock_embedding.embed_documents = AsyncMock()
                mock_embedding_class.return_value = mock_embedding

                crawler = TaxDataCrawler(
                    settings=settings,
                    use_qdrant=True,
                    qdrant_url='https://test.cloud.qdrant.io',
                    qdrant_api_key='test-key',
                )

                # Ensure batch is empty
                crawler.document_batch = []

                # Flush batch
                await crawler._flush_batch()

                # Verify no embedding or storage calls were made
                mock_embedding.embed_documents.assert_not_called()
                mock_qdrant.store_documents.assert_not_called()
