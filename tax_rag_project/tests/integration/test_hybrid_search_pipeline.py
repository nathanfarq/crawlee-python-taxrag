"""Integration tests for hybrid search pipeline.

Tests the complete end-to-end workflow:
document → dense embedding → sparse embedding → storage → hybrid search
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from tax_rag_scraper.utils.embeddings import EmbeddingService, SparseEmbeddingService
from tax_rag_scraper.storage.qdrant_client import TaxDataQdrantClient


@pytest.mark.integration
class TestHybridSearchPipeline:
    """Test complete pipeline from document to search."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(
        self,
        sample_tax_document,
        mock_openai_api_key,
        mock_qdrant_url,
        mock_qdrant_api_key,
    ) -> None:
        """Test full pipeline: document → dense embed → sparse embed → store → hybrid search."""

        # Setup services
        with patch('tax_rag_scraper.storage.qdrant_client.QdrantClient') as mock_qdrant_class:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.name = 'test-collection'
            mock_client.get_collections.return_value = MagicMock(collections=[mock_collection])
            mock_qdrant_class.return_value = mock_client

            # Initialize services
            dense_service = EmbeddingService(api_key=mock_openai_api_key)
            sparse_service = SparseEmbeddingService()
            qdrant_client = TaxDataQdrantClient(
                url=mock_qdrant_url,
                api_key=mock_qdrant_api_key,
                collection_name='test-collection',
                source='test',
            )

            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_response.usage.total_tokens = 50
            dense_service.client.embeddings.create = AsyncMock(return_value=mock_response)

            # Step 1: Generate dense embeddings
            dense_chunks = await dense_service.embed_documents([sample_tax_document])
            assert len(dense_chunks) > 0, "Dense embeddings should be generated"

            # Step 2: Generate sparse embeddings
            chunk_texts = [text for text, _, _ in dense_chunks]
            sparse_vectors = sparse_service.embed_texts(chunk_texts)
            assert len(sparse_vectors) == len(dense_chunks), "Sparse vectors should match dense chunks"

            # Step 3: Combine into hybrid chunks (4-tuples)
            hybrid_chunks = [
                (text, dense, sparse, metadata)
                for (text, dense, metadata), sparse in zip(dense_chunks, sparse_vectors)
            ]

            # Verify hybrid chunk structure
            assert len(hybrid_chunks) > 0
            for chunk in hybrid_chunks:
                assert len(chunk) == 4, "Hybrid chunk should have 4 elements: text, dense, sparse, metadata"
                text, dense_vec, sparse_vec, metadata = chunk
                assert isinstance(text, str)
                assert len(dense_vec) == 1536
                assert sparse_vec.indices is not None
                assert isinstance(metadata, dict)

            # Step 4: Store in Qdrant
            await qdrant_client.store_documents(hybrid_chunks)

            # Verify storage called
            assert mock_client.upsert.called, "Qdrant upsert should be called"

            # Step 5: Hybrid search
            query_dense = [0.2] * 1536
            query_sparse = sparse_service.embed_query('tax deduction')

            mock_client.query_points.return_value = MagicMock(points=[])
            results = qdrant_client.search(
                query_vector=query_dense,
                sparse_vector=query_sparse,
                limit=5
            )

            # Verify hybrid search executed with prefetch
            assert mock_client.query_points.called, "Qdrant query should be called"
            call_args = mock_client.query_points.call_args
            assert 'prefetch' in call_args.kwargs, "Hybrid search should use prefetch"

    @pytest.mark.asyncio
    async def test_metadata_preserved_through_pipeline(
        self,
        sample_tax_document,
        mock_openai_api_key,
    ) -> None:
        """Test that metadata is preserved throughout the entire pipeline."""
        dense_service = EmbeddingService(api_key=mock_openai_api_key)

        # Mock OpenAI
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_response.usage.total_tokens = 50
        dense_service.client.embeddings.create = AsyncMock(return_value=mock_response)

        # Generate embeddings
        chunks = await dense_service.embed_documents([sample_tax_document])

        # Verify metadata is preserved
        assert len(chunks) > 0
        for text, embedding, metadata in chunks:
            # Check all metadata fields
            assert metadata['parent_title'] == sample_tax_document['title'], \
                "parent_title should be preserved"
            assert metadata['parent_url'] == sample_tax_document['url'], \
                "parent_url should be preserved"
            assert metadata['parent_source'] == sample_tax_document['source'], \
                "parent_source should be preserved"
            assert metadata['parent_doc_type'] == sample_tax_document['doc_type'], \
                "parent_doc_type should be preserved"
            assert metadata['parent_scraped_at'] == sample_tax_document['scraped_at'], \
                "parent_scraped_at should be preserved"

            # Check chunk metadata
            assert 'chunk_index' in metadata
            assert 'total_chunks' in metadata
            assert metadata['chunk_index'] >= 0
            assert metadata['total_chunks'] >= 1
            assert metadata['chunk_index'] < metadata['total_chunks'], \
                "chunk_index should be less than total_chunks"

    @pytest.mark.asyncio
    async def test_pipeline_with_large_document(
        self,
        sample_large_document,
        mock_openai_api_key,
    ) -> None:
        """Test pipeline with large document requiring chunking."""
        dense_service = EmbeddingService(api_key=mock_openai_api_key)
        sparse_service = SparseEmbeddingService()

        # Determine expected number of chunks
        full_text = f"Title: {sample_large_document['title']}\nContent: {sample_large_document['content']}"
        expected_chunks = dense_service._chunk_text(full_text)
        num_expected_chunks = len(expected_chunks)

        # Mock OpenAI to return embeddings for each chunk
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[float(i) / 10] * 1536) for i in range(num_expected_chunks)]
        mock_response.usage.total_tokens = 1000
        dense_service.client.embeddings.create = AsyncMock(return_value=mock_response)

        # Generate dense embeddings
        dense_chunks = await dense_service.embed_documents([sample_large_document])

        # Should create multiple chunks
        assert len(dense_chunks) > 1, "Large document should be split into multiple chunks"
        assert len(dense_chunks) == num_expected_chunks

        # Generate sparse embeddings for all chunks
        chunk_texts = [text for text, _, _ in dense_chunks]
        sparse_vectors = sparse_service.embed_texts(chunk_texts)

        assert len(sparse_vectors) == len(dense_chunks), \
            "Should have one sparse vector per dense chunk"

        # Verify all chunks have correct metadata
        for i, (text, dense_vec, metadata) in enumerate(dense_chunks):
            assert metadata['chunk_index'] == i
            assert metadata['total_chunks'] == num_expected_chunks
            assert metadata['parent_title'] == sample_large_document['title']
