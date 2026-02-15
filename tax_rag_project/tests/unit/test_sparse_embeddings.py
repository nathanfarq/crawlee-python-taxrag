"""Test suite for SparseEmbeddingService.

Tests the new sparse embedding feature using FastEmbed SPLADE model
for keyword matching in hybrid search.
"""

import pytest
from qdrant_client.models import SparseVector
from tax_rag_scraper.utils.embeddings import SparseEmbeddingService


@pytest.mark.unit
class TestSparseEmbeddingService:
    """Test cases for SparseEmbeddingService class."""

    def test_init_with_default_model(self) -> None:
        """Test initialization with default SPLADE model."""
        service = SparseEmbeddingService()
        assert service.model_name == 'prithivida/Splade_PP_en_v1'
        assert service.model is not None

    def test_embed_texts_single_text(self) -> None:
        """Test embedding a single text returns SparseVector."""
        service = SparseEmbeddingService()
        texts = ['Tax information about deductions']

        vectors = service.embed_texts(texts)

        assert len(vectors) == 1
        assert isinstance(vectors[0], SparseVector)
        assert len(vectors[0].indices) > 0
        assert len(vectors[0].values) > 0
        assert len(vectors[0].indices) == len(vectors[0].values), \
            "Indices and values must have same length"

    def test_embed_texts_multiple_texts(self) -> None:
        """Test embedding multiple texts."""
        service = SparseEmbeddingService()
        texts = [
            'Tax deduction information',
            'Capital gains reporting',
            'Income tax filing requirements'
        ]

        vectors = service.embed_texts(texts)

        assert len(vectors) == 3
        for vector in vectors:
            assert isinstance(vector, SparseVector)
            assert len(vector.indices) > 0
            assert len(vector.values) > 0

    def test_embed_texts_empty_list(self) -> None:
        """Test embedding empty list returns empty list."""
        service = SparseEmbeddingService()
        vectors = service.embed_texts([])
        assert vectors == []

    def test_embed_query(self) -> None:
        """Test embedding a single search query."""
        service = SparseEmbeddingService()
        query = 'How to claim tax deductions?'

        vector = service.embed_query(query)

        assert isinstance(vector, SparseVector)
        assert len(vector.indices) > 0
        assert len(vector.values) > 0

    def test_sparse_vectors_are_different(self) -> None:
        """Test that different texts produce different sparse vectors."""
        service = SparseEmbeddingService()
        texts = ['Tax deduction', 'Capital gains']

        vectors = service.embed_texts(texts)

        # Vectors should be different (different indices or values)
        assert vectors[0].indices != vectors[1].indices or \
               vectors[0].values != vectors[1].values, \
            "Different texts should produce different sparse vectors"

    def test_sparse_vector_values_are_positive(self) -> None:
        """Test that sparse vector values are positive (SPLADE property)."""
        service = SparseEmbeddingService()
        vector = service.embed_query('tax')

        for value in vector.values:
            assert value > 0, "SPLADE values should be positive"

    def test_sparse_vector_indices_are_sorted(self) -> None:
        """Test that sparse vector indices are sorted."""
        service = SparseEmbeddingService()
        vector = service.embed_query('tax deduction')

        assert vector.indices == sorted(vector.indices), \
            "Sparse vector indices should be sorted"

    def test_sparse_vector_length_validation(self) -> None:
        """Test that sparse vectors have reasonable length (not empty, not too large)."""
        service = SparseEmbeddingService()
        vector = service.embed_query('tax')

        # Sparse vectors should have at least 1 non-zero element
        assert len(vector.indices) >= 1, "Sparse vector should have at least 1 element"

        # Sparse vectors should not be unreasonably large (SPLADE typically < 1000)
        assert len(vector.indices) < 10000, "Sparse vector should not be excessively large"
