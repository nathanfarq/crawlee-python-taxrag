"""Qdrant Cloud client for storing tax documentation with hybrid vector embeddings."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Document, PointStruct, Prefetch

logger = logging.getLogger(__name__)


class TaxDataQdrantClient:
    """Client for interacting with Qdrant Cloud vector database.

    This client handles:
    - Collection validation (collections must be created manually in Qdrant UI)
    - Document storage with hybrid (dense + sparse) named vectors
    - Hybrid similarity search
    - Document counting and deletion
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        collection_name: str,
        source: str,
        vector_size: int = 1536,
    ) -> None:
        """Initialize Qdrant Cloud client.

        Args:
            url: Qdrant Cloud URL (e.g., 'https://xyz.cloud.qdrant.io')
            api_key: API key for authentication
            collection_name: Name of the collection (e.g., 'cra-collection')
            source: Source prefix for named vectors (e.g., 'cra' -> 'cra-dense', 'cra-sparse')
            vector_size: Dimension of dense embedding vectors (default: 1536 for OpenAI text-embedding-3-small)
        """
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.source = source
        self.vector_size = vector_size
        self.dense_vector_name = f'{source}-dense'
        self.sparse_vector_name = f'{source}-sparse'

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
        )

        # Validate that collection exists (must be created manually in Qdrant UI)
        self._validate_collection_exists()

    def _validate_collection_exists(self) -> None:
        """Validate that the collection exists in Qdrant Cloud.

        Collections must be created manually in the Qdrant UI with the required
        vector configuration before running scrapers.

        Raises:
            RuntimeError: If the collection does not exist.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                raise RuntimeError(
                    f"Collection '{self.collection_name}' does not exist in Qdrant Cloud.\n"
                    f"\n"
                    f"Please create it manually in the Qdrant UI (https://cloud.qdrant.io) with:\n"
                    f"  - Dense vector: '{self.dense_vector_name}' (size={self.vector_size}, distance=Cosine)\n"
                    f"  - Sparse vector: '{self.sparse_vector_name}' (modifier=IDF, required for BM25)\n"
                )

            logger.info(f"Collection '{self.collection_name}' validated successfully")

        except RuntimeError:
            raise
        except Exception:
            logger.exception('Error validating collection exists')
            raise

    async def store_documents(
        self, chunks: list[tuple[str, list[float], dict[str, Any]]]
    ) -> None:
        """Store document chunks with hybrid embeddings in Qdrant.

        Each chunk is stored as a separate point with dense and sparse vectors.
        BM25 sparse vectors are computed by Qdrant using the chunk text directly.

        Args:
            chunks: List of tuples containing (chunk_text, dense_embedding, metadata)
        """
        try:
            points = []
            for chunk_text, dense_embedding, metadata in chunks:
                point_id = str(uuid.uuid4())

                point = PointStruct(
                    id=point_id,
                    vector={
                        self.dense_vector_name: dense_embedding,
                        self.sparse_vector_name: Document(text=chunk_text, model='Qdrant/bm25'),
                    },
                    payload={
                        'chunk_text': chunk_text,
                        'chunk_index': metadata.get('chunk_index', 0),
                        'total_chunks': metadata.get('total_chunks', 1),
                        'title': metadata.get('parent_title', ''),
                        'url': metadata.get('parent_url', ''),
                        'source': metadata.get('parent_source', ''),
                        'doc_type': metadata.get('parent_doc_type', ''),
                        'scraped_at': metadata.get('parent_scraped_at', ''),
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.info(f"Stored {len(points)} chunks in '{self.collection_name}'")

        except Exception:
            logger.exception('Error storing chunks in Qdrant')
            raise

    def search(
        self,
        query_vector: list[float],
        query_text: str | None = None,
        limit: int = 5,
    ) -> list[Any]:
        """Search for similar documents using hybrid search.

        Args:
            query_vector: Dense embedding vector for the search query
            query_text: Query text for BM25 sparse keyword matching
            limit: Maximum number of results to return

        Returns:
            List of search results with scores and payloads
        """
        try:
            if query_text:
                # Hybrid search with prefetch
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(
                            query=query_vector,
                            using=self.dense_vector_name,
                            limit=limit * 2,
                        ),
                        Prefetch(
                            query=Document(text=query_text, model='Qdrant/bm25'),
                            using=self.sparse_vector_name,
                            limit=limit * 2,
                        ),
                    ],
                    query=query_vector,
                    using=self.dense_vector_name,
                    limit=limit,
                )
            else:
                # Dense-only fallback
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using=self.dense_vector_name,
                    limit=limit,
                )
        except Exception:
            logger.exception('Error searching Qdrant')
            raise
        else:
            return results.points

    def count_documents(self) -> int:
        """Count total documents in the collection."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            logger.exception('Error counting documents')
            return 0
        else:
            return collection_info.points_count or 0

    def delete_collection(self) -> None:
        """Delete the entire collection.

        Warning: This permanently deletes all documents in the collection.
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception:
            logger.exception('Error deleting collection')
            raise

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            logger.exception('Error getting collection info')
            raise
        else:
            return {
                'name': self.collection_name,
                'source': self.source,
                'points_count': info.points_count,
                'dense_vector_name': self.dense_vector_name,
                'sparse_vector_name': self.sparse_vector_name,
            }
