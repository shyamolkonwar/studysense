from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import asyncio
from .embeddings import EmbeddingService
from .chroma_client import ChromaClient

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a single retrieval result with metadata"""
    text: str
    metadata: Dict[str, Any]
    score: float
    collection: str
    id: str
    document_id: Optional[str] = None

@dataclass
class RetrievalResponse:
    """Complete retrieval response with multiple results"""
    query: str
    results: List[RetrievalResult]
    total_results: int
    query_time: datetime
    collections_searched: List[str]
    metadata: Dict[str, Any]

class RetrievalPipeline:
    """
    Advanced retrieval pipeline with hybrid search, metadata filtering,
    and result aggregation for mental health knowledge base
    """

    def __init__(self, chroma_client: ChromaClient = None, embedding_service: EmbeddingService = None):
        """
        Initialize retrieval pipeline

        Args:
            chroma_client: ChromaDB client instance
            embedding_service: Embedding service for query processing
        """
        self.chroma_client = chroma_client or ChromaClient()
        self.embedding_service = embedding_service or EmbeddingService()

        # Default retrieval configuration
        self.default_k = 10
        self.min_relevance_threshold = 0.05  # Set to reasonable threshold for SentenceTransformer
        self.collection_weights = {
            "kb_global": 1.0,
            "campus_resources": 0.9,
            "user_context": 1.2,  # Higher weight for user context
            "conversation_context": 1.1
        }

    async def retrieve(self,
                      query: str,
                      user_id: Optional[str] = None,
                      collections: Optional[List[str]] = None,
                      k: Optional[int] = None,
                      filters: Optional[Dict[str, Any]] = None,
                      stress_level: Optional[str] = None,
                      target_audience: Optional[str] = None,
                      campus: Optional[str] = None) -> RetrievalResponse:
        """
        Main retrieval method with hybrid search across multiple collections

        Args:
            query: User query
            user_id: Optional user ID for personalized results
            collections: List of collections to search (default: all relevant)
            k: Number of results to return
            filters: Additional metadata filters
            stress_level: Filter by stress level (mild, moderate, severe)
            target_audience: Filter by target audience
            campus: Filter by specific campus

        Returns:
            RetrievalResponse with ranked results
        """
        start_time = datetime.now()

        try:
            # Set defaults
            k = k or self.default_k
            collections = collections or self._get_default_collections(user_id)

            # Expand query with semantic expansion
            expanded_query = await self._expand_query(query)

            # Build metadata filters
            metadata_filters = self._build_filters(filters, stress_level, target_audience, campus)

            # Search across collections
            collection_results = await self._search_collections(
                expanded_query, collections, user_id, k, metadata_filters
            )

            # Aggregate and re-rank results
            ranked_results = self._aggregate_and_rerank(
                collection_results, collections, query, user_id
            )

            # Apply final filtering
            final_results = self._apply_final_filters(ranked_results, k, stress_level, target_audience)

            response = RetrievalResponse(
                query=query,
                results=final_results,
                total_results=len(final_results),
                query_time=start_time,
                collections_searched=collections,
                metadata={
                    "expanded_query": expanded_query,
                    "filters_applied": metadata_filters,
                    "collection_weights": self.collection_weights,
                    "retrieval_time_ms": (datetime.now() - start_time).total_seconds() * 1000
                }
            )

            logger.info(f"Retrieved {len(final_results)} results for query: {query[:50]}...")
            return response

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            raise

    def _get_default_collections(self, user_id: Optional[str]) -> List[str]:
        """Get default collections to search based on user context"""
        if user_id:
            return ["kb_global", "campus_resources", "user_context"]
        else:
            return ["kb_global", "campus_resources"]

    async def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with semantic synonyms and related terms

        Args:
            query: Original query

        Returns:
            List of query variations
        """
        # For now, return original query
        # In future versions, this could use LLM-based query expansion
        return [query]

    def _build_filters(self,
                       filters: Optional[Dict[str, Any]],
                       stress_level: Optional[str],
                       target_audience: Optional[str],
                       campus: Optional[str]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB-compatible metadata filters"""

        metadata_filters = filters.copy() if filters else {}

        # ChromaDB doesn't support MongoDB-style operators like $or, $exists
        # We need to use simple key-value filters

        if stress_level:
            # For stress level, we'll filter in the application layer since ChromaDB
            # doesn't support complex logical operators
            metadata_filters["stress_level_filter"] = stress_level

        if target_audience:
            metadata_filters["target_audience"] = target_audience

        if campus and campus != "general":
            metadata_filters["campus"] = campus

        # Return None if no filters to avoid ChromaDB errors
        return metadata_filters if metadata_filters else None

    async def _search_collections(self,
                                 queries: List[str],
                                 collections: List[str],
                                 user_id: Optional[str],
                                 k: int,
                                 filters: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Search across specified collections with all query variations"""

        collection_results = {}

        for collection_name in collections:
            try:
                results = []

                for query in queries:
                    if collection_name == "kb_global":
                        result = self.chroma_client.query_kb_global(
                            query=query,
                            n_results=k,
                            where=filters if filters else None
                        )
                    elif collection_name == "campus_resources":
                        result = self.chroma_client.query_campus_resources(
                            query=query,
                            institution=filters.get("campus", "general"),
                            n_results=k,
                            where=filters if filters else None
                        )
                    elif collection_name == "user_context" and user_id:
                        result = self.chroma_client.query_user_context(
                            user_id=user_id,
                            query=query,
                            n_results=k,
                            where=filters if filters else None
                        )
                    elif collection_name == "conversation_context" and user_id:
                        result = self.chroma_client.query_conversation_context(
                            user_id=user_id,
                            conversation_id="recent",  # Would be parameterized in real use
                            query=query,
                            n_results=k,
                            where=filters if filters else None
                        )
                    else:
                        continue

                    if result and result.get("documents") and result["documents"][0]:
                        results.append(result)

                if results:
                    collection_results[collection_name] = results

            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {str(e)}")
                continue

        return collection_results

    def _aggregate_and_rerank(self,
                             collection_results: Dict[str, List[Dict]],
                             collections: List[str],
                             original_query: str,
                             user_id: Optional[str]) -> List[RetrievalResult]:
        """Aggregate results from multiple collections and apply re-ranking"""

        all_results = []

        for collection_name, results_list in collection_results.items():
            collection_weight = self.collection_weights.get(collection_name, 1.0)

            for result in results_list:
                documents = result.get("documents", [[]])[0]
                metadatas = result.get("metadatas", [[]])[0]
                ids = result.get("ids", [[]])[0]
                distances = result.get("distances", [[]])[0]

                for i, doc in enumerate(documents):
                    if i < len(metadatas) and i < len(ids) and i < len(distances):
                        # Convert distance to similarity score (assuming cosine distance)
                        similarity_score = 1 - distances[i]

                        # Apply collection weight
                        weighted_score = similarity_score * collection_weight

                        retrieval_result = RetrievalResult(
                            text=doc,
                            metadata=metadatas[i] if metadatas[i] else {},
                            score=weighted_score,
                            collection=collection_name,
                            id=ids[i],
                            document_id=metadatas[i].get("document_id") if metadatas[i] else None
                        )

                        all_results.append(retrieval_result)

        # Remove duplicates and re-rank
        unique_results = self._remove_duplicates(all_results)
        ranked_results = self._rerank_results(unique_results, original_query, user_id)

        return ranked_results

    def _remove_duplicates(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicate results based on document_id and text similarity"""

        seen_texts = set()
        seen_doc_ids = set()
        unique_results = []

        for result in results:
            # Check document-based duplicate
            doc_id = result.document_id
            text_key = result.text[:200]  # First 200 chars as text key

            is_duplicate = False

            if doc_id and doc_id in seen_doc_ids:
                is_duplicate = True
            elif text_key in seen_texts:
                is_duplicate = True

            if not is_duplicate:
                unique_results.append(result)
                if doc_id:
                    seen_doc_ids.add(doc_id)
                seen_texts.add(text_key)

        return unique_results

    def _rerank_results(self,
                       results: List[RetrievalResult],
                       query: str,
                       user_id: Optional[str]) -> List[RetrievalResult]:
        """
        Apply advanced re-ranking based on multiple factors

        Re-ranking factors:
        1. Semantic similarity score
        2. Freshness/recency
        3. User personalization
        4. Evidence quality
        5. Content diversity
        """

        if not results:
            return []

        # Sort by score initially
        results.sort(key=lambda x: x.score, reverse=True)

        # Apply re-ranking factors
        for i, result in enumerate(results):
            rerank_score = result.score

            # Freshness boost (prefer more recent content)
            if "last_updated" in result.metadata:
                try:
                    last_updated = datetime.fromisoformat(result.metadata["last_updated"])
                    days_old = (datetime.now() - last_updated).days
                    freshness_boost = max(0, 1 - (days_old / 365))  # Decay over 1 year
                    rerank_score *= (1 + freshness_boost * 0.1)
                except:
                    pass

            # User personalization boost
            if user_id and result.collection == "user_context":
                rerank_score *= 1.2

            # Evidence quality boost based on metadata
            if result.metadata.get("evidence_level") == "high":
                rerank_score *= 1.15
            elif result.metadata.get("evidence_level") == "medium":
                rerank_score *= 1.05

            # Diversity penalty for similar content from same collection
            if i > 0:
                for j in range(max(0, i-3), i):  # Check previous 3 results
                    if results[j].collection == result.collection and results[j].document_id == result.document_id:
                        rerank_score *= 0.9
                        break

            # Update final score
            result.score = rerank_score

        # Final sort by re-ranked scores
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def _apply_final_filters(self,
                             results: List[RetrievalResult],
                             k: int,
                             stress_level: Optional[str],
                             target_audience: Optional[str]) -> List[RetrievalResult]:
        """Apply final filtering and result selection"""

        if not results:
            return []

        # Filter by minimum relevance threshold
        filtered_results = [
            result for result in results
            if result.score >= self.min_relevance_threshold
        ]

        # Apply specific filters if provided
        if stress_level:
            stress_filtered = []
            for result in filtered_results:
                result_stress_levels = result.metadata.get("stress_level", [])
                if not result_stress_levels or stress_level in result_stress_levels:
                    stress_filtered.append(result)
            filtered_results = stress_filtered

        if target_audience:
            audience_filtered = []
            for result in filtered_results:
                result_audiences = result.metadata.get("target_audience", [])
                if not result_audiences or target_audience in result_audiences:
                    audience_filtered.append(result)
            filtered_results = audience_filtered

        # Return top k results
        return filtered_results[:k]

    async def retrieve_similar_documents(self,
                                        document_id: str,
                                        k: int = 5,
                                        collection: str = "kb_global") -> List[RetrievalResult]:
        """
        Retrieve documents similar to a specific document

        Args:
            document_id: ID of reference document
            k: Number of similar documents to return
            collection: Collection to search in

        Returns:
            List of similar documents
        """
        try:
            # This would be implemented with ChromaDB's get and query methods
            # For now, return empty list
            logger.warning("Similar document retrieval not yet implemented")
            return []

        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}")
            return []

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retrieval pipeline"""
        try:
            collection_stats = self.chroma_client.get_collection_stats()

            stats = {
                "pipeline_config": {
                    "default_k": self.default_k,
                    "min_relevance_threshold": self.min_relevance_threshold,
                    "collection_weights": self.collection_weights
                },
                "collection_stats": collection_stats,
                "embedding_info": {
                    "model": self.embedding_service.model_name,
                    "provider": self.embedding_service.provider,
                    "dimension": self.embedding_service.get_embedding_dimension()
                }
            }

            return stats

        except Exception as e:
            logger.error(f"Error getting retrieval statistics: {str(e)}")
            return {"error": str(e)}

# Global retrieval pipeline instance
retrieval_pipeline = RetrievalPipeline()
