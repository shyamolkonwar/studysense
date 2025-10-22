import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class ChromaClient:
    def __init__(self):
        # Connect to ChromaDB server
        self.client = chromadb.HttpClient(host="localhost", port=8000)

        # Use OpenAI embedding function if API key is available and valid, otherwise use SentenceTransformer
        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here" and len(settings.OPENAI_API_KEY.strip()) > 10:
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.EMBEDDING_MODEL
            )
        else:
            # Use SentenceTransformer embedding function for development/testing
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.warning("No valid OpenAI API key provided, using SentenceTransformer embedding function. This is not recommended for production.")

        # Initialize collections
        self._init_collections()

    def _init_collections(self):
        """Initialize all required collections with proper schemas"""

        # Global knowledge base collection
        self.kb_global = self._get_or_create_collection(
            name="kb_global",
            metadata={
                "description": "Global knowledge base with mental health resources",
                "type": "knowledge_base",
                "content_types": "articles,resources,guidelines",
                "language": "en",
                "version": "1.0"
            }
        )

        # Campus-specific resources collection
        self.campus_resources = self._get_or_create_collection(
            name="campus_resources",
            metadata={
                "description": "Institution-specific resources and services",
                "type": "campus_resources",
                "content_types": "counseling,support_groups,campus_services",
                "language": "en",
                "version": "1.0"
            }
        )

        # User context collection (per-user contextual embeddings)
        self.user_context = self._get_or_create_collection(
            name="user_context",
            metadata={
                "description": "User-specific context and behavioral patterns",
                "type": "user_context",
                "content_types": "messages,activities,conversations",
                "access_level": "private",
                "retention_days": str(settings.MESSAGE_RETENTION_DAYS),
                "version": "1.0"
            }
        )

        # Conversation context collection
        self.conversation_context = self._get_or_create_collection(
            name="conversation_context",
            metadata={
                "description": "Conversation history and context for chat continuity",
                "type": "conversation_context",
                "content_types": "chat_messages,recommendations,feedback",
                "access_level": "private",
                "retention_days": str(settings.MESSAGE_RETENTION_DAYS),
                "version": "1.0"
            }
        )

        # Risk patterns collection
        self.risk_patterns = self._get_or_create_collection(
            name="risk_patterns",
            metadata={
                "description": "Anonymized risk patterns and markers",
                "type": "risk_patterns",
                "content_types": "risk_markers,anomalies,patterns",
                "access_level": "research",
                "anonymized": "true",
                "version": "1.0"
            }
        )

        logger.info("All ChromaDB collections initialized successfully")

    def _get_or_create_collection(self, name: str, metadata: Dict[str, Any]):
        """Get or create a collection with the specified metadata"""
        try:
            collection = self.client.get_collection(
                name=name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Collection '{name}' already exists, using existing collection")
            return collection
        except Exception as e:
            logger.info(f"Collection '{name}' does not exist or error getting it: {e}, creating new collection")
            try:
                collection = self.client.create_collection(
                    name=name,
                    embedding_function=self.embedding_function,
                    metadata=metadata
                )
                logger.info(f"Created new collection '{name}'")
                return collection
            except Exception as create_e:
                if "already exists" in str(create_e).lower():
                    logger.info(f"Collection '{name}' already exists, getting without embedding function")
                    # Try to get collection without embedding function if it already exists
                    collection = self.client.get_collection(name=name)
                    logger.info(f"Retrieved existing collection '{name}'")
                    return collection
                else:
                    logger.error(f"Failed to create collection '{name}': {create_e}")
                    raise

    def add_to_kb_global(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to the global knowledge base"""
        return self.kb_global.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def add_to_campus_resources(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add campus-specific resources"""
        return self.campus_resources.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def add_user_context(self, user_id: str, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add user-specific context with user filtering"""
        # Add user_id to all metadata for filtering
        for metadata in metadatas:
            metadata["user_id"] = user_id

        return self.user_context.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def add_conversation_context(self, user_id: str, conversation_id: str, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add conversation context with user and conversation filtering"""
        # Add identifiers to all metadata for filtering
        for metadata in metadatas:
            metadata["user_id"] = user_id
            metadata["conversation_id"] = conversation_id

        return self.conversation_context.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query_kb_global(self, query: str, n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        """Query global knowledge base"""
        # Filter out unsupported MongoDB-style operators
        filtered_where = self._filter_chromadb_compatible_where(where)

        return self.kb_global.query(
            query_texts=[query],
            n_results=n_results,
            where=filtered_where
        )

    def query_campus_resources(self, query: str, institution: str, n_results: int = 5, where: Optional[Dict] = None) -> Dict:
        """Query campus-specific resources"""
        # Filter out unsupported MongoDB-style operators
        filtered_where = self._filter_chromadb_compatible_where(where)

        # Add institution filter if provided
        if filtered_where is None:
            filtered_where = {}
        filtered_where["institution"] = institution

        return self.campus_resources.query(
            query_texts=[query],
            n_results=n_results,
            where=filtered_where
        )

    def query_user_context(self, user_id: str, query: str, n_results: int = 10, where: Optional[Dict] = None) -> Dict:
        """Query user-specific context with privacy filtering"""
        if where is None:
            where = {}
        where["user_id"] = user_id

        return self.user_context.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

    def query_conversation_context(self, user_id: str, conversation_id: str, query: str, n_results: int = 10, where: Optional[Dict] = None) -> Dict:
        """Query conversation context with filtering"""
        if where is None:
            where = {}
        where["user_id"] = user_id
        where["conversation_id"] = conversation_id

        return self.conversation_context.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

    def hybrid_search(self, query: str, collections: List[str], user_id: Optional[str] = None, n_results: int = 10) -> Dict[str, Any]:
        """Perform hybrid search across multiple collections"""
        results = {}

        for collection_name in collections:
            try:
                if collection_name == "kb_global":
                    result = self.query_kb_global(query, n_results)
                elif collection_name == "campus_resources":
                    result = self.query_campus_resources(query, "default", n_results)
                elif collection_name == "user_context" and user_id:
                    result = self.query_user_context(user_id, query, n_results)
                elif collection_name == "conversation_context" and user_id:
                    result = self.query_conversation_context(user_id, "recent", query, n_results)
                else:
                    continue

                results[collection_name] = result

            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {str(e)}")
                continue

        return results

    def delete_user_data(self, user_id: str):
        """Delete all data associated with a user (GDPR compliance)"""
        try:
            # Delete from user_context collection
            user_context_results = self.user_context.get(where={"user_id": user_id})
            if user_context_results["ids"]:
                self.user_context.delete(ids=user_context_results["ids"])

            # Delete from conversation_context collection
            conversation_results = self.conversation_context.get(where={"user_id": user_id})
            if conversation_results["ids"]:
                self.conversation_context.delete(ids=conversation_results["ids"])

            logger.info(f"Successfully deleted all data for user {user_id}")

        except Exception as e:
            logger.error(f"Error deleting user data for {user_id}: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about all collections"""
        stats = {}

        collections_info = {
            "kb_global": self.kb_global,
            "campus_resources": self.campus_resources,
            "user_context": self.user_context,
            "conversation_context": self.conversation_context,
            "risk_patterns": self.risk_patterns
        }

        for name, collection in collections_info.items():
            try:
                count = collection.count()
                stats[name] = {
                    "document_count": count,
                    "metadata": collection.metadata
                }
            except Exception as e:
                logger.error(f"Error getting stats for {name}: {str(e)}")
                stats[name] = {"error": str(e)}

        return stats

    def _filter_chromadb_compatible_where(self, where: Optional[Dict]) -> Optional[Dict]:
        """Filter out MongoDB-style operators that ChromaDB doesn't support"""
        if not where:
            return None

        # ChromaDB supports simple key-value equality filters
        # Remove any MongoDB-style operators like $or, $exists, etc.
        filtered_where = {}

        for key, value in where.items():
            # Skip MongoDB operators
            if key.startswith('$'):
                continue

            # Skip complex objects with operators
            if isinstance(value, dict):
                # Check if it's a simple value or contains operators
                has_operators = any(k.startswith('$') for k in value.keys())
                if not has_operators:
                    filtered_where[key] = value
                # Skip fields with operators like {"$exists": False}
            else:
                # Simple key-value pair
                filtered_where[key] = value

        return filtered_where if filtered_where else None


# Global instance
chroma_client = ChromaClient()
