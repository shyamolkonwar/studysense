from typing import List, Optional, Union, Dict, Any
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import openai
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Provider-agnostic embedding service supporting multiple models"""

    def __init__(self, model_name: Optional[str] = None, provider: Optional[str] = None):
        """
        Initialize embedding service with specified model and provider

        Args:
            model_name: Name of the embedding model to use
            provider: Provider to use ("openai", "sentence_transformers", "huggingface")
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.provider = provider or "openai" if settings.OPENAI_API_KEY else "sentence_transformers"
        self._model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model based on provider"""
        try:
            if self.provider == "openai":
                if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your_openai_api_key_here" or len(settings.OPENAI_API_KEY.strip()) <= 10:
                    raise ValueError("Valid OpenAI API key is required for OpenAI embeddings")
                openai.api_key = settings.OPENAI_API_KEY
                # OpenAI embeddings are stateless, no model initialization needed
                logger.info(f"Using OpenAI embeddings with model: {self.model_name}")

            elif self.provider == "sentence_transformers":
                # Use a proper SentenceTransformer model
                if self.model_name.startswith("text-embedding") or self.model_name == "text-embedding-ada-002":
                    self.model_name = "all-MiniLM-L6-v2"
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")

            elif self.provider == "huggingface":
                # For future implementation with HuggingFace embeddings
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                logger.info(f"Loaded HuggingFace model: {self.model_name}")

            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            # Fallback to SentenceTransformer
            if self.provider != "sentence_transformers":
                logger.warning("Falling back to SentenceTransformer")
                self.provider = "sentence_transformers"
                self.model_name = "all-MiniLM-L6-v2"
                self._initialize_model()
            else:
                raise

    async def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s)

        Args:
            text: Single text string or list of texts to embed

        Returns:
            Embedding vector(s) as list(s) of floats
        """
        try:
            if isinstance(text, str):
                text = [text]
                single_text = True
            else:
                single_text = False

            if self.provider == "openai":
                embeddings = await self._embed_with_openai(text)
            elif self.provider == "sentence_transformers":
                embeddings = self._embed_with_sentence_transformers(text)
            elif self.provider == "huggingface":
                embeddings = await self._embed_with_huggingface(text)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            return embeddings[0] if single_text else embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI's API"""
        try:
            response = await openai.Embedding.acreate(
                model=self.model_name,
                input=texts
            )

            embeddings = [item['embedding'] for item in response['data']]
            return embeddings

        except Exception as e:
            logger.error(f"OpenAI embedding error: {str(e)}")
            raise

    def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using SentenceTransformer"""
        try:
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Convert numpy arrays to lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"SentenceTransformer embedding error: {str(e)}")
            raise

    async def _embed_with_huggingface(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using HuggingFace transformers"""
        try:
            # This is a basic implementation - can be enhanced with mean pooling
            import torch

            embeddings = []
            for text in texts:
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    # Use mean pooling of the last hidden state
                    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                    # Normalize the embedding
                    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                    embeddings.append(embedding[0].tolist())

            return embeddings

        except Exception as e:
            logger.error(f"HuggingFace embedding error: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        if self.provider == "openai":
            # OpenAI embedding dimensions
            if "ada" in self.model_name:
                return 1536
            elif "small" in self.model_name:
                return 1536
            elif "large" in self.model_name:
                return 3072
            else:
                return 1536  # default for most OpenAI models

        elif self.provider == "sentence_transformers":
            if self._model:
                return self._model.get_sentence_embedding_dimension()
            else:
                return 384  # default for all-MiniLM-L6-v2

        elif self.provider == "huggingface":
            if hasattr(self, '_model') and hasattr(self._model, 'config'):
                return self._model.config.hidden_size
            else:
                return 768  # common default for many models

        return 384  # safe default

    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0

    async def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed texts in batches for efficiency

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        try:
            if self.provider == "sentence_transformers":
                # SentenceTransformer has built-in batching
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=True
                )
                return embeddings.tolist()

            else:
                # For other providers, process in batches manually
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_embeddings = await self.embed_text(batch)
                    all_embeddings.extend(batch_embeddings)

                return all_embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise

    def validate_embeddings(self, embeddings: List[List[float]]) -> bool:
        """
        Validate that embeddings are properly formatted

        Args:
            embeddings: List of embedding vectors

        Returns:
            True if valid, False otherwise
        """
        if not embeddings:
            return False

        expected_dim = self.get_embedding_dimension()

        for embedding in embeddings:
            if not isinstance(embedding, list):
                return False
            if len(embedding) != expected_dim:
                return False
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            if any(np.isnan(x) or np.isinf(x) for x in embedding):
                return False

        return True

    async def test_embedding_service(self) -> Dict[str, Any]:
        """
        Test the embedding service with a sample text

        Returns:
            Dictionary with test results and metadata
        """
        test_text = "This is a test sentence for embedding generation."

        try:
            # Test single embedding
            embedding = await self.embed_text(test_text)

            # Test batch embedding
            batch_texts = [test_text, "Another test sentence", "Third test sentence"]
            batch_embeddings = await self.embed_text(batch_texts)

            # Calculate similarity
            similarity = self.calculate_similarity(embedding, batch_embeddings[0])

            results = {
                "status": "success",
                "provider": self.provider,
                "model": self.model_name,
                "embedding_dimension": self.get_embedding_dimension(),
                "test_embedding_length": len(embedding),
                "batch_embedding_count": len(batch_embeddings),
                "similarity_test": similarity,
                "validation_passed": self.validate_embeddings([embedding])
            }

            logger.info(f"Embedding service test passed: {results}")
            return results

        except Exception as e:
            error_result = {
                "status": "failed",
                "error": str(e),
                "provider": self.provider,
                "model": self.model_name
            }
            logger.error(f"Embedding service test failed: {error_result}")
            return error_result

# Default embedding service instance
embedding_service = EmbeddingService()
