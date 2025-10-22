import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import hashlib

from .chunker import content_chunker, DocumentChunk
from .embeddings import embedding_service
from .chroma_client import ChromaClient
from .retrieval import retrieval_pipeline

logger = logging.getLogger(__name__)

class KnowledgeBaseIngestion:
    """
    Knowledge base ingestion system for processing documents and
    storing them in ChromaDB collections with proper metadata
    """

    def __init__(self, chroma_client: ChromaClient = None):
        """
        Initialize KB ingestion system

        Args:
            chroma_client: ChromaDB client instance
        """
        self.chroma_client = chroma_client or ChromaClient()
        self.kb_directory = Path(__file__).parent.parent / "data" / "kb"
        self.supported_extensions = {'.md', '.txt', '.pdf'}

    async def ingest_all_documents(self, force_reingest: bool = False) -> Dict[str, Any]:
        """
        Ingest all documents from the knowledge base directory

        Args:
            force_reingest: Whether to force re-ingestion of existing documents

        Returns:
            Dictionary with ingestion statistics and results
        """
        logger.info("Starting knowledge base ingestion")

        stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_skipped": 0,
            "chunks_created": 0,
            "errors": [],
            "processing_time": 0
        }

        start_time = datetime.now()

        try:
            if not self.kb_directory.exists():
                logger.error(f"Knowledge base directory not found: {self.kb_directory}")
                stats["errors"].append(f"Directory not found: {self.kb_directory}")
                return stats

            # Find all supported files
            files_to_process = []
            for ext in self.supported_extensions:
                files_to_process.extend(self.kb_directory.glob(f"*{ext}"))
                # Also search subdirectories
                files_to_process.extend(self.kb_directory.rglob(f"*{ext}"))

            stats["files_found"] = len(files_to_process)

            logger.info(f"Found {stats['files_found']} files to process")

            # Process each file
            for file_path in files_to_process:
                try:
                    result = await self.ingest_single_document(file_path, force_reingest)

                    if result["status"] == "success":
                        stats["files_processed"] += 1
                        stats["chunks_created"] += result["chunks_created"]
                        logger.info(f"Successfully processed: {file_path.name}")
                    else:
                        stats["files_skipped"] += 1
                        if result.get("error"):
                            stats["errors"].append(f"{file_path.name}: {result['error']}")

                except Exception as e:
                    stats["files_skipped"] += 1
                    error_msg = f"Error processing {file_path.name}: {str(e)}"
                    stats["errors"].append(error_msg)
                    logger.error(error_msg)

            stats["processing_time"] = (datetime.now() - start_time).total_seconds()

            logger.info(f"Ingestion completed. Processed: {stats['files_processed']}, "
                       f"Chunks created: {stats['chunks_created']}, "
                       f"Time: {stats['processing_time']:.2f}s")

            return stats

        except Exception as e:
            error_msg = f"Knowledge base ingestion failed: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
            stats["processing_time"] = (datetime.now() - start_time).total_seconds()
            return stats

    async def ingest_single_document(self, file_path: Path, force_reingest: bool = False) -> Dict[str, Any]:
        """
        Inest a single document file

        Args:
            file_path: Path to the document file
            force_reingest: Whether to force re-ingestion if already processed

        Returns:
            Dictionary with processing results
        """
        try:
            # Check if file should be re-ingested
            file_hash = self._calculate_file_hash(file_path)
            document_id = file_path.stem

            if not force_reingest:
                existing_doc = self._check_document_exists(document_id)
                if existing_doc and existing_doc.get("file_hash") == file_hash:
                    return {
                        "status": "skipped",
                        "reason": "Document already exists and unchanged",
                        "document_id": document_id,
                        "chunks_created": 0
                    }

            # Read file content
            content = file_path.read_text(encoding='utf-8')

            # Process document
            chunks = content_chunker.process_document(
                content=content,
                source_file=str(file_path),
                additional_metadata={
                    "file_hash": file_hash,
                    "ingestion_date": datetime.now().isoformat(),
                    "file_extension": file_path.suffix
                }
            )

            if not chunks:
                return {
                    "status": "failed",
                    "error": "No chunks created from document",
                    "document_id": document_id,
                    "chunks_created": 0
                }

            # Determine collection based on metadata
            collection_name = self._determine_collection(chunks[0].metadata)

            # Remove existing document from collection if it exists
            if existing_doc:
                await self._remove_document_from_collection(collection_name, document_id)

            # Add to ChromaDB
            await self._add_chunks_to_collection(collection_name, chunks)

            logger.info(f"Successfully ingested {file_path.name}: {len(chunks)} chunks in {collection_name}")

            return {
                "status": "success",
                "document_id": document_id,
                "chunks_created": len(chunks),
                "collection": collection_name,
                "file_hash": file_hash
            }

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "document_id": file_path.stem,
                "chunks_created": 0
            }

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content for change detection"""
        try:
            content = file_path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {str(e)}")
            return datetime.now().isoformat()

    def _check_document_exists(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Check if document already exists in any collection"""
        try:
            collections = ["kb_global", "campus_resources"]

            for collection_name in collections:
                collection = getattr(self.chroma_client, collection_name, None)
                if collection:
                    result = collection.get(where={"document_id": document_id})
                    if result and result.get("ids"):
                        return {
                            "collection": collection_name,
                            "document_id": document_id,
                            "chunks_found": len(result["ids"]),
                            "metadata": result.get("metadatas", [[]])[0] if result.get("metadatas") else {}
                        }

            return None

        except Exception as e:
            logger.error(f"Error checking document existence: {str(e)}")
            return None

    def _determine_collection(self, metadata: Dict[str, Any]) -> str:
        """Determine which collection a document should belong to based on metadata"""

        category = metadata.get("category", "").lower()
        campus = metadata.get("campus", "").lower()

        if category == "campus_resources" or campus != "general":
            return "campus_resources"
        else:
            return "kb_global"

    async def _remove_document_from_collection(self, collection_name: str, document_id: str):
        """Remove existing document from collection"""
        try:
            collection = getattr(self.chroma_client, collection_name, None)
            if collection:
                result = collection.get(where={"document_id": document_id})
                if result and result.get("ids"):
                    collection.delete(ids=result["ids"])
                    logger.info(f"Removed existing document {document_id} from {collection_name}")

        except Exception as e:
            logger.error(f"Error removing document from collection: {str(e)}")

    async def _add_chunks_to_collection(self, collection_name: str, chunks: List[DocumentChunk]):
        """Add processed chunks to the specified ChromaDB collection"""
        try:
            collection = getattr(self.chroma_client, collection_name)
            if not collection:
                raise ValueError(f"Collection {collection_name} not found")

            # Prepare data for ChromaDB
            documents = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]

            # Add to collection in batches (if there are many chunks)
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )

            logger.info(f"Added {len(chunks)} chunks to {collection_name}")

        except Exception as e:
            logger.error(f"Error adding chunks to collection: {str(e)}")
            raise

    async def update_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Update a specific document (reprocess with new content)

        Args:
            file_path: Path to the document file to update

        Returns:
            Dictionary with update results
        """
        return await self.ingest_single_document(file_path, force_reingest=True)

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document from all collections

        Args:
            document_id: ID of the document to delete

        Returns:
            Dictionary with deletion results
        """
        result = {
            "document_id": document_id,
            "status": "success",
            "chunks_deleted": 0,
            "collections": []
        }

        try:
            collections = ["kb_global", "campus_resources"]

            for collection_name in collections:
                try:
                    collection = getattr(self.chroma_client, collection_name)
                    if collection:
                        # Find and delete chunks
                        search_result = collection.get(where={"document_id": document_id})

                        if search_result and search_result.get("ids"):
                            chunk_count = len(search_result["ids"])
                            collection.delete(ids=search_result["ids"])

                            result["chunks_deleted"] += chunk_count
                            result["collections"].append({
                                "name": collection_name,
                                "chunks_deleted": chunk_count
                            })

                            logger.info(f"Deleted {chunk_count} chunks from {collection_name}")

                except Exception as e:
                    logger.error(f"Error deleting from collection {collection_name}: {str(e)}")
                    result["status"] = "partial_error"

            if result["chunks_deleted"] == 0:
                result["status"] = "not_found"

            return result

        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current status of the knowledge base ingestion"""
        try:
            collection_stats = self.chroma_client.get_collection_stats()

            status = {
                "kb_directory": str(self.kb_directory),
                "directory_exists": self.kb_directory.exists(),
                "collection_stats": collection_stats,
                "supported_extensions": list(self.supported_extensions)
            }

            # Count files in directory
            if self.kb_directory.exists():
                status["files_in_directory"] = sum(
                    1 for ext in self.supported_extensions
                    for _ in self.kb_directory.glob(f"*{ext}")
                )

            return status

        except Exception as e:
            logger.error(f"Error getting ingestion status: {str(e)}")
            return {"error": str(e)}

    async def test_retrieval(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Test the retrieval system with sample queries

        Args:
            test_queries: List of test queries to run

        Returns:
            Dictionary with test results
        """
        if test_queries is None:
            test_queries = [
                "How can I manage exam stress?",
                "What are effective mindfulness techniques?",
                "Where can I find campus counseling services?",
                "How do I create a study schedule?",
                "What should I do if I'm feeling overwhelmed?"
            ]

        test_results = {
            "queries_tested": len(test_queries),
            "results": [],
            "total_retrieval_time": 0,
            "average_relevance": 0
        }

        total_relevance = 0
        total_time = 0

        for query in test_queries:
            try:
                start_time = datetime.now()
                response = await retrieval_pipeline.retrieve(
                    query=query,
                    k=5,
                    collections=["kb_global", "campus_resources"]
                )
                end_time = datetime.now()

                query_time = (end_time - start_time).total_seconds() * 1000
                total_time += query_time

                # Calculate average relevance score
                avg_relevance = sum(r.score for r in response.results) / len(response.results) if response.results else 0
                total_relevance += avg_relevance

                test_results["results"].append({
                    "query": query,
                    "results_count": len(response.results),
                    "retrieval_time_ms": query_time,
                    "average_relevance": avg_relevance,
                    "collections_searched": response.collections_searched
                })

            except Exception as e:
                test_results["results"].append({
                    "query": query,
                    "error": str(e)
                })

        test_results["total_retrieval_time"] = total_time
        test_results["average_relevance"] = total_relevance / len(test_queries) if test_queries else 0

        return test_results

# Global ingestion instance
kb_ingestion = KnowledgeBaseIngestion()