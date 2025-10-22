#!/usr/bin/env python3
"""
StudySense RAG System Demo
Demonstrates the Phase 2 RAG knowledge base functionality
"""

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from rag.kb_ingestion import kb_ingestion
from rag.retrieval import retrieval_pipeline
from rag.embeddings import embedding_service
from rag.chroma_client import chroma_client
from rag.chunker import content_chunker

class RAGDemo:
    """Demonstration class for the RAG system"""

    def __init__(self):
        self.demo_queries = [
            "How can I manage exam stress and anxiety?",
            "What are effective mindfulness techniques for students?",
            "Where can I find campus counseling services?",
            "How do I create a productive study schedule?",
            "What should I do if I'm feeling overwhelmed with coursework?",
            "What are evidence-based coping strategies for academic pressure?",
            "How can I improve my sleep hygiene during exam periods?",
            "What are some relaxation techniques I can use between classes?"
        ]

    async def run_complete_demo(self):
        """Run a complete demonstration of the RAG system"""
        print("🚀 StudySense Phase 2 RAG System Demo")
        print("=" * 60)

        try:
            # 1. Check system status
            await self._check_system_status()

            # 2. Ingest knowledge base (if needed)
            await self._ensure_knowledge_base()

            # 3. Demonstrate embedding capabilities
            await self._demo_embeddings()

            # 4. Show chunking process
            await self._demo_chunking()

            # 5. Demonstrate retrieval with multiple queries
            await self._demo_retrieval()

            # 6. Show advanced features
            await self._demo_advanced_features()

            print("\n🎉 Demo completed successfully!")
            print("The RAG system is ready for Phase 3 integration.")

        except Exception as e:
            print(f"\n❌ Demo failed: {str(e)}")
            raise

    async def _check_system_status(self):
        """Check system status and configurations"""
        print("\n📊 Checking System Status")
        print("-" * 30)

        # ChromaDB connection
        try:
            stats = chroma_client.get_collection_stats()
            print(f"✅ ChromaDB connected")
            for collection, info in stats.items():
                if isinstance(info, dict) and 'document_count' in info:
                    print(f"   {collection}: {info['document_count']} documents")
        except Exception as e:
            print(f"❌ ChromaDB connection failed: {e}")
            raise

        # Embedding service
        try:
            embedding_test = await embedding_service.test_embedding_service()
            if embedding_test["status"] == "success":
                print(f"✅ Embedding service ({embedding_test['provider']} - {embedding_test['model']})")
                print(f"   Dimension: {embedding_test['embedding_dimension']}")
            else:
                print(f"❌ Embedding service test failed: {embedding_test.get('error')}")
        except Exception as e:
            print(f"❌ Embedding service error: {e}")

    async def _ensure_knowledge_base(self):
        """Ensure knowledge base is ingested"""
        print("\n📚 Checking Knowledge Base")
        print("-" * 30)

        kb_status = kb_ingestion.get_ingestion_status()
        print(f"📁 KB Directory: {kb_status['kb_directory']}")
        print(f"📂 Directory exists: {kb_status['directory_exists']}")

        if kb_status['directory_exists']:
            # Check if we need ingestion
            total_docs = sum(
                info.get('document_count', 0)
                for info in kb_status['collection_stats'].values()
                if isinstance(info, dict)
            )

            if total_docs == 0:
                print("📥 Ingesting knowledge base...")
                ingestion_results = await kb_ingestion.ingest_all_documents()
                print(f"✅ Ingestion complete: {ingestion_results['chunks_created']} chunks created")
            else:
                print(f"✅ Knowledge base ready: {total_docs} chunks in collections")

    async def _demo_embeddings(self):
        """Demonstrate embedding capabilities"""
        print("\n🧠 Embedding Service Demo")
        print("-" * 30)

        test_texts = [
            "Managing exam stress effectively",
            "Mindfulness meditation techniques",
            "Campus counseling services"
        ]

        print("Generating embeddings for sample texts...")
        embeddings = await embedding_service.embed_text(test_texts)

        for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
            print(f"\n{i+1}. Text: {text}")
            print(f"   Embedding dimension: {len(embedding)}")
            print(f"   First 5 values: {[round(x, 4) for x in embedding[:5]]}")

        # Calculate similarity
        if len(embeddings) >= 2:
            similarity = embedding_service.calculate_similarity(embeddings[0], embeddings[1])
            print(f"\n📊 Similarity between first two texts: {similarity:.3f}")

    async def _demo_chunking(self):
        """Demonstrate content chunking"""
        print("\n✂️  Content Chunking Demo")
        print("-" * 30)

        sample_content = """
# Managing Academic Stress

## Introduction

Academic stress is a common experience for students at all levels. It can manifest in various ways including physical symptoms, emotional changes, and behavioral patterns.

## Identifying Stress Triggers

Common academic stress triggers include:

1. Upcoming exams and deadlines
2. Heavy course loads
3. Perfectionism
4. Time management challenges
5. Social pressures

## Effective Coping Strategies

### Time Management

Creating a structured study schedule helps manage workload and reduce last-minute stress.

### Mindfulness Techniques

Regular mindfulness practice can help reduce anxiety and improve focus.

### Physical Activity

Exercise releases endorphins and reduces stress hormones.
        """.strip()

        # Process the content
        chunks = content_chunker.process_document(
            content=sample_content,
            source_file="demo_content.md"
        )

        print(f"Created {len(chunks)} chunks from sample content")
        print(f"Target chunk size: {content_chunker.chunk_size} tokens")

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\nChunk {i+1}:")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Preview: {chunk.text[:100]}...")

    async def _demo_retrieval(self):
        """Demonstrate retrieval with sample queries"""
        print("\n🔍 Retrieval System Demo")
        print("-" * 30)

        print(f"Testing with {len(self.demo_queries)} sample queries...\n")

        total_results = 0
        total_time = 0

        for i, query in enumerate(self.demo_queries[:4], 1):  # Test first 4 queries
            print(f"{i}. Query: {query}")

            try:
                start_time = datetime.now()
                response = await retrieval_pipeline.retrieve(
                    query=query,
                    k=3,
                    collections=["kb_global", "campus_resources"]
                )
                query_time = (datetime.now() - start_time).total_seconds() * 1000

                print(f"   📊 Results: {len(response.results)} chunks in {query_time:.1f}ms")
                total_results += len(response.results)
                total_time += query_time

                if response.results:
                    best_result = response.results[0]
                    print(f"   🏆 Best match (score: {best_result.score:.3f}):")
                    print(f"      Source: {best_result.metadata.get('document_id', 'Unknown')}")
                    print(f"      Preview: {best_result.text[:120]}...")

            except Exception as e:
                print(f"   ❌ Error: {str(e)}")

        print(f"\n📈 Summary:")
        print(f"   Total results: {total_results}")
        print(f"   Average time: {total_time/4:.1f}ms per query")

    async def _demo_advanced_features(self):
        """Demonstrate advanced RAG features"""
        print("\n🔧 Advanced Features Demo")
        print("-" * 30)

        query = "stress management techniques"

        # 1. Filtered retrieval
        print("1. Filtered retrieval (stress_level='moderate'):")
        response = await retrieval_pipeline.retrieve(
            query=query,
            k=3,
            stress_level="moderate"
        )
        print(f"   Found {len(response.results)} results")

        # 2. Collection-specific search
        print("\n2. Campus resources only:")
        response = await retrieval_pipeline.retrieve(
            query=query,
            k=3,
            collections=["campus_resources"]
        )
        print(f"   Found {len(response.results)} results")

        # 3. User-context simulation (if available)
        print("\n3. Global knowledge base search:")
        response = await retrieval_pipeline.retrieve(
            query=query,
            k=3,
            collections=["kb_global"]
        )
        print(f"   Found {len(response.results)} results")

        # 4. Show re-ranking in action
        print("\n4. Re-ranking demonstration:")
        if response.results:
            print("   Top 3 results (re-ranked):")
            for i, result in enumerate(response.results[:3], 1):
                print(f"   {i}. Score: {result.score:.3f} | {result.metadata.get('document_id', 'Unknown')}")

    def print_usage_examples(self):
        """Print usage examples for developers"""
        print("\n💻 Usage Examples for Developers")
        print("=" * 40)

        usage_examples = '''
# Basic retrieval usage
from app.rag import retrieval_pipeline

response = await retrieval_pipeline.retrieve(
    query="How can I manage exam stress?",
    k=5,
    collections=["kb_global", "campus_resources"]
)

# Process results
for result in response.results:
    print(f"Content: {result.text}")
    print(f"Score: {result.score}")
    print(f"Source: {result.metadata}")

# Advanced retrieval with filters
response = await retrieval_pipeline.retrieve(
    query="counseling services",
    user_id="user123",
    stress_level="moderate",
    target_audience="undergraduate",
    k=10
)

# Knowledge base management
from app.rag import kb_ingestion

# Ingest new document
result = await kb_ingestion.ingest_single_document(Path("new_content.md"))

# Update existing document
result = await kb_ingestion.update_document(Path("existing_content.md"))

# Get system statistics
stats = chroma_client.get_collection_stats()
        '''

        print(usage_examples)

async def main():
    """Run the demonstration"""
    demo = RAGDemo()
    await demo.run_complete_demo()
    demo.print_usage_examples()

if __name__ == "__main__":
    asyncio.run(main())