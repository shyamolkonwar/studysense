#!/usr/bin/env python3
"""
Command line interface for StudySense RAG system
"""

import asyncio
import argparse
import sys
from pathlib import Path
import json
import logging
from typing import List, Dict, Any

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from rag.kb_ingestion import kb_ingestion
from rag.retrieval import retrieval_pipeline
from rag.embeddings import embedding_service
from rag.chroma_client import chroma_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def ingest_knowledge_base(force_reingest: bool = False):
    """Ingest all documents into the knowledge base"""
    print("🔍 Starting knowledge base ingestion...")

    try:
        # Check ChromaDB connection
        stats = chroma_client.get_collection_stats()
        print(f"📊 Current ChromaDB stats: {json.dumps(stats, indent=2)}")

        # Run ingestion
        results = await kb_ingestion.ingest_all_documents(force_reingest=force_reingest)

        print(f"\n✅ Ingestion completed!")
        print(f"📁 Files found: {results['files_found']}")
        print(f"✅ Files processed: {results['files_processed']}")
        print(f"⏭️  Files skipped: {results['files_skipped']}")
        print(f"🧩 Chunks created: {results['chunks_created']}")
        print(f"⏱️  Processing time: {results['processing_time']:.2f}s")

        if results['errors']:
            print(f"\n⚠️  Errors encountered:")
            for error in results['errors']:
                print(f"  • {error}")

        # Final stats
        final_stats = chroma_client.get_collection_stats()
        print(f"\n📊 Final ChromaDB stats: {json.dumps(final_stats, indent=2)}")

    except Exception as e:
        print(f"❌ Ingestion failed: {str(e)}")
        raise

async def test_retrieval():
    """Test the retrieval system with sample queries"""
    print("🔍 Testing retrieval system...")

    try:
        # Test embedding service
        print("\n🧠 Testing embedding service...")
        embedding_test = await embedding_service.test_embedding_service()
        print(f"   Embedding test: {json.dumps(embedding_test, indent=4)}")

        # Test retrieval with sample queries
        print("\n🔎 Testing retrieval queries...")
        test_results = await kb_ingestion.test_retrieval()

        print(f"Queries tested: {test_results['queries_tested']}")
        print(f"Total retrieval time: {test_results['total_retrieval_time']:.2f}ms")
        print(f"Average relevance: {test_results['average_relevance']:.3f}")

        print("\nDetailed results:")
        for result in test_results['results']:
            print(f"\nQuery: {result['query']}")
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  ✅ Results: {result['results_count']}")
                print(f"  ⏱️  Time: {result['retrieval_time_ms']:.2f}ms")
                print(f"  📊 Avg relevance: {result['average_relevance']:.3f}")

    except Exception as e:
        print(f"❌ Retrieval testing failed: {str(e)}")
        raise

async def interactive_query():
    """Interactive query interface"""
    print("🤖 Interactive Query Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)

    try:
        while True:
            query = input("\n🔍 Enter your query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit/q - Exit the interactive mode")
                print("  help - Show this help message")
                print("  stats - Show collection statistics")
                print("  Any other text will be treated as a search query")
                continue
            elif query.lower() == 'stats':
                stats = chroma_client.get_collection_stats()
                print("\n📊 Collection Statistics:")
                print(json.dumps(stats, indent=2))
                continue
            elif not query:
                continue

            # Perform retrieval
            try:
                print(f"\n🔍 Searching for: {query}")
                response = await retrieval_pipeline.retrieve(
                    query=query,
                    k=5,
                    collections=["kb_global", "campus_resources"]
                )

                if not response.results:
                    print("  ❌ No results found")
                    continue

                print(f"  📊 Found {len(response.results)} results in {response.metadata['retrieval_time_ms']:.2f}ms")
                print(f"  🗂️  Collections searched: {', '.join(response.collections_searched)}")

                for i, result in enumerate(response.results, 1):
                    print(f"\n  {i}. Score: {result.score:.3f} | Collection: {result.collection}")
                    print(f"     Document: {result.metadata.get('document_id', 'Unknown')}")
                    if result.metadata.get('section_heading'):
                        print(f"     Section: {result.metadata['section_heading']}")
                    print(f"     Content: {result.text[:200]}...")

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Interactive mode error: {str(e)}")

async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="StudySense RAG System CLI")
    parser.add_argument(
        "command",
        choices=["ingest", "test", "query", "all"],
        help="Command to execute"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion of all documents"
    )

    args = parser.parse_args()

    try:
        if args.command == "ingest":
            await ingest_knowledge_base(force_reingest=args.force)

        elif args.command == "test":
            await test_retrieval()

        elif args.command == "query":
            await interactive_query()

        elif args.command == "all":
            print("🚀 Running complete Phase 2 setup...")
            await ingest_knowledge_base(force_reingest=args.force)
            await test_retrieval()
            print("\n🎉 Phase 2 RAG system setup completed!")

    except Exception as e:
        print(f"❌ Command failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())