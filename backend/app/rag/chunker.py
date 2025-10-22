from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_file: str
    start_char: int
    end_char: int
    token_count: int

class ContentChunker:
    """
    Intelligent content chunker for processing knowledge base content.
    Implements semantic chunking with metadata preservation.
    """

    def __init__(self, chunk_size: int = 700, overlap: int = 100):
        """
        Initialize the content chunker

        Args:
            chunk_size: Target token count per chunk (500-800 recommended)
            overlap: Token overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = 200  # Minimum chunk size to avoid tiny chunks

        # Simple token approximation (roughly 1 token = 4 characters for English)
        self.chars_per_token = 4
        self.chunk_char_size = chunk_size * self.chars_per_token
        self.overlap_char_size = overlap * self.chars_per_token

    def parse_yaml_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse YAML frontmatter from markdown content

        Args:
            content: Raw document content with potential YAML frontmatter

        Returns:
            Tuple of (metadata_dict, content_without_frontmatter)
        """
        try:
            # Check if content starts with YAML frontmatter
            if content.startswith('---'):
                # Find the end of frontmatter
                end_marker = content.find('\n---', 3)
                if end_marker != -1:
                    frontmatter = content[3:end_marker]
                    body = content[end_marker + 4:]

                    # Parse YAML
                    metadata = yaml.safe_load(frontmatter)
                    if metadata is None:
                        metadata = {}

                    return metadata, body.strip()
                else:
                    # No end marker found, treat entire content as body
                    return {}, content
            else:
                # No frontmatter found
                return {}, content

        except yaml.YAMLError as e:
            logger.warning(f"Error parsing YAML frontmatter: {e}")
            return {}, content
        except Exception as e:
            logger.error(f"Unexpected error parsing frontmatter: {e}")
            return {}, content

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text (rough approximation)

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation: tokens ≈ chars / 4 for English
        # This can be enhanced with proper tokenizers if needed
        return len(text) // self.chars_per_token

    def split_by_structure(self, content: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split content by structural elements (headers, paragraphs, lists)

        Args:
            content: Content to split
            metadata: Base metadata to inherit

        Returns:
            List of (text, metadata) tuples
        """
        sections = []

        # Split by headings
        lines = content.split('\n')
        current_section = ""
        current_heading = ""
        heading_level = 0

        for line in lines:
            # Check for Markdown headers
            if line.strip().startswith('#'):
                # Save previous section if it exists
                if current_section.strip():
                    section_metadata = metadata.copy()
                    section_metadata.update({
                        "section_heading": current_heading,
                        "heading_level": heading_level,
                        "section_type": "content"
                    })
                    sections.append((current_section.strip(), section_metadata))

                # Start new section
                current_heading = line.strip()
                heading_level = len(line) - len(line.lstrip('#'))
                current_section = line + "\n"

            elif line.strip() and not line.startswith(('* ', '-', '+ ')):
                # Regular content line
                current_section += line + "\n"
            elif line.strip().startswith(('* ', '-', '+ ')):
                # List item - add to current section
                current_section += line + "\n"
            else:
                # Empty line - preserve formatting
                current_section += line + "\n"

        # Add the last section
        if current_section.strip():
            section_metadata = metadata.copy()
            section_metadata.update({
                "section_heading": current_heading,
                "heading_level": heading_level,
                "section_type": "content"
            })
            sections.append((current_section.strip(), section_metadata))

        return sections

    def create_chunks(self, text: str, base_metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Create semantic chunks from text

        Args:
            text: Text to chunk
            base_metadata: Base metadata for all chunks

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk_text = ""
        current_chunk_tokens = 0
        chunk_number = 0

        for i, paragraph in enumerate(paragraphs):
            paragraph_tokens = self.estimate_tokens(paragraph)

            # If paragraph is larger than max chunk size, split it
            if paragraph_tokens > self.chunk_size:
                # Save current chunk if it has content
                if current_chunk_text.strip():
                    chunks.append(self._create_chunk(
                        current_chunk_text, base_metadata, chunk_number,
                        0, len(current_chunk_text)
                    ))
                    chunk_number += 1
                    current_chunk_text = ""
                    current_chunk_tokens = 0

                # Split large paragraph into smaller chunks
                large_chunks = self._split_large_text(paragraph)
                for large_chunk in large_chunks:
                    chunks.append(self._create_chunk(
                        large_chunk, base_metadata, chunk_number,
                        0, len(large_chunk)
                    ))
                    chunk_number += 1
            else:
                # Check if adding this paragraph would exceed chunk size
                if current_chunk_tokens + paragraph_tokens > self.chunk_size and current_chunk_text.strip():
                    # Save current chunk and start a new one
                    chunks.append(self._create_chunk(
                        current_chunk_text, base_metadata, chunk_number,
                        0, len(current_chunk_text)
                    ))
                    chunk_number += 1
                    current_chunk_text = paragraph
                    current_chunk_tokens = paragraph_tokens
                else:
                    # Add to current chunk
                    if current_chunk_text:
                        current_chunk_text += "\n\n" + paragraph
                    else:
                        current_chunk_text = paragraph
                    current_chunk_tokens += paragraph_tokens

        # Add final chunk if it has content
        if current_chunk_text.strip():
            chunks.append(self._create_chunk(
                current_chunk_text, base_metadata, chunk_number,
                0, len(current_chunk_text)
            ))

        # Add overlap between consecutive chunks
        if self.overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_large_text(self, text: str) -> List[str]:
        """
        Split a large piece of text into smaller chunks

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Calculate end position
            end_pos = current_pos + self.chunk_char_size

            # Don't split words - find the last space before end_pos
            if end_pos < len(text):
                space_pos = text.rfind(' ', current_pos, end_pos)
                if space_pos != -1:
                    end_pos = space_pos

            # Extract chunk
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            # Move to next position with overlap
            current_pos = max(current_pos + 1, end_pos - self.overlap_char_size)

        return chunks

    def _add_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Add overlapping content between consecutive chunks

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of chunks with overlap added
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no previous overlap
                overlapped_chunks.append(chunk)
            else:
                # Add overlap from previous chunk
                previous_chunk = chunks[i-1]

                # Find overlap text (last few paragraphs of previous chunk)
                paragraphs = previous_chunk.text.split('\n\n')
                overlap_paragraphs = []
                overlap_chars = 0

                # Take paragraphs from the end until we reach desired overlap
                for paragraph in reversed(paragraphs):
                    if overlap_chars + len(paragraph) <= self.overlap_char_size:
                        overlap_paragraphs.insert(0, paragraph)
                        overlap_chars += len(paragraph)
                    else:
                        break

                # Create overlapped chunk
                if overlap_paragraphs:
                    overlap_text = '\n\n'.join(overlap_paragraphs)
                    overlapped_text = overlap_text + '\n\n' + chunk.text

                    overlapped_chunk = DocumentChunk(
                        text=overlapped_text,
                        metadata=chunk.metadata.copy(),
                        chunk_id=f"{chunk.metadata.get('document_id', 'doc')}_chunk_{i}_overlapped",
                        source_file=chunk.source_file,
                        start_char=max(0, chunk.start_char - overlap_chars),
                        end_char=chunk.end_char,
                        token_count=self.estimate_tokens(overlapped_text)
                    )

                    # Add overlap metadata
                    overlapped_chunk.metadata["has_overlap"] = True
                    overlapped_chunk.metadata["overlap_chars"] = overlap_chars

                    overlapped_chunks.append(overlapped_chunk)
                else:
                    overlapped_chunks.append(chunk)

        return overlapped_chunks

    def _create_chunk(self, text: str, metadata: Dict[str, Any], chunk_number: int,
                     start_char: int, end_char: int) -> DocumentChunk:
        """
        Create a DocumentChunk object

        Args:
            text: Chunk text
            metadata: Chunk metadata
            chunk_number: Chunk number within document
            start_char: Start character position
            end_char: End character position

        Returns:
            DocumentChunk object
        """
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_number": chunk_number,
            "chunk_type": "semantic",
            "token_count": self.estimate_tokens(text),
            "char_count": len(text)
        })

        return DocumentChunk(
            text=text,
            metadata=chunk_metadata,
            chunk_id=f"{metadata.get('document_id', 'doc')}_chunk_{chunk_number}",
            source_file=metadata.get("source_file", "unknown"),
            start_char=start_char,
            end_char=end_char,
            token_count=self.estimate_tokens(text)
        )

    def process_document(self, content: str, source_file: str,
                        additional_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Process a complete document with YAML frontmatter

        Args:
            content: Raw document content with YAML frontmatter
            source_file: Source file path or identifier
            additional_metadata: Additional metadata to include

        Returns:
            List of DocumentChunk objects
        """
        try:
            # Parse YAML frontmatter
            frontmatter_metadata, body_content = self.parse_yaml_frontmatter(content)

            # Create base metadata
            base_metadata = {
                "source_file": source_file,
                "document_id": Path(source_file).stem,
                "processed_at": str(Path(source_file).stat().st_mtime if Path(source_file).exists() else 0),
                "chunking_strategy": "semantic_with_structure"
            }

            # Add frontmatter metadata
            base_metadata.update(frontmatter_metadata)

            # Add additional metadata if provided
            if additional_metadata:
                base_metadata.update(additional_metadata)

            # Split by structure first
            structured_sections = self.split_by_structure(body_content, base_metadata)

            all_chunks = []

            for section_content, section_metadata in structured_sections:
                # Create chunks for this section
                section_chunks = self.create_chunks(section_content, section_metadata)
                all_chunks.extend(section_chunks)

            logger.info(f"Processed document {source_file}: created {len(all_chunks)} chunks")
            return all_chunks

        except Exception as e:
            logger.error(f"Error processing document {source_file}: {str(e)}")
            raise

    def process_multiple_documents(self, documents: List[Dict[str, str]],
                                 additional_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Process multiple documents

        Args:
            documents: List of dicts with 'content' and 'source_file' keys
            additional_metadata: Additional metadata to include for all documents

        Returns:
            List of all DocumentChunk objects from all documents
        """
        all_chunks = []

        for doc in documents:
            try:
                chunks = self.process_document(
                    doc["content"],
                    doc["source_file"],
                    additional_metadata
                )
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Error processing document {doc.get('source_file', 'unknown')}: {str(e)}")
                continue

        logger.info(f"Processed {len(documents)} documents: created {len(all_chunks)} total chunks")
        return all_chunks

    def get_chunking_stats(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunking results

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        token_counts = [chunk.token_count for chunk in chunks]
        char_counts = [len(chunk.text) for chunk in chunks]

        stats = {
            "total_chunks": len(chunks),
            "token_stats": {
                "min": min(token_counts),
                "max": max(token_counts),
                "average": sum(token_counts) / len(token_counts),
                "total": sum(token_counts)
            },
            "character_stats": {
                "min": min(char_counts),
                "max": max(char_counts),
                "average": sum(char_counts) / len(char_counts),
                "total": sum(char_counts)
            },
            "source_files": list(set(chunk.source_file for chunk in chunks)),
            "chunks_with_overlap": sum(1 for chunk in chunks if chunk.metadata.get("has_overlap", False))
        }

        return stats

# Global chunker instance
content_chunker = ContentChunker()