from __future__ import annotations
from typing import List, Dict, Tuple
import json
import os
import openai
import uuid
import math
import re
import numpy as np
from collections import Counter, defaultdict
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from memalpha.utils import count_tokens


class Memory:
    """Holds core, semantic, episodic memories entirely in RAM."""

    # Maximum number of items to show for semantic and episodic memories
    MAX_MEMORY_ITEMS = 20
    MEMORY_CONSOLIDATE_STEP = 5 # The number of memories to consolidate at a time
    MODEL = "gpt-4.1-mini"  # Same model as agent.py
    TOPK = 20

    def __init__(self, including_core: bool = False) -> None:
        if including_core:
            self.core: str = ""  # Changed to simple string
        else:
            self.core = None
        self.instructions = None
        self.semantic: List[Dict[str, str]] = []
        self.episodic: List[Dict[str, str]] = []
        # Embeddings stored as matrices for batch operations
        self.semantic_embedding_matrix: np.ndarray = np.empty((0, 1536))  # text-embedding-3-small has 1536 dimensions
        self.episodic_embedding_matrix: np.ndarray = np.empty((0, 1536))
        # Memory ID mappings to track which row corresponds to which memory
        self.semantic_embedding_ids: List[str] = []
        self.episodic_embedding_ids: List[str] = []
        self.including_core = including_core

    def total_length(self):
        total_length = 0
        if self.core is not None:
            # Core is now a simple string
            total_length += count_tokens(self.core)
        
        # Handle semantic and episodic memories
        for mem_type, mem_list in [("semantic", self.semantic), ("episodic", self.episodic)]:
            for mem_idx, mem in enumerate(mem_list):
                for mem_id, content in mem.items():
                    # Debug: Check if content is problematic with detailed info
                    if not isinstance(content, str):
                        print(f"!!!! MEMORY ERROR: Non-string content found!")
                        print(f"  Memory type: {mem_type}")
                        print(f"  Memory index: {mem_idx}")
                        print(f"  Memory ID: {mem_id}")
                        print(f"  Content: {repr(content)}")
                        print(f"  Content type: {type(content)}")
                        print(f"  Memory object ID: {id(mem)}")
                        print(f"  Full memory item: {repr(mem)}")
                        # Also check if it's a numpy/torch type
                        if hasattr(content, 'item'):
                            print(f"  Has .item() method, value: {content.item()}")
                        if hasattr(content, 'dtype'):
                            print(f"  Has dtype: {content.dtype}")
                    total_length += count_tokens(content)
        
        return total_length

    def _generate_memory_id(self) -> str:
        """Generate a unique ID for a memory item."""
        return str(uuid.uuid4())[:4]  # Using first 4 characters of UUID

    def _content_exists(self, memory_type: str, content: str) -> bool:
        """Check if content already exists in the specified memory type."""
        if memory_type == 'core':
            return self.core == content if self.core is not None else False
        else:
            mem_list = getattr(self, memory_type)
            for mem in mem_list:
                for _, existing_content in mem.items():
                    if existing_content == content:
                        return True
            return False

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI's embedding model."""
        try:
            load_dotenv()
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            # print(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return np.zeros(1536)  # text-embedding-3-small has 1536 dimensions

    # --------------------------------------------------
    # Rendering helpers
    # --------------------------------------------------
    def _block(self, title: str = '', lines: List[Dict[str, str]] = [], content: str = None) -> str:
        # Handle simple string content (for core memory)
        if content is not None:
            if title:
                return f"<{title}>\n{content}\n</{title}>"
            else:
                return content

        # Handle list of dictionaries (for semantic/episodic memories)
        if not lines:
            if title:
                return f"<{title}>\nEmpty.\n</{title}>"
            else:
                return f"Empty."
        
        # Convert each memory dict to a string representation
        formatted_lines = []
        for mem in lines:
            for mem_id, content in mem.items():
                formatted_lines.append(f"[{mem_id}] {content}")
        
        body = "\n".join(formatted_lines)
        if title:
            return f"<{title}>\n{body}\n</{title}>"
        else:
            return body

    def render_system_prompt(self, status: str = "chat", query: str = None, max_num_of_recent_chunks: int = None) -> List[Dict[str, str]]:
        """Return the system prompt expected by the model.
        
        Args:
            status: The mode of operation, can be:
                - "memorie": For memorizing and storing information
                - "chat": For normal conversation and information retrieval
                - "rethink": For memory consolidation and reorganization
        """
        
        max_num_of_recent_chunks = max_num_of_recent_chunks if max_num_of_recent_chunks is not None else self.MAX_MEMORY_ITEMS
        if max_num_of_recent_chunks > 0:
            # If max_num_of_recent_chunks is larger than actual memory count, use all memories
            if max_num_of_recent_chunks >= len(self.semantic):
                semantic_items = self.semantic
            else:
                semantic_items = self.semantic[-max_num_of_recent_chunks:]
                
            if max_num_of_recent_chunks >= len(self.episodic):
                episodic_items = self.episodic
            else:
                episodic_items = self.episodic[-max_num_of_recent_chunks:]
        else:
            episodic_items = []
            semantic_items = []
        
        # Handle core memory based on including_core flag
        if self.including_core and self.core is not None:
            core_block = self._block('core_memory', content=self.core)
            memory_blocks = (
                f"{core_block}\n\n"
                f"{self._block('semantic_memory', semantic_items)}\n\n"
                f"{self._block('episodic_memory', episodic_items)}"
            )
        else:
            memory_blocks = (
                f"{self._block('semantic_memory', semantic_items)}\n\n"
                f"{self._block('episodic_memory', episodic_items)}"
            )
        
        if status == "memorie":
            # System prompt for memorizing mode - focus on understanding and storing information
            core_memory_section = ""
            if self.including_core and self.core is not None:
                core_memory_section = f"""
<core_memory>
{self.core}
</core_memory>
"""
            
            system_prompt = (f'''You are a personal assistant with a sophisticated memory system. Your primary task is to carefully analyze, understand, and memorize the information provided by the user.

MEMORIZING MODE INSTRUCTIONS:
- Read and understand all information shared by the user
- Identify key facts, concepts, and relationships
- Store important information using the appropriate memory type:
{"* core_memory: Information stored so far (stored as a compact paragraph)" if self.including_core else ""}
* semantic_memory: General knowledge, factual or conceptual information
* episodic_memory: Specific personal experiences or events with timestamp (mandatory), place, or context
- Use these cues to decide memory type based on content

CURRENT MEMORY STATE:
{core_memory_section}
<semantic_memory> (Only show the most recent {min(len(semantic_items), len(self.semantic))} out of {len(self.semantic)} memories)
{self._block(lines=semantic_items)}
</semantic_memory>

<episodic_memory> (Only show the most recent {min(len(episodic_items), len(self.episodic))} out of {len(self.episodic)} memories)
{self._block(lines=episodic_items)}
</episodic_memory>

Focus on understanding and memorizing. Use memory tools actively to store new information.
Since this is the memorization process, if you think all the information has been memorized, you can respond with 'Done'. This information will not be seen by the user.
Meanwhile, you will be queried only once, so make sure to call all the memory insertion functions in one turn.''')

            return [
                {"role": "system", "content": system_prompt},
            ]
        
        elif status == "rethink":
            # System prompt for memory consolidation mode
            system_prompt = f'''You are a memory consolidation specialist tasked with optimizing the memory system's organization and efficiency.

CONSOLIDATION OBJECTIVES:
1. **Redundancy Elimination**: Minimize redundant information while preserving all critical data. Restructure and rephrase memory entries for optimal clarity and conciseness without data loss. To do this, you can use the memory_delete and memory_update functions.

2. **Information Synthesis**: Generate additional insights and inferences from existing data patterns to enhance the knowledge base comprehensiveness. To do this, you can use the memory_update and memory_insert function.

3. **Memory Organization**: Identify patterns and relationships between different memories to create a more coherent and accessible memory structure.'''

            user_message = f"The following is the current memory state:\n\n{memory_blocks}\n\nPlease use all the functions to delete old memories, update existing memories and generate new inferred memories."
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

        else:  # status == "chat"
            # System prompt for answering mode - focus on retrieving and using stored information
            # Check if we're showing all memories or just recent ones
            showing_all_semantic = len(semantic_items) == len(self.semantic)
            showing_all_episodic = len(episodic_items) == len(self.episodic)
            
            if showing_all_semantic and showing_all_episodic:
                semantic_desc = f"All {len(self.semantic)} semantic memories"
                episodic_desc = f"All {len(self.episodic)} episodic memories"
                search_instructions = "All memories are available in the context. You can directly use the information to answer the query."
            else:
                semantic_desc = f"Only show the most relevant {len(semantic_items)} out of {len(self.semantic)} memories retrieved using bm25 search with the query '''{query}'''"
                episodic_desc = f"Only show the most relevant {len(episodic_items)} out of {len(self.episodic)} memories retrieved using bm25 search with the query '''{query}'''"
                search_instructions = "If you want to look closer or conduct more searches, you can adjust the query and call the `search_memory` function again. You can also set search_method as 'text-embedding' to use embedding similarity search. Be an active searcher and try to use all kinds of queries and search methods to find the results. Do not easily give up."
            
            core_memory_section = ""
            if self.including_core and self.core is not None:
                core_memory_section = f"""
<core_memory>
{self.core}
</core_memory>
"""
            
            system_prompt = (
                "You are a reasoning assistant. For each incoming query or task—whether it's a question, command, or summary request—"
                "use the structured memory below to retrieve and synthesize relevant information to produce your response.\n\n"
                f"""Based on the query {query}, the following are the retrieved memories:
{core_memory_section}
<semantic_memory> ({semantic_desc})
{self._block(lines=semantic_items)}
</semantic_memory>

<episodic_memory> ({episodic_desc})
{self._block(lines=episodic_items)}
</episodic_memory>

{search_instructions}"""
            )
            return [
                {"role": "system", "content": system_prompt},
            ]

    # --------------------------------------------------
    # Memory operations – called by functions.py
    # --------------------------------------------------
    def new_memory_insert(self, memory_type: str, content: str):
        """Insert a new memory with a unique ID. Skips insertion if content already exists."""
        # Check if trying to insert core memory when core is not available
        if memory_type == 'core' and not self.including_core:
            raise ValueError("Core memory is not available. Set including_core=True to use core memory.")
        
        if memory_type == 'core' and self.core is None:
            raise ValueError("Core memory is not initialized. Set including_core=True to use core memory.")
        
        if memory_type == 'core':
            # Core memory can only be updated, not inserted
            raise ValueError("Core memory cannot be inserted. Use memory_update to modify core memory content.")
        else:
            # Check if content already exists in the memory pool
            if self._content_exists(memory_type, content):
                # Return None to indicate that insertion was skipped
                return None
            
            # For semantic and episodic memories, use the existing logic
            memory_id = self._generate_memory_id()
            getattr(self, memory_type).append({memory_id: content})
            
            # Generate and store embedding for semantic and episodic memories
            if memory_type in ['semantic', 'episodic']:
                embedding = self._get_embedding(content)
                # Add embedding to matrix
                embedding_matrix = getattr(self, f"{memory_type}_embedding_matrix")
                embedding_ids = getattr(self, f"{memory_type}_embedding_ids")
                
                # Append embedding to matrix
                new_matrix = np.vstack([embedding_matrix, embedding.reshape(1, -1)])
                setattr(self, f"{memory_type}_embedding_matrix", new_matrix)
                embedding_ids.append(memory_id)
            
            return {memory_id: content}

    def memory_update(self, memory_type: str, new_content: str, memory_id: str=None):
        """Update a memory by its ID."""
        # Check if trying to update core memory when core is not available
        if memory_type == 'core' and not self.including_core:
            raise ValueError("Core memory is not available. Set including_core=True to use core memory.")
        
        if memory_type == 'core' and self.core is None:
            raise ValueError("Core memory is not initialized. Set including_core=True to use core memory.")
        
        if memory_type == 'core':
            # For core memory, replace the entire content with 512 token limit
            token_count = count_tokens(new_content)
            if token_count > 512:
                # Truncate content to fit within 512 tokens
                # We'll iteratively truncate and check until we're under the limit
                truncation_msg = " [content exceeds 512 tokens, truncated]"
                truncation_msg_tokens = count_tokens(truncation_msg)
                target_tokens = 512 - truncation_msg_tokens
                
                # Simple truncation by splitting into words and reducing
                words = new_content.split()
                truncated_content = new_content
                
                while count_tokens(truncated_content) > target_tokens and words:
                    words.pop()  # Remove last word
                    truncated_content = " ".join(words)
                
                self.core = truncated_content + truncation_msg
            else:
                self.core = new_content
            return self.core
        else:
            # For semantic and episodic memories, use the existing logic
            mem_list = getattr(self, memory_type)
            for i, mem in enumerate(mem_list):
                if memory_id in mem:
                    mem_list[i] = {memory_id: new_content}
                    break
            
            # Update embedding for semantic and episodic memories
            if memory_type in ['semantic', 'episodic']:
                embedding = self._get_embedding(new_content)
                embedding_matrix = getattr(self, f"{memory_type}_embedding_matrix")
                embedding_ids = getattr(self, f"{memory_type}_embedding_ids")
                
                # Find and update the embedding in the matrix
                idx = embedding_ids.index(memory_id)
                embedding_matrix[idx] = embedding
            
            updated_memory = {memory_id: new_content}
            return updated_memory

    def memory_delete(self, memory_type: str, memory_id: str = None):
        """Delete a memory by its ID. For core memory, clears the entire content if no memory_id is provided."""
        # Check if trying to delete core memory when core is not available
        if memory_type == 'core' and not self.including_core:
            raise ValueError("Core memory is not available. Set including_core=True to use core memory.")
        
        if memory_type == 'core' and self.core is None:
            raise ValueError("Core memory is not initialized. Set including_core=True to use core memory.")
        
        if memory_type == 'core':
            # For core memory, clear the entire content
            self.core = ""
            return
        else:
            # For semantic and episodic memories, use the existing logic
            mem_list = getattr(self, memory_type)
            for i, mem in enumerate(mem_list):
                if memory_id in mem:
                    mem_list.pop(i)
                    break
            
            # Delete corresponding embedding for semantic and episodic memories
            if memory_type in ['semantic', 'episodic']:
                embedding_matrix = getattr(self, f"{memory_type}_embedding_matrix")
                embedding_ids = getattr(self, f"{memory_type}_embedding_ids")
                
                # Find and remove the embedding from the matrix
                try:
                    idx = embedding_ids.index(memory_id)
                    # Remove row from matrix
                    new_matrix = np.delete(embedding_matrix, idx, axis=0)
                    setattr(self, f"{memory_type}_embedding_matrix", new_matrix)
                    # Remove ID from list
                    embedding_ids.pop(idx)
                except ValueError:
                    # Memory ID not found in embeddings, this shouldn't happen but handle gracefully
                    print(f"Warning: Memory ID {memory_id} not found in embedding matrix")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace and punctuation."""
        # Convert to lowercase and split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def memory_search(self, memory_type: str, query: str, top_k: int = None, min_score: float = 0.0, search_method: str = "bm25") -> List[Tuple[Dict[str, str], float]]:
        """Search for memories using BM25 or text embedding similarity. Note that the whole Core Memory is in the system prompt so you don't need to search it.
        
        Args:
            memory_type: Type of memory to search ('semantic' or 'episodic')
            query: Search query string
            top_k: Maximum number of results to return (None for all)
            min_score: Minimum score threshold (BM25 score or cosine similarity)
            search_method: Search method to use ('bm25' or 'text-embedding')
            
        Returns:
            List of tuples containing (memory_dict, score) sorted by score descending
        """
        # Core memory doesn't support searching since it's always included in context
        if memory_type == 'core':
            raise ValueError("Core memory doesn't support searching. Core memory is always included in the system prompt.")
        
        # For semantic and episodic memories only
        if memory_type not in ['semantic', 'episodic']:
            raise ValueError(f"Invalid memory_type: {memory_type}. Only 'semantic' and 'episodic' are supported for searching.")
        
        mem_list = getattr(self, memory_type)
        if not mem_list or not query.strip():
            return []
        
        if search_method == "bm25":
            return self._search_bm25(memory_type, query, top_k, min_score)
        elif search_method == "text-embedding":
            return self._search_embedding(memory_type, query, top_k, min_score)
        else:
            raise ValueError(f"Unknown search method: {search_method}. Use 'bm25' or 'text-embedding'.")

    def _search_bm25(self, memory_type: str, query: str, top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, str], float]]:
        """Search using BM25 ranking algorithm with rank_bm25 library."""
        mem_list = getattr(self, memory_type)
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Prepare documents and their metadata
        documents = []
        doc_contents = []
        
        for mem in mem_list:
            for memory_id, content in mem.items():
                documents.append((memory_id, content))
                doc_contents.append(content)
        
        if not documents:
            return []
        
        # Tokenize all documents
        tokenized_corpus = []
        for content in doc_contents:
            doc_tokens = self._tokenize(content)
            tokenized_corpus.append(doc_tokens)
        
        # Create BM25 object
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Get scores for the query
        doc_scores = bm25.get_scores(query_tokens)
        
        # Create results with scores
        results = []
        for i, (memory_id, content) in enumerate(documents):
            score = doc_scores[i]
            if score >= min_score:
                results.append(({memory_id: content}, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results

    def _search_embedding(self, memory_type: str, query: str, top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, str], float]]:
        """Search using text embedding cosine similarity with batch calculation."""
        mem_list = getattr(self, memory_type)
        embedding_matrix = getattr(self, f"{memory_type}_embedding_matrix")
        embedding_ids = getattr(self, f"{memory_type}_embedding_ids")
        
        if not mem_list or embedding_matrix.shape[0] == 0:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if np.allclose(query_embedding, 0):  # Check if embedding generation failed
            return []
        
        # Batch calculate cosine similarity for all embeddings at once
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            embedding_matrix
        )[0]  # Extract the first (and only) row
        
        results = []
        
        # Create a mapping from memory_id to content for fast lookup
        id_to_content = {}
        for mem in mem_list:
            id_to_content.update(mem)
        
        # Combine similarities with memory content
        for i, (memory_id, similarity) in enumerate(zip(embedding_ids, similarities)):
            if similarity >= min_score and memory_id in id_to_content:
                results.append(({memory_id: id_to_content[memory_id]}, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
