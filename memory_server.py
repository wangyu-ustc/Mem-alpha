#!/usr/bin/env python3
"""
Memory-powered Question Answering Server

This server processes requests containing memories and questions,
constructs system prompts from the memories, and generates responses using OpenAI models.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallel threads before multiprocessing fork
import json
import logging
import argparse
import re
import multiprocessing
from typing import List, Dict, Any, Tuple
from flask import Flask, request, jsonify
from openai import OpenAI, AzureOpenAI
from transformers import AutoTokenizer
import dotenv
from openrouter_worker import init_openrouter_worker, run_openrouter_completion

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_json_from_markdown(response_text: str) -> Dict[str, Any]:
    """
    Parse JSON content from markdown code blocks in API responses.

    Args:
        response_text: The raw response text containing JSON in markdown code blocks

    Returns:
        Parsed JSON as a dictionary, or None if parsing fails
    """
    try:
        # Extract JSON from markdown code blocks (```json or just ```)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
            return json.loads(json_content)
        else:
            logger.warning("No JSON markdown block found in response")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from markdown: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON from markdown: {str(e)}")
        return None

app = Flask(__name__)

# Model configuration - change this to switch between models
MODEL_NAME = "qwen3-32b"  # Options: "gpt-4o-mini", "gpt-4.1-mini", "qwen3-32b"

# Global variable to store server URL from command line
SERVER_URL = None

class MemoryProcessor:
    """Processes memories and generates responses using OpenAI."""

    def __init__(self, server_url=None):
        """Initialize the OpenAI client based on model configuration."""
        self.model = MODEL_NAME

        if self.model == "qwen3-32b":
            if server_url:
                base_url = server_url
            else:
                base_url = os.getenv("QWEN_URL")

            self.client = OpenAI(
                base_url=base_url,
                api_key=os.getenv("OPENROUTER_API_KEY", "EMPTY")
            )
            self.model_name = os.getenv("QWEN_MODEL_NAME")

            # Initialize tokenizer for prompt conversion
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

            logger.info(f"Initialized Qwen model client with base_url: {base_url}")
        else:
            # Azure OpenAI configuration for gpt models
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("AZURE_OPENAI_API_KEY environment variable is required for Azure models")

            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2025-01-01-preview",
                azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
            )
            self.model_name = self.model  # Use the model name as configured

            # For Azure models, we don't need tokenizer as we use the API directly
            self.tokenizer = None

            logger.info(f"Initialized Azure OpenAI client for {self.model}")


    def analyze_memory_content(self, memory_type: str, content: str) -> Dict[str, Any]:
        """Analyze memory content using LLM and return quality assessment."""
        try:
            # Create analysis prompt based on memory type
            system_prompt = self._get_analysis_prompt(memory_type)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this {memory_type} memory content:\n\n{content}"}
            ]

            max_tries = 3
            for attempt in range(max_tries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.1,  # Low temperature for consistent analysis
                        max_tokens=512,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )

                    analysis_text = response.choices[0].message.content

                    # Check if the response is in the expected JSON format
                    if self._validate_json_format(analysis_text):
                        # Parse the analysis result
                        return self._parse_analysis_result(analysis_text)
                    else:
                        logger.warning(f"Attempt {attempt + 1}: Response not in expected JSON format")
                        if attempt < max_tries - 1:
                            # Add instruction to the user message for retry
                            messages[1]["content"] += "\n\nPlease respond ONLY with a JSON code block in the exact format specified."
                            continue
                        else:
                            logger.error("Max retries reached, using fallback parsing")
                            return self._parse_analysis_result(analysis_text)

                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_tries - 1:
                        raise e

        except Exception as e:
            logger.error(f"Error in LLM-based memory analysis: {str(e)}")
            return {
                "is_valid": False,
                "issues": [f"Analysis failed: {str(e)}"],
                "score": 0.0,
                "explanation": f"Could not analyze content due to error: {str(e)}"
            }

    def _get_analysis_prompt(self, memory_type: str) -> str:
        """Get analysis prompt template for specific memory type."""
        if memory_type == 'core':
            return """You are an expert memory analyst. Analyze the quality of core memory content.

The core memory is invalid if any of the following meets:
(1) The literal content "core memory" appears in the memory such as "This is core memory ...", "The core memory has been updated ...".
(2) The core memory is apparently a placeholder such as "Here we save the summary" while not stating what the "summary" is, "Here are some rules" and not stating what the "rules" are.

Otherwise, the core memory is valid.

Respond ONLY with a JSON code block in this exact format:
```json
{
  "VALID": true/false,
  "ISSUES": [list any problems found],
  "EXPLANATION": "brief explanation of the assessment"
}
```"""

        elif memory_type == 'semantic':
            return """You are an expert memory analyst. Analyze the quality of semantic memory content.

Semantic memory should contain:
- Information or Knowledge about somebody or something
- Definitions, theories, principles, or explanations
- How-to knowledge or procedural information
- Research findings or established facts

Two other memories are Core memory (User Personalities) and Episodic memory (User Experiences). The information not suitable for these two memories should be considered as semantic memory.

Respond ONLY with a JSON code block in this exact format:
```json
{
  "VALID": true/false,
  "ISSUES": [list any problems found],
  "EXPLANATION": "brief explanation of the assessment"
}
```"""

        elif memory_type == 'episodic':
            return """You are an expert memory analyst. Analyze the quality of episodic memory content.

Episodic memory should contain:
- Experiences or events
- Clear temporal information (when it happened)
- Contextual details (what happened)

Respond ONLY with a JSON code block in this exact format:
```json
{
  "VALID": true/false,
  "ISSUES": [list any problems found],
  "EXPLANATION": "brief explanation of the assessment"
}
```"""

        else:
            raise ValueError(f"Unknown memory type: {memory_type}")

    def _validate_json_format(self, analysis_text: str) -> bool:
        """Validate that the analysis response is in the expected JSON format."""
        try:
            # Check if response contains JSON code block
            if '```json' not in analysis_text or '```' not in analysis_text:
                return False

            # Extract JSON from markdown code block
            parsed_json = parse_json_from_markdown(analysis_text)
            if parsed_json is None:
                return False

            # Check required fields
            required_fields = ['VALID', 'ISSUES', 'EXPLANATION']
            for field in required_fields:
                if field not in parsed_json:
                    logger.warning(f"Missing required field: {field}")
                    return False

            # Validate field types
            if not isinstance(parsed_json['VALID'], bool):
                logger.warning("VALID field must be a boolean")
                return False

            if not isinstance(parsed_json['ISSUES'], list):
                logger.warning("ISSUES field must be a list")
                return False

            if not isinstance(parsed_json['EXPLANATION'], str):
                logger.warning("EXPLANATION field must be a string")
                return False

            return True

        except Exception as e:
            logger.warning(f"JSON validation failed: {str(e)}")
            return False

    def _parse_analysis_result(self, analysis_text: str) -> Dict[str, Any]:
        """Parse the LLM analysis result into structured format."""
        try:
            # First try to parse JSON properly
            parsed_json = parse_json_from_markdown(analysis_text)

            if parsed_json is not None:
                # Extract fields from the JSON
                is_valid = parsed_json.get('VALID', False)
                issues = parsed_json.get('ISSUES', [])
                explanation = parsed_json.get('EXPLANATION', 'LLM-based analysis completed')

                # Ensure issues is a list
                if not isinstance(issues, list):
                    issues = [str(issues)] if issues else []

                return {
                    "is_valid": is_valid,
                    "score": 1.0 if is_valid else 0.0,
                    "issues": issues,
                    "explanation": explanation
                }
            else:
                # Fallback to simple string parsing if JSON parsing fails
                logger.warning("JSON parsing failed, using fallback string parsing")

                # Simple check for VALID field in the response
                if '"VALID": true' in analysis_text:
                    is_valid = True
                elif '"VALID": false' in analysis_text:
                    is_valid = False
                else:
                    # Fallback - assume invalid if not found
                    is_valid = False

                # Try to extract explanation if available
                explanation = "LLM-based analysis completed"
                if '"EXPLANATION":' in analysis_text:
                    try:
                        import re
                        explanation_match = re.search(r'"EXPLANATION":\s*"([^"]*)"', analysis_text)
                        if explanation_match:
                            explanation = explanation_match.group(1)
                    except Exception:
                        pass

                return {
                    "is_valid": is_valid,
                    "score": 1.0 if is_valid else 0.0,
                    "issues": [] if is_valid else ["Analysis failed"],
                    "explanation": explanation
                }

        except Exception as e:
            logger.error(f"Error parsing analysis result: {str(e)}")
            return {
                "is_valid": False,
                "score": 0.0,
                "issues": [f"Failed to parse analysis: {str(e)}"],
                "explanation": f"Could not parse analysis result: {analysis_text[:200]}..."
            }

    def _format_memory_block(self, memory_list: List[Dict[str, str]], block_name: str) -> str:
        """Format a memory block for the system prompt (for semantic and episodic memories)."""
        if not memory_list:
            return f"<{block_name}>\n(No memories stored)\n</{block_name}>"

        formatted_memories = []
        for i, memory_item in enumerate(memory_list, 1):
            if isinstance(memory_item, dict):
                # Handle dict format like {'id': 'content'} or {'content': 'text'}
                if len(memory_item) == 1:
                    # Single key-value pair, use the value
                    content = list(memory_item.values())[0]
                else:
                    # Multiple keys, look for common content keys
                    content = memory_item.get('content', str(memory_item))
            else:
                content = str(memory_item)

            formatted_memories.append(f"{i}. {content}")

        return f"<{block_name}>\n" + "\n".join(formatted_memories) + f"\n</{block_name}>"

    def _format_core_memory_block(self, core_memory: Any) -> str:
        """Format core memory block (core memory is a string or None)."""
        if not core_memory:
            return f"<core_memory>\n(No core memory stored)\n</core_memory>"

        return f"<core_memory>\n{core_memory}\n</core_memory>"

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace and punctuation."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate tokenizer."""
        import traceback

        # Convert input to string if it's not already a string
        if not isinstance(text, str):
            print(f"!!!! WARNING: Non-string input to MemoryProcessor.count_tokens: {repr(text)} (type: {type(text)})")
            print("!!!! STACK TRACE:")
            traceback.print_stack()
            print("!!!! END STACK TRACE")

            # Handle lists by joining them
            if isinstance(text, list):
                text = " ".join(str(item) for item in text)
                print(f"!!!! FIXED: Converted list to string for tokenization: {repr(text)}")
            else:
                text = str(text)
                print(f"!!!! FIXED: Converted to string for tokenization: {repr(text)}")

        try:
            if self.tokenizer is not None:
                # Use the model's tokenizer for accurate token counting
                return len(self.tokenizer.encode(text))
            else:
                # For Azure models without tokenizer, use tiktoken
                import tiktoken
                # Use GPT-4 encoding as a reasonable approximation
                encoding = tiktoken.encoding_for_model("gpt-4")
                return len(encoding.encode(text))
        except Exception as e:
            print(f"!!!! ERROR in MemoryProcessor.count_tokens when processing text: {text}")
            print(f"!!!! ERROR type: {type(e).__name__}: {e}")
            print("!!!! STACK TRACE:")
            traceback.print_stack()
            print("!!!! END STACK TRACE")
            return 0

    def search_memories(self, memory_data: Dict[str, Any], query: str, top_k: int = 20) -> Dict[str, Any]:
        """Search memories using BM25 and return top-k results for semantic and episodic memories.

        Args:
            memory_data: Dictionary containing 'core', 'semantic', and 'episodic' memories
            query: Search query string
            top_k: Number of top results to return for each memory type

        Returns:
            Dictionary with same structure as memory_data but with filtered memories
        """
        from rank_bm25 import BM25Okapi

        result = {
            'core': memory_data.get('core', None),
            'semantic': [],
            'episodic': []
        }

        # Process semantic and episodic memories
        for memory_type in ['semantic', 'episodic']:
            memories = memory_data.get(memory_type, [])
            if not memories or not query.strip():
                continue

            # Prepare documents for BM25
            documents = []
            doc_contents = []
            original_indices = []  # Track original position for episodic sorting

            for idx, mem in enumerate(memories):
                # Handle the expected structure: each mem is a dict with single key-value pair
                # where key is memory_id and value is content

                assert isinstance(mem, dict) and len(mem) == 1

                memory_id = list(mem.keys())[0]
                content = list(mem.values())[0]

                # Ensure content is a string
                if not isinstance(content, str):
                    content = str(content)

                documents.append((memory_id, content, idx))
                doc_contents.append(content)
                original_indices.append(idx)

            if not documents:
                continue

            # Tokenize query and documents
            query_tokens = self._tokenize(query)
            if not query_tokens:
                continue

            tokenized_corpus = []
            for content in doc_contents:
                doc_tokens = self._tokenize(content)
                tokenized_corpus.append(doc_tokens)

            # Perform BM25 search
            bm25 = BM25Okapi(tokenized_corpus)
            doc_scores = bm25.get_scores(query_tokens)

            # Create results with scores
            scored_results = []
            for i, (memory_id, content, orig_idx) in enumerate(documents):
                score = doc_scores[i]
                scored_results.append(({memory_id: content}, score, orig_idx))

            # Sort by score descending and take top_k
            scored_results.sort(key=lambda x: x[1], reverse=True)
            top_results = scored_results[:top_k]

            # # For episodic memory, sort by original order after retrieval
            # if memory_type == 'episodic':
            #     top_results.sort(key=lambda x: x[2])  # Sort by original index

            # Extract just the memory dictionaries
            filtered_memories = [result[0] for result in top_results]
            result[memory_type] = filtered_memories

        return result

    def construct_system_prompt(self, memory_data: Dict[str, Any], original_memory_data: Dict[str, Any] = None) -> str:
        """Construct system prompt from memory data."""
        core_memory = memory_data.get('core', None)  # core is string or None
        semantic_memories = memory_data.get('semantic', [])  # list of dicts
        episodic_memories = memory_data.get('episodic', [])  # list of dicts

        # Format memory blocks
        memory_blocks = []
        memory_structure_items = []

        # Only include core memory if it has content
        if core_memory and core_memory.strip():
            core_block = self._format_core_memory_block(core_memory)
            memory_blocks.append(core_block)
            memory_structure_items.append("- Core Memory: Fundamental facts about the user (preferences, roles, goals, etc.)")

        semantic_block = self._format_memory_block(semantic_memories, "semantic_memory")
        episodic_block = self._format_memory_block(episodic_memories, "episodic_memory")
        memory_blocks.extend([semantic_block, episodic_block])
        memory_structure_items.extend([
            "- Semantic Memory: General knowledge, factual or conceptual information",
            "- Episodic Memory: Specific personal experiences or events with time and context"
        ])

        memory_structure = "\n".join(memory_structure_items)
        memory_content = "\n\n".join(memory_blocks)

        system_prompt = f"""You are a reasoning assistant with access to structured memory. Use the memories below to provide accurate, relevant, and comprehensive responses to user queries.

MEMORY STRUCTURE:
{memory_structure}

CURRENT MEMORY STATE:

{memory_content}

INSTRUCTIONS:
- Use the memories above to inform your responses
- If information is available in memory, reference it appropriately
- If memory is insufficient to answer a question, acknowledge this clearly
- Provide helpful and contextual responses based on the available memory
- Be concise but comprehensive in your answers"""

        return system_prompt

    def generate_response(self, memory_data: Dict[str, List], question: str) -> str:
        """Generate a response for a single question using the memory data."""
        try:
            # First try with all memories, then filter if needed
            system_prompt = self.construct_system_prompt(memory_data, memory_data)

            # Count tokens and reduce memories if exceeds 32k
            token_count = self.count_tokens(system_prompt)
            if token_count > 2048 * 15:
                logger.info(f"System prompt has {token_count} tokens (>32k), filtering memories with top_k=15")
                # Filter memories using search with top_k=15
                filtered_memory_data = self.search_memories(memory_data, question, top_k=15) # using 15 is because each memory item is less than 2048 tokens.
                system_prompt = self.construct_system_prompt(filtered_memory_data, memory_data)
                token_count = self.count_tokens(system_prompt)
                logger.info(f"After filtering, system prompt has {token_count} tokens")

            # # Assert it's less than 28k after potential filtering
            # assert token_count < 28000, f"System prompt has {token_count} tokens, exceeds 28k limit even after filtering (question: {question[:50]}...)"
            # logger.debug(f"System prompt token count: {token_count}/28000")

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def process_batch(self, memories: List[Dict[str, Any]], questions: List[List[str]]) -> List[List[str]]:
        """Process a batch of memories and questions."""
        results = []

        for i, (memory_data, question_list) in enumerate(zip(memories, questions)):
            logger.info(f"Processing batch item {i+1}/{len(memories)} with {len(question_list)} questions")

            batch_results = []
            for j, question in enumerate(question_list):
                logger.info(f"  Processing question {j+1}/{len(question_list)}: {question[:50]}...")
                response = self.generate_response(memory_data, question)
                batch_results.append(response)

            results.append(batch_results)

        return results

    def agentic_search_and_respond(self, memory_data: Dict[str, Any], question: str,
                                  max_iterations: int = 5, temperature: float = 0.7,
                                  max_tokens: int = 2048) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Perform iterative memory search using function calling to find the best answer.

        Args:
            memory_data: Dictionary containing 'core', 'semantic', and 'episodic' memories
            question: The question to answer
            max_iterations: Maximum number of search iterations
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response

        Returns:
            Tuple of (final_response, search_history) where search_history contains details
            of each search iteration
        """
        search_history = []

        # Step 1: Initial retrieval using the question (like batch_process)
        initial_top_k = 2
        filtered_memory_data = self.search_memories(memory_data, question, top_k=initial_top_k)
        system_prompt = self.construct_system_prompt(filtered_memory_data, memory_data)

        # Define available functions for memory search
        memory_search_tool = {
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Search for additional relevant memories using BM25 keyword search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "enum": ["semantic", "episodic"],
                            "description": "Type of memory to search"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant memories"
                        },
                    },
                    "required": ["memory_type", "query"]
                }
            }
        }

        # Step 2: Start conversation with initial context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        # Track retrieved memory IDs to avoid duplicates
        retrieved_semantic_ids = {list(mem.keys())[0] for mem in filtered_memory_data['semantic']}
        retrieved_episodic_ids = {list(mem.keys())[0] for mem in filtered_memory_data['episodic']}

        for iteration in range(max_iterations):
            logger.info(f"Agentic search iteration {iteration + 1}/{max_iterations}")

            try:
                # Call the model with function calling capability
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=[memory_search_tool],
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

                assistant_message = response.choices[0].message

                # If no function calls, we have the final answer
                if not assistant_message.tool_calls:
                    logger.info(f"Agent provided final answer without additional searches")
                    search_history.append({
                        'iteration': iteration + 1,
                        'action': 'final_answer',
                        'message': 'Agent provided final answer'
                    })
                    return assistant_message.content, search_history

                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in assistant_message.tool_calls]
                })

                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    if tool_call.function.name == "search_memory":
                        # Parse function arguments
                        args = json.loads(tool_call.function.arguments)
                        memory_type = args.get('memory_type')
                        query = args.get('query')
                        initial_top_k = 10  # Retrieve more memories initially
                        final_top_k = 2     # Final number to return after filtering

                        logger.info(f"  Searching {memory_type} memory with query: '{query}' (initial_top_k={initial_top_k}, final_top_k={final_top_k})")

                        # Perform the search
                        if memory_type in ['semantic', 'episodic']:
                            search_results = self.search_memories(
                                {memory_type: memory_data.get(memory_type, [])},
                                query,
                                top_k=initial_top_k
                            )

                            # Filter out already retrieved memories and preserve ranking
                            new_memories = []
                            retrieved_ids = retrieved_semantic_ids if memory_type == 'semantic' else retrieved_episodic_ids

                            for mem in search_results[memory_type]:
                                mem_id = list(mem.keys())[0]
                                if mem_id not in retrieved_ids:
                                    new_memories.append(mem)

                                    # Stop when we have enough new memories
                                    if len(new_memories) >= final_top_k:
                                        break

                            # Add the selected new memories to retrieved set
                            for mem in new_memories:
                                mem_id = list(mem.keys())[0]
                                retrieved_ids.add(mem_id)

                            # Format search results for tool return
                            if new_memories:
                                formatted_results = self._format_memory_block(new_memories, f"{memory_type}_memory")
                                tool_result = f"Found {len(new_memories)} new {memory_type} memories:\n{formatted_results}"
                            else:
                                tool_result = f"No new {memory_type} memories found for query '{query}'"

                            # Record search in history
                            search_history.append({
                                'iteration': iteration + 1,
                                'memory_type': memory_type,
                                'query': query,
                                'search_method': 'bm25',
                                'results_count': len(search_results[memory_type]),
                                'new_items': len(new_memories)
                            })

                        else:
                            tool_result = f"Invalid memory type: {memory_type}"

                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })

            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                search_history.append({
                    'iteration': iteration + 1,
                    'action': 'error',
                    'message': str(e)
                })
                # Continue to next iteration or break
                if iteration == max_iterations - 1:
                    break

        # If we've reached max iterations, get final response
        logger.info(f"Reached max iterations ({max_iterations}), generating final response")

        try:
            final_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            search_history.append({
                'iteration': max_iterations + 1,
                'action': 'max_iterations_reached',
                'message': 'Generated final response after reaching max iterations'
            })

            return final_response.choices[0].message.content, search_history

        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            return f"Error generating response: {str(e)}", search_history

# Global processor and processor variables
processor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model": processor.model,
        "model_name": processor.model_name,
        "processor_available": processor is not None,
        "analysis_mode": "llm-based" if processor is not None else "rule-based"
    })

@app.route('/process', methods=['POST'])
def process_memories_and_questions():
    """
    Main endpoint to process memories and questions.

    Expected payload:
    {
        "memories": [
            {
                "core": [...],
                "semantic": [...],
                "episodic": [...]
            },
            ...
        ],
        "questions": [
            ["question1", "question2", ...],
            ...
        ]
    }

    Returns:
    {
        "result": [
            ["response1", "response2", ...],
            ...
        ],
        "status": "success"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        memories = data.get('memories', [])
        questions = data.get('questions', [])

        if not memories or not questions:
            return jsonify({"error": "Both 'memories' and 'questions' are required"}), 400

        if len(memories) != len(questions):
            return jsonify({"error": "Number of memory sets must match number of question sets"}), 400

        logger.info(f"Processing {len(memories)} memory sets with questions")

        # Process the batch
        results = processor.process_batch(memories, questions)

        return jsonify({
            "result": results,
            "status": "success",
            "processed_count": len(results)
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze_function', methods=['POST'])
def analyze_function():
    """
    Analyze function calls for memory operations and provide quality assessment.
    Uses batch processing with apply_chat_template and completions API.

    Expected payload:
    {
        "batch": [
            [
                {
                    "tool_name": "new_memory_insert" or "memory_update",
                    "tool_arguments": {
                        "memory_type": "core|semantic|episodic",
                        "content": "...",
                        "new_content": "...",  // for memory_update
                        "memory_id": "..."     // for memory_update of semantic/episodic
                    }
                }, ...
            ],
            [
                // another group of items
            ], ...
        ],
        "qwen_batch_size": 32,  // optional, defaults to 32
        "azure_batch_size": 20  // optional, defaults to 20
    }

    Returns:
    {
        "scores": [mean_score_group1, mean_score_group2, ...],
        "detailed_analyses": [[group1_analyses], [group2_analyses], ...],
        "total_processed": int,
        "total_items": int,
        "total_groups": int,
        "status": "success"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Process batch request
        return _process_function_batch(data)

    except Exception as e:
        logger.error(f"Error analyzing function call: {str(e)}")
        return jsonify({"error": str(e)}), 500


def _process_function_batch(data):
    """
    Process a batch of function call groups for analysis using proper batch inference.
    Uses apply_chat_template and completions API like the batch_process function.

    Args:
        data: Dictionary containing 'batch' list of lists and optional batch size parameters
        batch format: [[item1, item2], [item3], [item4, item5, item6], ...]

    Returns:
        JSON response with grouped batch analysis results and mean scores per group
    """
    global processor

    try:
        batch_data = data.get('batch')

        if batch_data is None:
            return jsonify({"error": "Missing 'batch' field. Expected format: {'batch': [[...], [...], ...]}"}), 400

        if not isinstance(batch_data, list):
            return jsonify({"error": "Batch data must be a list"}), 400

        if not batch_data:
            return jsonify({"error": "Empty batch data provided"}), 400

        # Validate that each element is a list (group of items)
        for i, group in enumerate(batch_data):
            if not isinstance(group, list):
                return jsonify({"error": f"Group {i} must be a list. Expected format: [[...], [...], ...]"}), 400

        # Configurable batch sizes for different models
        qwen_batch_size = data.get('qwen_batch_size', 32)
        azure_batch_size = data.get('azure_batch_size', 20)

        # Count total items across all groups
        total_items = sum(len(group) for group in batch_data)
        logger.info(f"Processing {len(batch_data)} groups with {total_items} total function calls")

        # Flatten all items for processing but keep track of group membership
        all_items = []
        group_mapping = []  # Track which group each item belongs to

        for group_idx, group in enumerate(batch_data):
            for item in group:
                all_items.append(item)
                group_mapping.append(group_idx)

        # Validate all items first and create prompts for valid ones
        valid_items = []
        item_analyses = [None] * len(all_items)  # Pre-allocate with None

        for i, item in enumerate(all_items):
            try:
                # Validate individual item structure
                tool_name = item.get('name')
                tool_arguments = item.get('arguments', {})
                tool_success = item.get('success', None)

                if not tool_name or not tool_arguments:
                    item_analyses[i] = {
                        "error": "Both 'tool_name' and 'tool_arguments' are required",
                        "is_valid": False,
                        "score": 0.0
                    }
                    continue

                # Only analyze specific memory operations
                if tool_name not in ['new_memory_insert', 'memory_update']:
                    item_analyses[i] = {
                        "error": f"Analysis not supported for tool: {tool_name}",
                        "is_valid": False,
                        "score": 0.0
                    }
                    continue

                if tool_arguments is None:
                    item_analyses[i] = {
                        "error": "Missing tool_arguments",
                        "is_valid": False,
                        "score": 0.0
                    }
                    continue

                # Basic validation
                memory_type = tool_arguments.get('memory_type', '')
                if tool_name == 'new_memory_insert':
                    content = tool_arguments.get('content', '')
                elif tool_name == 'memory_update':
                    content = tool_arguments.get('new_content', '')
                else:
                    item_analyses[i] = {
                        "error": f"Unsupported tool: {tool_name}",
                        "is_valid": False,
                        "score": 0.0
                    }
                    continue

                if not memory_type:
                    item_analyses[i] = {
                        "error": "Missing memory_type",
                        "is_valid": False,
                        "score": 0.0
                    }
                    continue

                if memory_type not in ['core', 'semantic', 'episodic']:
                    item_analyses[i] = {
                        "error": f"Unsupported memory type: {memory_type}",
                        "is_valid": False,
                        "score": 0.0
                    }
                    continue

                if not content or not content.strip():
                    item_analyses[i] = {
                        "error": "Empty content",
                        "is_valid": False,
                        "score": 0.0
                    }
                    continue

                # Item is valid, add to processing list
                valid_items.append((i, tool_name, tool_arguments, memory_type, content))

            except Exception as e:
                logger.error(f"Error validating function call {i}: {str(e)}")
                item_analyses[i] = {
                    "error": str(e),
                    "is_valid": False,
                    "score": 0.0
                }

        # Create analysis prompts for all valid items
        all_prompts = []
        analysis_indices = []  # Track where each prompt result should go in the final array

        for item_index, tool_name, tool_arguments, memory_type, content in valid_items:
            # Create analysis prompt based on memory type
            system_prompt = processor._get_analysis_prompt(memory_type)

            # Create message dictionary
            dict_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this {memory_type} memory content:\n\n{content}"}
            ]

            # Convert to prompt using tokenizer if available (for Qwen)
            if processor.tokenizer is not None:
                prompt = processor.tokenizer.apply_chat_template(
                    dict_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                all_prompts.append(prompt)
            else:
                # For Azure models, we'll use message format
                all_prompts.append(dict_messages)

            analysis_indices.append(item_index)

        # Perform batch inference if we have valid items
        llm_results = []
        if all_prompts:
            if processor.model == "qwen3-32b" and processor.tokenizer is not None:
                # For Qwen model, use completions API with converted prompts
                effective_batch_size = qwen_batch_size

                if len(all_prompts) > effective_batch_size:
                    # Process in mini-batches
                    logger.info(f"Processing {len(all_prompts)} prompts in mini-batches of {effective_batch_size}")

                    for i in range(0, len(all_prompts), effective_batch_size):
                        batch_prompts = all_prompts[i:i + effective_batch_size]
                        batch_num = (i // effective_batch_size) + 1
                        total_batches = (len(all_prompts) + effective_batch_size - 1) // effective_batch_size

                        logger.info(f"Processing mini-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")

                        try:

                            resp = processor.client.completions.create(
                                model=processor.model_name,
                                prompt=batch_prompts,
                                max_tokens=512,
                                temperature=0.1,
                                stream=False
                            )

                            # Extract results from this batch
                            batch_results = [choice.text for choice in resp.choices]
                            llm_results.extend(batch_results)

                        except Exception as e:
                            logger.error(f"Error processing mini-batch {batch_num}: {str(e)}")
                            # Add placeholder results for failed batch
                            llm_results.extend([f"Error: {str(e)}"] * len(batch_prompts))
                else:
                    # Process all at once if batch size is acceptable
                    logger.info(f"Processing {len(all_prompts)} prompts in single batch")
                    try:
                        resp = processor.client.completions.create(
                            model=processor.model_name,
                            prompt=all_prompts,
                            max_tokens=512,
                            temperature=0.1,
                            stream=False
                        )

                        # Extract results from response
                        llm_results = [choice.text for choice in resp.choices]
                    except Exception as e:
                        logger.error(f"Error processing single batch: {str(e)}")
                        llm_results = [f"Error: {str(e)}"] * len(all_prompts)
            else:
                # For Azure OpenAI models, process with mini-batching support
                effective_batch_size = azure_batch_size

                if len(all_prompts) > effective_batch_size:
                    # Process in mini-batches
                    logger.info(f"Processing {len(all_prompts)} prompts in mini-batches of {effective_batch_size} for Azure OpenAI")

                    for i in range(0, len(all_prompts), effective_batch_size):
                        batch_prompts = all_prompts[i:i + effective_batch_size]
                        batch_num = (i // effective_batch_size) + 1
                        total_batches = (len(all_prompts) + effective_batch_size - 1) // effective_batch_size

                        logger.info(f"Processing mini-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")

                        batch_results = []
                        for messages in batch_prompts:
                            try:
                                response = processor.client.chat.completions.create(
                                    model=processor.model_name,
                                    messages=messages,
                                    temperature=0.1,
                                    max_tokens=512,
                                    extra_body={
                                        "chat_template_kwargs": {"enable_thinking": False},
                                    },
                                )
                                batch_results.append(response.choices[0].message.content)
                            except Exception as e:
                                logger.error(f"Error processing individual prompt in batch {batch_num}: {str(e)}")
                                batch_results.append(f"Error: {str(e)}")

                        llm_results.extend(batch_results)
                else:
                    # Process all prompts individually if batch size is acceptable
                    logger.info(f"Processing {len(all_prompts)} prompts individually for Azure OpenAI")
                    for messages in all_prompts:
                        try:
                            response = processor.client.chat.completions.create(
                                model=processor.model_name,
                                messages=messages,
                                temperature=0.1,
                                max_tokens=512,
                                extra_body={
                                    "chat_template_kwargs": {"enable_thinking": False},
                                },
                            )
                            llm_results.append(response.choices[0].message.content)
                        except Exception as e:
                            logger.error(f"Error processing individual prompt: {str(e)}")
                            llm_results.append(f"Error: {str(e)}")

        # Process LLM results and insert them at correct positions
        total_processed = 0
        for i, (analysis_idx, llm_result) in enumerate(zip(analysis_indices, llm_results)):
            try:
                # Parse the analysis result
                if llm_result.startswith("Error:"):
                    analysis_result = {
                        "is_valid": False,
                        "issues": [llm_result],
                        "score": 0.0,
                        "explanation": "LLM analysis failed"
                    }
                else:
                    # Validate and parse JSON response
                    if processor._validate_json_format(llm_result):
                        analysis_result = processor._parse_analysis_result(llm_result)
                        total_processed += 1
                    else:
                        analysis_result = {
                            "is_valid": False,
                            "issues": ["Invalid response format"],
                            "score": 0.0,
                            "explanation": "Failed to parse LLM response"
                        }

                # Insert result at the correct position
                item_analyses[analysis_idx] = analysis_result

            except Exception as e:
                logger.error(f"Error processing LLM result {i}: {str(e)}")
                analysis_result = {
                    "is_valid": False,
                    "issues": [str(e)],
                    "score": 0.0,
                    "explanation": "Error processing LLM result"
                }
                item_analyses[analysis_idx] = analysis_result

        # Group results and calculate means
        grouped_analyses, mean_scores = _group_and_calculate_means(item_analyses, group_mapping, len(batch_data))

        logger.info(f"Completed batch processing: {total_processed} items successfully processed out of {total_items}")

        return jsonify({
            "scores": mean_scores,
            "detailed_analyses": grouped_analyses,
            "total_processed": total_processed,
            "total_items": total_items,
            "total_groups": len(batch_data),
            "llm_processed": len(llm_results),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return jsonify({"error": str(e)}), 500


def _group_and_calculate_means(item_analyses, group_mapping, num_groups):
    """
    Group individual item analyses by their group membership and calculate mean scores.

    Args:
        item_analyses: List of analysis results for all items
        group_mapping: List indicating which group each item belongs to
        num_groups: Total number of groups

    Returns:
        tuple: (grouped_analyses, mean_scores)
            - grouped_analyses: List of lists, where each sub-list contains analyses for one group
            - mean_scores: List of mean scores for each group
    """
    # Initialize grouped results
    grouped_analyses = [[] for _ in range(num_groups)]
    group_scores = [[] for _ in range(num_groups)]

    # Group analyses and collect scores
    for item_idx, analysis in enumerate(item_analyses):
        if analysis is not None:
            group_idx = group_mapping[item_idx]
            grouped_analyses[group_idx].append(analysis)

            # Extract score from analysis
            score = analysis.get('score', 0.0)
            if isinstance(score, (int, float)):
                group_scores[group_idx].append(score)
            else:
                group_scores[group_idx].append(0.0)

    # Calculate mean scores for each group
    mean_scores = []
    for scores in group_scores:
        if scores:
            mean_score = sum(scores) / len(scores)
        else:
            mean_score = 0.0
        mean_scores.append(mean_score)

    return grouped_analyses, mean_scores


def _analyze_memory_function_call(tool_name: str, tool_arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze memory function calls for quality and correctness using LLM-based analysis.

    Returns:
        Dictionary with analysis results including validity, issues, and score
    """
    global processor

    memory_type = tool_arguments.get('memory_type', '')

    # Get content based on tool type
    if tool_name == 'new_memory_insert':
        content = tool_arguments.get('content', '')
    elif tool_name == 'memory_update':
        content = tool_arguments.get('new_content', '')
    else:
        return {"is_valid": False, "issues": [f"Unsupported tool: {tool_name}"], "score": 0.0}

    # Validate basic inputs
    if not memory_type:
        return {"is_valid": False, "issues": ["Missing memory_type"], "score": 0.0}

    if not content or not content.strip():
        return {"is_valid": False, "issues": ["Empty content"], "score": 0.0}

    # Use LLM-based analysis if processor is available
    if processor is not None:
        try:
            logger.info(f"Using LLM-based analysis for {memory_type} memory")
            return processor.analyze_memory_content(memory_type, content)
        except Exception as e:
            logger.error(f"LLM analysis failed, falling back to rule-based: {str(e)}")
            # Fall back to rule-based analysis if LLM fails
    is_valid = False
    issues = "Error"
    score = 0.0

    return {
        "is_valid": is_valid,
        "issues": issues,
        "score": score,
        "explanation": f"Rule-based analysis completed with {len(issues)} issues found"
    }

@app.route("/agentic_process", methods=['POST'])
def agentic_process():
    """
    Agentic inference endpoint that takes the same input as /process but uses iterative function calling
    to search and refine memory retrieval until finding a valid answer.

    Unlike batch_process which uses simple BM25 retrieval, this endpoint:
    - Uses function calling to dynamically search memories
    - Can refine queries based on initial results
    - Iteratively searches until sufficient information is found
    - Supports both BM25 and embedding-based search methods

    Expected payload:
    {
        "memories": [
            {
                "core": "...",
                "semantic": [...],
                "episodic": [...]
            },
            ...
        ],
        "questions": [
            ["question1", "question2", ...],
            ...
        ],
        "max_iterations": 5,  // optional, defaults to 5
        "temperature": 0.7,   // optional
        "max_tokens": 2048    // optional
    }

    Returns:
    {
        "result": [
            ["response1", "response2", ...],
            ...
        ],
        "status": "success",
        "processed_count": N,
        "search_iterations": [...]  // Details of search iterations
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        memories = data.get('memories', [])
        questions = data.get('questions', [])
        max_iterations = data.get('max_iterations', 5)
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 2048)

        if not memories or not questions:
            return jsonify({"error": "Both 'memories' and 'questions' are required"}), 400

        if len(memories) != len(questions):
            return jsonify({"error": "Number of memory sets must match number of question sets"}), 400

        logger.info(f"Processing agentic inference for {len(memories)} memory sets")

        # Process each memory-question pair
        results = []
        all_search_iterations = []

        for mem_idx, (memory_data, question_list) in enumerate(zip(memories, questions)):
            logger.info(f"Processing memory set {mem_idx+1}/{len(memories)} with {len(question_list)} questions")

            batch_results = []
            batch_iterations = []

            for q_idx, question in enumerate(question_list):
                logger.info(f"  Processing question {q_idx+1}/{len(question_list)}: {question[:50]}...")

                # Perform agentic search for this question
                response, search_history = processor.agentic_search_and_respond(
                    memory_data=memory_data,
                    question=question,
                    max_iterations=max_iterations,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                batch_results.append(response)
                batch_iterations.append(search_history)

            results.append(batch_results)
            all_search_iterations.append(batch_iterations)

        logger.info(f"Successfully completed agentic processing for {len(memories)} memory sets")

        return jsonify({
            "result": results,
            "status": "success",
            "processed_count": sum(len(r) for r in results),
            "search_iterations": all_search_iterations
        })

    except Exception as e:
        logger.error(f"Error in agentic process: {str(e)}")
        return jsonify({"error": str(e)}), 500





@app.route('/batch_process', methods=['POST'])
def batch_process():
    """
    Batch inference endpoint that takes the same input as /process but converts to prompts for batch inference.
    Includes mini-batch processing to handle large batches efficiently and avoid API rate limits.

    Expected payload:
    {
        "memories": [
            {
                "core": [...],
                "semantic": [...],
                "episodic": [...]
            },
            ...
        ],
        "questions": [
            ["question1", "question2", ...],
            ...
        ],
        "max_tokens": 128,
        "temperature": 0.7,
        "enable_thinking": false
    }

    Mini-batch processing:
    - Qwen model: Batches of up to 32 prompts (configurable with qwen_batch_size)
    - Azure OpenAI: Batches of up to 20 prompts (configurable with azure_batch_size)
    - Automatically splits large requests into smaller chunks
    - Includes error handling for individual batch failures

    Returns:
    {
        "result": [
            ["response1", "response2", ...],
            ...
        ],
        "status": "success",
        "processed_count": N
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        memories = data.get('memories', [])
        questions = data.get('questions', [])
        max_tokens = data.get('max_tokens', 2048)
        temperature = data.get('temperature', 0.7)
        enable_thinking = data.get('enable_thinking', False)

        # Configurable batch sizes
        qwen_batch_size = data.get('qwen_batch_size', 32)
        azure_batch_size = data.get('azure_batch_size', 20)

        if not memories or not questions:
            return jsonify({"error": "Both 'memories' and 'questions' are required"}), 400

        if len(memories) != len(questions):
            return jsonify({"error": "Number of memory sets must match number of question sets"}), 400

        logger.info(f"Processing batch inference for {len(memories)} memory sets")

        # Create prompts for all memory/question combinations and track structure
        all_prompts = []
        structure_info = []  # Track which memory and question index each prompt belongs to

        for mem_idx, (memory_data, question_list) in enumerate(zip(memories, questions)):
            for q_idx, question in enumerate(question_list):
                # Retrieve relevant memories for this specific question with adaptive top_k
                top_k = 20
                filtered_memory_data = processor.search_memories(memory_data, question, top_k=top_k)

                # Construct system prompt with filtered memories
                system_prompt = processor.construct_system_prompt(filtered_memory_data, memory_data)

                # Count tokens and progressively reduce top_k if exceeds 30k
                token_count = processor.count_tokens(system_prompt)
                while token_count > 30000 and top_k > 1:
                    original_top_k = top_k
                    top_k -= 1
                    logger.info(f"System prompt has {token_count} tokens (>30k), reducing top_k from {original_top_k} to {top_k}")
                    filtered_memory_data = processor.search_memories(memory_data, question, top_k=top_k)
                    system_prompt = processor.construct_system_prompt(filtered_memory_data, memory_data)
                    token_count = processor.count_tokens(system_prompt)
                    logger.info(f"After reduction to top_k={top_k}, system prompt has {token_count} tokens")

                # Assert it's less than 30k after potential reduction
                assert token_count < 30000, f"System prompt has {token_count} tokens, exceeds 30k limit even after reduction to top_k={top_k} (question: {question[:50]}...)"
                logger.debug(f"System prompt token count: {token_count}/30000")

                # Create message dictionary
                dict_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]

                # Convert to prompt using tokenizer if available (for Qwen)
                if processor.tokenizer is not None:
                    prompt = processor.tokenizer.apply_chat_template(
                        dict_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=enable_thinking
                    )
                    all_prompts.append(prompt)
                else:
                    # For Azure models, we'll handle differently
                    all_prompts.append(dict_messages)

                # Track which memory and question this prompt belongs to
                structure_info.append((mem_idx, q_idx))

        # Perform batch inference
        if "qwen3-32b" in processor.model and processor.tokenizer is not None:
            # For Qwen model, use completions API with converted prompts
            # Process in mini-batches to avoid API limits
            batch_size = qwen_batch_size  # Maximum batch size for Qwen API
            all_results = []

            base_url_obj = getattr(processor.client, "base_url", "")
            base_url_str = str(base_url_obj) if base_url_obj else ""
            is_openrouter = bool(base_url_str) and "openrouter" in base_url_str.lower()

            if len(all_prompts) > batch_size:
                # Process in mini-batches
                logger.info(f"Processing {len(all_prompts)} prompts in mini-batches of {batch_size}")

                for i in range(0, len(all_prompts), batch_size):
                    batch_prompts = all_prompts[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(all_prompts) + batch_size - 1) // batch_size

                    logger.info(f"Processing mini-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")

                    try:

                        if is_openrouter:
                            openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "EMPTY")
                            cpu_total = os.cpu_count() or 1
                            max_workers = max(1, min(len(batch_prompts), max(cpu_total - 1, 1)))
                            task_args = [
                                (prompt, max_tokens, temperature) for prompt in batch_prompts
                            ]
                            ctx = multiprocessing.get_context("spawn")
                            with ctx.Pool(
                                processes=max_workers,
                                initializer=init_openrouter_worker,
                                initargs=(
                                    base_url_str,
                                    openrouter_api_key,
                                    processor.model_name,
                                ),
                            ) as pool:
                                batch_results = pool.map(run_openrouter_completion, task_args)

                        else:
                            resp = processor.client.completions.create(
                                model=processor.model_name,
                                prompt=batch_prompts,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                stream=False
                            )

                            # Extract results from this batch
                            batch_results = [choice.text for choice in resp.choices]

                        all_results.extend(batch_results)

                    except Exception as e:
                        logger.error(f"Error processing mini-batch {batch_num}: {str(e)}")
                        # Add placeholder results for failed batch
                        all_results.extend([f"Error: {str(e)}"] * len(batch_prompts))
            else:
                # Process all at once if batch size is acceptable
                logger.info(f"Processing {len(all_prompts)} prompts in single batch")
                if is_openrouter:
                    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "EMPTY")
                    cpu_total = os.cpu_count() or 1
                    max_workers = max(1, min(len(all_prompts), max(cpu_total - 1, 1)))
                    task_args = [
                        (prompt, max_tokens, temperature) for prompt in all_prompts
                    ]
                    ctx = multiprocessing.get_context("spawn")
                    with ctx.Pool(
                        processes=max_workers,
                        initializer=init_openrouter_worker,
                        initargs=(base_url_str, openrouter_api_key, processor.model_name),
                    ) as pool:
                        all_results = pool.map(run_openrouter_completion, task_args)
                else:
                    resp = processor.client.completions.create(
                        model=processor.model_name,
                        prompt=all_prompts,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )

                    # Extract results from response
                    all_results = [choice.text for choice in resp.choices]
        else:
            # For Azure OpenAI models, process with mini-batching support
            batch_size = azure_batch_size  # Conservative batch size for Azure OpenAI
            all_results = []

            if len(all_prompts) > batch_size:
                # Process in mini-batches
                logger.info(f"Processing {len(all_prompts)} prompts in mini-batches of {batch_size} for Azure OpenAI")

                for i in range(0, len(all_prompts), batch_size):
                    batch_prompts = all_prompts[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(all_prompts) + batch_size - 1) // batch_size

                    logger.info(f"Processing mini-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")

                    batch_results = []
                    for messages in batch_prompts:
                        try:
                            response = processor.client.chat.completions.create(
                                model=processor.model_name,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            batch_results.append(response.choices[0].message.content)
                        except Exception as e:
                            logger.error(f"Error processing individual prompt in batch {batch_num}: {str(e)}")
                            batch_results.append(f"Error: {str(e)}")

                    all_results.extend(batch_results)
            else:
                # Process all prompts if batch size is acceptable
                logger.info(f"Processing {len(all_prompts)} prompts individually for Azure OpenAI")
                for messages in all_prompts:
                    try:
                        response = processor.client.chat.completions.create(
                            model=processor.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        all_results.append(response.choices[0].message.content)
                    except Exception as e:
                        logger.error(f"Error processing individual prompt: {str(e)}")
                        all_results.append(f"Error: {str(e)}")

        # Reshape results to match input structure using structure_info
        # Pre-allocate results with correct sizes for each memory's question list
        results = []
        for question_list in questions:
            results.append([None] * len(question_list))  # Pre-allocate with correct size

        # Place results in correct positions
        for result_idx, (mem_idx, q_idx) in enumerate(structure_info):
            results[mem_idx][q_idx] = all_results[result_idx]

        # Assert no None values exist in results
        for mem_idx, result_list in enumerate(results):
            for q_idx, result in enumerate(result_list):
                assert result is not None, f"Missing result for memory {mem_idx}, question {q_idx}"

        logger.info(f"Successfully processed {len(all_results)} completions")

        return jsonify({
            "result": results,
            "status": "success",
            "processed_count": len(all_results)
        })

    except Exception as e:
        logger.error(f"Error in batch inference: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Memory-powered Question Answering Server')
    parser.add_argument('--server_url',
                      help='Server URL for the model API (for Qwen models)',
                      default=None)
    parser.add_argument('--port',
                      type=int,
                      default=5000,
                      help='Port to run the server on (default: 5000)')
    parser.add_argument('--host',
                      default='0.0.0.0',
                      help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--model_name',
                      help='Model name to use for API calls',
                      default=None)

    args = parser.parse_args()

    # Update MODEL_NAME if provided via command line
    if args.model_name:
        MODEL_NAME = args.model_name

    # Initialize the processor with the server URL
    processor = MemoryProcessor(server_url=args.server_url)

    # Check if required environment variables are set for Azure models
    if MODEL_NAME != "qwen3-32b" and not os.getenv("AZURE_OPENAI_API_KEY"):
        logger.error("AZURE_OPENAI_API_KEY environment variable is required for Azure models")
        exit(1)

    # Run the server
    logger.info("Starting Memory-powered Question Answering Server...")
    logger.info(f"Using model: {processor.model}")
    logger.info(f"Model name for API calls: {processor.model_name}")
    if args.server_url:
        logger.info(f"Using custom server URL: {args.server_url}")
    logger.info("Available endpoints:")
    logger.info("  GET  /health - Health check")
    logger.info("  POST /process - Process memories and questions")
    logger.info("  POST /batch_process - Batch process memories and questions")
    logger.info("  POST /agentic_process - Agentic memory search and response")
    logger.info("  POST /analyze_function - Analyze memory function calls for quality")

    # Get debug mode from environment variable, default to False for production
    app.run(host=args.host, port=args.port)
