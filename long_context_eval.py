import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer parallel threads before multiprocessing fork
import sys
import json
import time
import yaml
import dotenv
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from openai import AzureOpenAI, OpenAI
import multiprocessing
from memalpha.utils import evaluate_eurlex
from memalpha.llm_agent.metrics import evaluate_wrt_source, _extract_answer_from_response
from openrouter_worker import init_openrouter_worker, run_openrouter_completion

import aiohttp
import asyncio
from transformers import AutoTokenizer

# Load environment variables
dotenv.load_dotenv()

QWEN_URL = os.getenv("QWEN_URL")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class LongContextEvaluator:
    """
    Long Context Evaluator for various datasets including pubmed-rct and booksum.

    This evaluator now supports data source-specific prompts and formatting:
    - booksum: Uses ROUGE scoring and book summary context prompts
    - pubmed-rct: Uses classification scoring and \\boxed{label} format
    - Other datasets: Maintains existing scoring methods

    Key improvements:
    - Loads prompts from config/prompts_wrt_datasource.yaml
    - Uses appropriate system prompts for each dataset
    - Formats questions with dataset-specific query prompts
    - Enhanced answer extraction for \\boxed{} format
    """

    def __init__(self, dataset, model_name="gpt-4o-mini", without_chunks=False, force_rescore=False):
        """Initialize the evaluator with Azure OpenAI or Qwen client"""
        self.model = model_name
        # Track which dataset this evaluator instance is working on so checkpoints/results don't mix
        self.dataset_name = dataset
        self.without_chunks = without_chunks
        self.force_rescore = force_rescore
        self.token_counter = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

        # Load data source specific prompts
        try:
            with open('config/prompts_wrt_datasource.yaml', 'r') as f:
                self.prompts_wrt_datasource = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("prompts_wrt_datasource.yaml not found, using default prompts")
            self.prompts_wrt_datasource = {}

        if self.model == 'mem1':

            with open(f"./MEM1/Mem1/{dataset}_results.json", "r") as file:
                all_lines = file.read()
            all_lines = all_lines.split("\n")
            self.mem1_memories = {}
            for line in all_lines:
                if line:
                    try:
                        item = json.loads(line)
                    except:
                        from json_repair import repair_json
                        line = repair_json(line)
                        item = json.loads(line)
                    self.mem1_memories[item['index']] = item['memory']

        # Initialize MemAgent specific configurations
        if self.model == "memagent-14b" or self.model == "memagent-7b":
            # MemAgent configurations
            self.memagent_models = {
                "memagent-7b": "BytedTsinghua-SIA/RL-MemoryAgent-7B",
                "memagent-14b": "BytedTsinghua-SIA/RL-MemoryAgent-14B"
            }
            if self.model == "memagent-14b":
                self.memagent_url = "http://localhost:8000/v1"
            elif self.model == "memagent-7b":
                self.memagent_url = "http://localhost:8001/v1"
            self.memagent_api_key = "EMPTY"
            self.recurrent_chunk_size = 5000
            self.recurrent_max_new = 1024

            # Templates for MemAgent
            self.memagent_template = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

            self.memagent_template_final = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

            self.no_memory = "No previous memory"

            # Initialize tokenizers for MemAgent models
            self.memagent_tokenizers = {}
            for model_name, model_path in self.memagent_models.items():
                try:
                    self.memagent_tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    print(f"Initialized tokenizer for {model_name}")
                except Exception as e:
                    print(f"Failed to initialize tokenizer for {model_name}: {e}")
                    # Fallback to default tokenizer if specific one fails
                    self.memagent_tokenizers[model_name] = AutoTokenizer.from_pretrained("gpt2")

        # Initialize client based on model type
        elif self.model == "qwen3-32b" or self.model == "qwen3-32b-bm25" or self.model == 'mem1':
            # Qwen model configuration
            self.qwen_base_url = QWEN_URL
            self.qwen_is_openrouter = bool(self.qwen_base_url) and "openrouter" in self.qwen_base_url.lower()
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if self.qwen_is_openrouter and not self.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not found in environment variables for OpenRouter requests")

            self.client = OpenAI(
                base_url=self.qwen_base_url,
                api_key=self.openrouter_api_key if self.qwen_is_openrouter else "EMPTY",
            )
            self.model_name = os.getenv("QWEN_MODEL_NAME", "qwen3-32b")  # The actual model name for API calls

            # Initialize tokenizer for prompt conversion
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
            print("Initialized Qwen model client")

        else:
            # Azure OpenAI configuration for gpt models
            self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            if not self.api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")

            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version="2025-01-01-preview",
                azure_endpoint=azure_endpoint
            )
            self.azure_client = self.client
            # For BM25 models, use the base model name for API calls
            if self.model == "gpt-4o-mini-bm25":
                self.model_name = "gpt-4o-mini"
            else:
                self.model_name = self.model  # Use the model name as configured
            self.tokenizer = None

            print(f"Initialized Azure OpenAI client for {self.model}")

        # Load datasets
        self.train_data = None
        self.test_data = None

        # Checkpoint functionality
        self.checkpoint_dir = "results/checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Results functionality
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Existing results cache
        self.existing_results = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace and punctuation."""
        # Convert to lowercase and split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _bm25_search_chunks(self, chunks: List[str], query: str, top_k: int = 2) -> List[str]:
        """Search for top-k chunks using BM25 ranking algorithm."""
        if not chunks or not query.strip():
            return chunks[:top_k] if chunks else []

        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return chunks[:top_k] if chunks else []

        # Tokenize all chunks
        tokenized_chunks = [self._tokenize(chunk) for chunk in chunks]

        # Create BM25 object
        bm25 = BM25Okapi(tokenized_chunks)

        # Get scores for the query
        chunk_scores = bm25.get_scores(query_tokens)

        # Create list of (chunk, score) pairs
        chunk_score_pairs = list(zip(chunks, chunk_scores))

        # Sort by score descending
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k chunks
        top_chunks = [chunk for chunk, score in chunk_score_pairs[:top_k]]

        return top_chunks

    def _get_results_filename(self, dataset):
        """Get results filename based on model and dataset"""
        safe_model_name = self.model.replace(".", "_").replace("-", "_")
        chunks_suffix = "_no_chunks" if self.without_chunks else "_with_chunks"
        return os.path.join(self.results_dir, f"{safe_model_name}{chunks_suffix}_{dataset}_results.json")

    def _load_existing_results(self, dataset):
        """Load existing results if available"""
        results_file = self._get_results_filename(dataset)

        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)

                # If force rescore is enabled, don't load existing results
                if self.force_rescore:
                    print(f"Force rescore enabled: ignoring existing results from {results_file}")
                    self.existing_results = {}
                    return {}

                # Extract existing results and create question-based lookup
                existing_results = {}
                for result_type in ["train_results", "test_results"]:
                    if result_type in data:
                        for result in data[result_type]:
                            question = result.get("question", "")
                            data_source = result.get("data_source", "")
                            key = (question, data_source)
                            predicted_answer = result.get("predicted_answer", "")
                            # Only use results that have actual predicted answers
                            if predicted_answer and predicted_answer.strip():
                                existing_results[key] = predicted_answer

                self.existing_results = existing_results
                print(f"Loaded {len(existing_results)} existing results from {results_file}")
                return existing_results
            except Exception as e:
                logger.warning(f"Error loading existing results: {e}")
                return {}

        return {}

    def _get_existing_answer(self, question, data_source):
        """Get existing predicted answer for a question if available"""
        key = (question, data_source)
        return self.existing_results.get(key, None)

    def _format_question_with_data_source_prompt(self, question: str, data_source: str) -> str:
        """Format question with data source specific query prompt if available"""
        if data_source in self.prompts_wrt_datasource:
            query_prompt = self.prompts_wrt_datasource[data_source].get('query_prompt')
            if query_prompt is not None:
                return query_prompt + "\n\n" + question
        return question

    def _get_system_prompt_for_data_source(self, data_source: str) -> str:
        """Get appropriate system prompt based on data source"""
        if data_source == 'booksum' or 'sum' in data_source:
            return ("You are a helpful assistant with access to comprehensive book summaries stored in your memory. "
                   "Use the information from the provided context to answer questions about the book content. "
                   "Provide detailed and accurate responses based on the book content.")
        elif 'icl' in data_source or data_source.startswith('pubmed'):
            return ("You are a helpful assistant specialized in test-time learning classification. "
                   "Based on the provided context and examples, learn the classification patterns and "
                   "classify new content according to the established patterns. When providing classification answers, "
                   "use the format \\boxed{label} where label is a single number representing the classification category.")
        elif data_source == 'lme_train' or data_source.startswith('longmemeval_s'):
            return ("You are a helpful assistant with access to personalized information about the user. "
                   "Use the information from the provided context to answer questions in a personalized manner. "
                   "Make sure to recall and utilize the user's personal information correctly in your responses.")
        else:
            return ("You are a helpful assistant. Answer the question as accurately as possible based on "
                   "the provided context. Be concise and focus on providing the most relevant information.")

    def _get_checkpoint_filename(self, data_type="test"):
        """Get checkpoint filename based on model and data type"""
        safe_model_name = self.model.replace(".", "_").replace("-", "_")
        chunks_suffix = "_no_chunks" if self.without_chunks else "_with_chunks"
        dataset_suffix = f"_{self.dataset_name}" if getattr(self, "dataset_name", None) else ""
        return os.path.join(
            self.checkpoint_dir,
            f"{safe_model_name}{chunks_suffix}{dataset_suffix}_{data_type}_checkpoint.json"
        )

    def _save_checkpoint(self, results, data_type="test"):
        """Save current progress to checkpoint file with backup and atomic write"""
        checkpoint_data = {
            "model": self.model,
            "without_chunks": self.without_chunks,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

        checkpoint_file = self._get_checkpoint_filename(data_type)
        temp_file = checkpoint_file + ".tmp"
        backup_file = checkpoint_file + ".backup"

        try:
            # Write to temporary file first
            with open(temp_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            # Create backup of existing checkpoint if it exists
            if os.path.exists(checkpoint_file):
                if os.path.exists(backup_file):
                    os.remove(backup_file)  # Remove old backup
                os.rename(checkpoint_file, backup_file)

            # Atomically move temp file to final location
            os.rename(temp_file, checkpoint_file)

            print(f"Checkpoint saved: {len(results)} results")

        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            # Clean up temp file if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def _load_checkpoint(self, data_type="test"):
        """Load existing checkpoint if available"""
        checkpoint_file = self._get_checkpoint_filename(data_type)

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, "r") as f:
                    checkpoint_data = json.load(f)

                if (checkpoint_data.get("model") == self.model and
                    checkpoint_data.get("without_chunks") == self.without_chunks):
                    results = checkpoint_data.get("results", [])
                    print(f"Loading checkpoint: {len(results)} results")

                    # Force rescore if requested - set all scores to None
                    if self.force_rescore:
                        for result in results:
                            if result.get('predicted_answer'):  # Only rescore if we have a predicted answer
                                result['score'] = None
                        print(f"Force rescore enabled: will recompute scores for all {len(results)} results")
                    else:
                        # Check if any results need score computation
                        missing_scores = sum(1 for r in results if r.get('score') is None)
                        if missing_scores > 0:
                            print(f"Found {missing_scores} results without scores, will recompute them")

                    return results
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Corrupted checkpoint file detected: {str(e)}")

                # Try to recover from backup if it exists
                backup_file = checkpoint_file + ".backup"
                if os.path.exists(backup_file):
                    try:
                        print(f"Attempting to load backup checkpoint: {backup_file}")
                        with open(backup_file, "r") as f:
                            backup_data = json.load(f)

                        if (backup_data.get("model") == self.model and
                            backup_data.get("without_chunks") == self.without_chunks):
                            results = backup_data.get("results", [])
                            print(f"Successfully loaded backup checkpoint: {len(results)} results")

                            # Move corrupted file to .corrupted extension
                            corrupted_file = checkpoint_file + ".corrupted"
                            os.rename(checkpoint_file, corrupted_file)
                            print(f"Moved corrupted file to: {corrupted_file}")

                            return results
                    except Exception as backup_e:
                        print(f"Backup checkpoint also corrupted: {str(backup_e)}")

                # If no backup or backup also corrupted, move corrupted file and start fresh
                corrupted_file = checkpoint_file + f".corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(checkpoint_file, corrupted_file)
                print(f"Moved corrupted checkpoint to: {corrupted_file}")
                print("Starting fresh without checkpoint...")

        return []

    def _validate_checkpoint_integrity(self, checkpoint_file):
        """Validate that a checkpoint file is not corrupted"""
        if not os.path.exists(checkpoint_file):
            return False

        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)

            # Check required fields
            required_fields = ["model", "without_chunks", "results", "timestamp"]
            for field in required_fields:
                if field not in data:
                    return False

            # Check that results is a list
            if not isinstance(data["results"], list):
                return False

            # Check that each result has required fields
            for result in data["results"]:
                if not isinstance(result, dict):
                    return False
                required_result_fields = ["data_source", "question", "answer", "predicted_answer"]
                for field in required_result_fields:
                    if field not in result:
                        return False

            return True
        except Exception:
            return False

    def clean_corrupted_checkpoints(self):
        """Clean up any corrupted checkpoint files"""
        print("Checking for corrupted checkpoint files...")

        for data_type in ["train", "test"]:
            checkpoint_file = self._get_checkpoint_filename(data_type)

            if os.path.exists(checkpoint_file):
                if not self._validate_checkpoint_integrity(checkpoint_file):
                    corrupted_file = checkpoint_file + f".corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.rename(checkpoint_file, corrupted_file)
                    print(f"Moved corrupted checkpoint to: {corrupted_file}")
                else:
                    print(f"Checkpoint {checkpoint_file} is valid")

    def _recompute_missing_scores(self, results):
        """Recompute scores for results that are missing them"""
        recomputed_count = 0
        for i, result in enumerate(results):
            if result.get('score') is None:
                score = self._compute_score(
                    result['data_source'],
                    result['predicted_answer'],
                    result['answer'],
                    question=result.get('question')
                )
                result['score'] = score
                recomputed_count += 1

                # Save checkpoint periodically
                if recomputed_count % 100 == 0:
                    self._save_checkpoint(results, "test")
                    print(f"Recomputed scores for {recomputed_count} results")

        if recomputed_count > 0:
            print(f"Recomputed scores for {recomputed_count} results total")
            self._save_checkpoint(results, "test")

        return results

    def _compute_score(self, data_source, predicted_answer, gold_answer, question=None):

        if "<think>" and "</think>" in predicted_answer:
            predicted_answer = predicted_answer.split("</think>")[1].strip()
        if "<think>" in predicted_answer:
            predicted_answer = "Empty"

        """Compute evaluation score based on data source."""

        if data_source == 'booksum' or data_source == 'infbench_sum_eng_shots2':

            if not isinstance(gold_answer, list):
                gold_answer = gold_answer.split(", ")

            hit = 0
            for keyword in gold_answer:
                if keyword.lower() in predicted_answer.lower():
                    hit += 1
            return hit / len(gold_answer)

        elif data_source == 'eurlex':
            # F1 score for legal document classification

            # Extract the final answer inside \boxed{}
            extracted_content = _extract_answer_from_response(predicted_answer)
            return evaluate_eurlex([extracted_content], [gold_answer])

        elif data_source == 'friends':

            if question is not None:
                ACCURACY_PROMPT = """
                    Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
                        (1) a question (posed by one user to another user),
                        (2) a 'gold' (ground truth) answer,
                        (3) a generated answer
                    which you will score as CORRECT/WRONG.

                    The point of the question is to ask about the reason of someone saying something. The gold answer is a concise and short answer referring to the evidence happened before. For example:
                    Question: Why does Joey say: "No, no, no, no!" in Line 9?
                    Gold answer: The group reacts to Joey trying to climb over the balcony.
                    The generated answer might be trying to explaining it without referring to the specific evidience, such as "Joey is angry". In this case we should mark the answer as WRONG.

                    Thus the key is to identify whether the answer correctly identifies the evidence mentioned in the golden answer. Rephrasing is allowed, as long as it is related to the evidence.
                    All other answers that are too general, unrelated to the evidence, should be marked as WRONG.

                    Now it's time for the real question:
                    Question: {question}
                    Gold answer: {gold_answer}
                    Generated answer: {generated_answer}

                    First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
                    Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

                    Just return the label CORRECT or WRONG in a json format with the key as "label".
                    """

                response = self.azure_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": ACCURACY_PROMPT.format(question=question.split("\n\n")[-1], gold_answer=gold_answer, generated_answer=predicted_answer)}],
                    max_tokens=100,
                    temperature=0.1
                )
                label = json.loads(response.choices[0].message.content)["label"]
                return 1 if label == "CORRECT" else 0
            else:
                return 0

        elif data_source == 'wos46985':
            # WOS dataset evaluation with partial scoring for hierarchical classification
            extracted_answer = _extract_answer_from_response(predicted_answer)

            # Remove quotes and strip whitespace
            extracted_answer = extracted_answer.strip('"\'').strip()
            gold_answer_str = str(gold_answer).strip('"\'').strip()

            # STRICT pattern: must be EXACTLY "digit > digit > digit" with nothing else
            wos_strict_pattern = r'^\d+\s*>\s*\d+\s*>\s*\d+$'

            if not re.match(wos_strict_pattern, extracted_answer):
                return 0.0

            # Parse the hierarchical labels
            try:
                # Extract digits from both answers
                predicted_parts = [part.strip() for part in extracted_answer.split('>')]
                gold_parts = [part.strip() for part in gold_answer_str.split('>')]

                # Ensure both have exactly 3 parts
                if len(predicted_parts) != 3 or len(gold_parts) != 3:
                    return 0.0

                # Calculate partial score based on hierarchical matching
                # Level 1 (most general): weight 0.5
                # Level 2 (intermediate): weight 0.3
                # Level 3 (most specific): weight 0.2

                weights = [0.5, 0.3, 0.2]
                total_score = 0.0

                for i in range(3):
                    if predicted_parts[i] == gold_parts[i]:
                        total_score += weights[i]
                    else:
                        # If a higher level is wrong, lower levels shouldn't get credit
                        # This enforces the hierarchical nature
                        break

                return total_score

            except Exception as e:
                return 0.0

        elif data_source == 'pubmed-rct':
            # PUBMED dataset evaluation: MUST be ONLY a single digit
            extracted_answer = _extract_answer_from_response(predicted_answer)

            # Remove quotes and strip whitespace
            extracted_answer = extracted_answer.strip('"\'').strip()

            # STRICT pattern: must be EXACTLY a single digit with nothing else
            single_digit_pattern = r'^\d+$'

            if not re.match(single_digit_pattern, extracted_answer):
                return 0.0

            gold_num = str(gold_answer).strip('"\'').strip()

            return 1.0 if extracted_answer == gold_num else 0.0

        elif data_source == 'test_time_learning' or data_source.startswith('pubmed'):
            # Test-time learning dataset evaluation: similar to PUBMED-RCT, MUST be ONLY a single digit
            extracted_answer = _extract_answer_from_response(predicted_answer)

            # Remove quotes and strip whitespace
            extracted_answer = extracted_answer.strip('"\'').strip()

            # STRICT pattern: must be EXACTLY a single digit with nothing else
            single_digit_pattern = r'^\d+$'

            if not re.match(single_digit_pattern, extracted_answer):
                return 0.0

            gold_num = str(gold_answer).strip('"\'').strip()

            return 1.0 if extracted_answer == gold_num else 0.0

        elif data_source == 'perltqa':
            if ";" in gold_answer:
                gold_answer = gold_answer.split(";")
                total_hit = 0
                for answer in gold_answer:
                    if answer.lower().strip() in predicted_answer:
                        total_hit += 1
                return total_hit / len(gold_answer)

            else:
                return 1.0 if gold_answer.lower() in predicted_answer.lower() else 0.0

        elif data_source == 'arxiv-classification':
            # ARXIV_CLASSIFICATION dataset evaluation: MUST be ONLY a single digit
            extracted_answer = _extract_answer_from_response(predicted_answer)

            # Remove quotes and strip whitespace
            extracted_answer = extracted_answer.strip('"\'').strip()

            # STRICT pattern: must be EXACTLY a single digit with nothing else
            single_digit_pattern = r'^\d+$'

            if not re.match(single_digit_pattern, extracted_answer):
                return 0.0

            gold_num = str(gold_answer).strip('"\'').strip()

            return 1.0 if extracted_answer == gold_num else 0.0

        elif data_source == 'narrativeqa':
            # Containment score for NarrativeQA (similar to SQuAD/HotpotQA)
            if isinstance(gold_answer, list):
                answer_text = str(gold_answer[0]) if gold_answer else ""
            else:
                answer_text = gold_answer.get('text', gold_answer) if isinstance(gold_answer, dict) else str(gold_answer)

            return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

        elif data_source.startswith('longmemeval_s'):
            # LME Train and LongMemEval datasets - use LLM-based evaluation for personalized responses
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, gold_answer, predicted_answer)

            # Use the Qwen model for evaluation (similar to evaluate_agent_results.py)
            qwen_client = OpenAI(
                base_url=QWEN_URL,
                api_key="EMPTY"
            )

            response = qwen_client.chat.completions.create(
                model='qwen3-32b',
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )

            if "yes" in response.choices[0].message.content.strip().lower() and "no" not in response.choices[0].message.content.strip().lower():
                return 1.0
            else:
                return 0.0

        elif data_source in ['ruler_qa1_197K', 'ruler_qa2_421K']:
            # Accurate retrieval datasets (RULER) - use same logic as squad/hotpotqa
            if isinstance(gold_answer, list):
                answer_text = str(gold_answer[0]) if gold_answer else ""
            else:
                answer_text = gold_answer.get('text', gold_answer) if isinstance(gold_answer, dict) else str(gold_answer)

            return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

        elif data_source == 'squad' or data_source == 'hotpotqa':
            # Default: containment score for other data sources
            if isinstance(gold_answer, list):
                answer_text = str(gold_answer[0]) if gold_answer else ""
            else:
                answer_text = gold_answer.get('text', gold_answer) if isinstance(gold_answer, dict) else str(gold_answer)

            return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

        else:
            # memory agent bench
            return evaluate_wrt_source({'output': predicted_answer}, gold_answer, data_source)

    def load_datasets(self, dataset="memalpha"):
        """Load the parquet datasets"""
        print(f"Loading {dataset} datasets...")
        # Keep dataset name in sync for checkpoint/result file names
        self.dataset_name = dataset

        if dataset == "memalpha":
            self.train_data = pd.read_parquet('data/memalpha/train.parquet')
            self.test_data = pd.read_parquet('data/memalpha/test.parquet')
            print(f"Loaded {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == 'squad':
            self.train_data = pd.read_parquet('data/memalpha/train.parquet')
            self.test_data = pd.read_parquet('data/memalpha/test.parquet')
            self.train_data = self.train_data[self.train_data['data_source'] == 'squad']
            self.test_data = self.test_data[self.test_data['data_source'] == 'squad']
            print(f"Loaded Squad dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")
        
        elif dataset == 'hotpotqa':
            self.train_data = pd.read_parquet('data/memalpha/train.parquet')
            self.test_data = pd.read_parquet('data/memalpha/test.parquet')
            self.train_data = self.train_data[self.train_data['data_source'] == 'hotpotqa']
            self.test_data = self.test_data[self.test_data['data_source'] == 'hotpotqa']
            print(f"Loaded HotpotQA dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == "booksum":
            self.train_data = pd.read_parquet('data/memalpha/train.parquet')
            self.test_data = pd.read_parquet('data/memalpha/test.parquet')
            self.train_data = self.train_data[self.train_data['data_source'] == 'booksum']
            self.test_data = self.test_data[self.test_data['data_source'] == 'booksum']
            print(f"Loaded BookSum dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == 'long_range_understanding':
            self.train_data = pd.read_parquet("./data/memoryagentbench/train.parquet")
            # self.train_data = self.train_data[self.train_data['data_source'].isin(['infbench_sum_eng_shots2'])]
            self.test_data = pd.read_parquet("./data/memoryagentbench/test.parquet")
            self.test_data = self.test_data[self.test_data['data_source'].isin(['infbench_sum_eng_shots2'])]
            print(np.unique(self.test_data['data_source'].tolist(), return_counts=True))
            print(f"Loaded Long Range Understanding dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == 'accurate_retrieval':
            self.train_data = pd.read_parquet("./data/memoryagentbench/train.parquet")
            # self.train_data = self.train_data[self.train_data['data_source'].isin(['infbench_sum_eng_shots2'])]
            self.test_data = pd.read_parquet("./data/memoryagentbench/test.parquet")
            # self.test_data = self.test_data[self.test_data['data_source'].isin(['ruler_qa1_197K', 'ruler_qa2_421K', 'longmemeval_s*'])]
            self.test_data = self.test_data[self.test_data['data_source'].isin(['ruler_qa1_197K', 'ruler_qa2_421K'])]
            print(f"Loaded Accurate Retrieval dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == 'test_time_learning':
            self.train_data = pd.read_parquet("./data/memoryagentbench/train.parquet")
            self.test_data = pd.read_parquet("./data/memoryagentbench/test.parquet")
            # Filter for test-time learning datasets (same loading as accurate_retrieval)
            self.test_data = self.test_data[self.test_data['data_source'].isin(['icl_banking77_5900shot_balance', 'icl_clinic150_7050shot_balance', 'icl_nlu_8296shot_balance', 'icl_trec_coarse_6600shot_balance', 'icl_trec_fine_6400shot_balance'])]
            print(f"Loaded Test Time Learning dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == 'memoryagentbench':
            self.train_data = pd.read_parquet("./data/memoryagentbench/train.parquet")
            self.test_data = pd.read_parquet("./data/memoryagentbench/test.parquet")
            print(f"Loaded MemoryAgentBench dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == "pubmed-rct":
            self.train_data = pd.read_parquet("./data/memalpha/pubmed-rct/train.parquet")
            self.test_data = pd.read_parquet("./data/memalpha/pubmed-rct/test.parquet")
            print(f"Loaded PubMed-RCT dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == "perltqa":
            self.train_data = pd.read_parquet('data/memalpha/train.parquet')
            self.train_data = self.train_data[self.train_data['data_source'] == 'perltqa']
            self.test_data = pd.read_parquet('data/memalpha/test.parquet')
            self.test_data = self.test_data[self.test_data['data_source'] == 'perltqa']
            print(f"Loaded PerlTQA dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        elif dataset == 'longmemeval':
            # Load LongMemEval datasets - filter for longmemeval_s* data sources
            self.train_data = pd.read_parquet("./data/memoryagentbench/train.parquet")
            # self.train_data = self.train_data[self.train_data['data_source'].str.startswith('longmemeval_s*')]
            self.test_data = pd.read_parquet("./data/memoryagentbench/test.parquet")
            self.test_data = self.test_data[self.test_data['data_source'].str.startswith('longmemeval_s*')]
            print(f"Loaded LongMemEval dataset: {len(self.train_data)} train samples, {len(self.test_data)} test samples")

        else:
            raise ValueError(f"Unknown dataset: {dataset}. Supported datasets: 'memalpha', 'pubmed-rct', 'perltqa', 'longmemeval', 'memoryagentbench', 'squad', 'hotpotqa'")

    async def async_query_memagent(self, prompt, chunks, questions, model_name, tokenizer, data_source=None):
        """Asynchronously query MemAgent model with the given prompt, chunks, and questions"""
        memory = self.no_memory

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:
            # Process each chunk and update memory
            for chunk in chunks:
                msg = self.memagent_template.format(prompt=prompt, chunk=chunk, memory=memory)
                try:
                    async with session.post(
                        url=self.memagent_url + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.memagent_api_key}", "api-key": self.memagent_api_key},
                        json=dict(
                            model=self.memagent_models[model_name],
                            messages=[{"role": "user", "content": msg}],
                            temperature=0.1,
                            top_p=0.95,
                            max_tokens=self.recurrent_max_new
                        ),
                    ) as resp:
                        status = resp.status
                        if status != 200:
                            print(f"Error: {status=}, {model_name=}")
                            return [""] * len(questions), [0] * len(questions)  # 返回token数组
                        data = await resp.json()
                        memory = data["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"Error processing chunk with {model_name}: {e}")
                    return [""] * len(questions), [0] * len(questions)  # 返回token数组

            # 计算最终memory的token数
            memory_tokens = len(self.token_counter.encode(memory))

            # Generate answers for all questions using the accumulated memory
            all_answers = []
            all_memory_tokens = []  # 存储每个问题的memory token数

            for question in questions:
                msg = self.memagent_template_final.format(prompt=question, memory=memory)
                try:
                    async with session.post(
                        url=self.memagent_url + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.memagent_api_key}", "api-key": self.memagent_api_key},
                        json=dict(
                            model=self.memagent_models[model_name],
                            messages=[{"role": "user", "content": msg}],
                            temperature=0.1,
                            top_p=0.95,
                            max_tokens=self.recurrent_max_new  # 使用固定的 RECURRENT_MAX_NEW
                        ),
                    ) as resp:
                        status = resp.status
                        if status != 200:
                            print(f"Error: {status=}, {model_name=}")
                            all_answers.append("")
                            all_memory_tokens.append(0)
                        else:
                            data = await resp.json()
                            all_answers.append(data["choices"][0]["message"]["content"])
                            all_memory_tokens.append(memory_tokens)  # 每个问题都使用相同的memory
                except Exception as e:
                    print(f"Error generating answer for question with {model_name}: {e}")
                    all_answers.append("")
                    all_memory_tokens.append(0)

            return all_answers, all_memory_tokens

    def call_model(self, messages: List[Dict], max_tokens=2048, temperature=0.1):
        """Call the Azure OpenAI or Qwen model"""
        if self.model == "qwen3-32b" or self.model == "qwen3-32b-bm25":
            # For Qwen model, convert messages to prompt using tokenizer
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Check token length and truncate if necessary (30k token limit)
            max_input_tokens = 30000
            prompt_tokens = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']

            if len(prompt_tokens) > max_input_tokens:
                print(f"Warning: Input length ({len(prompt_tokens)} tokens) exceeds 30k limit, truncating to last 30k tokens")
                # Keep the last 30k tokens
                truncated_tokens = prompt_tokens[-max_input_tokens:]
                prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)

            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text
        else:
            # Azure OpenAI chat completions
            # For gpt-4o-mini and gpt-4o-mini-bm25, check and truncate if necessary (126k token limit)
            if self.model == "gpt-4o-mini" or self.model == "gpt-4o-mini-bm25":
                max_input_tokens = 126000
                # Use tiktoken to count tokens for GPT models
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")

                # Convert messages to string for token counting
                messages_str = ""
                for msg in messages:
                    messages_str += f"{msg['role']}: {msg['content']}\n"

                tokens = encoding.encode(messages_str)

                if len(tokens) > max_input_tokens:
                    print(f"Warning: Input length ({len(tokens)} tokens) exceeds 126k limit for gpt-4o-mini, truncating to last 126k tokens")
                    # Truncate the content of the last message (usually the user message with context)
                    if messages and messages[-1]['role'] == 'user':
                        user_content = messages[-1]['content']
                        user_tokens = encoding.encode(user_content)

                        # Calculate how many tokens to keep from user content
                        other_tokens = len(tokens) - len(user_tokens)
                        available_for_user = max_input_tokens - other_tokens - 1000  # Leave some buffer

                        if available_for_user > 0:
                            # Keep the last tokens of user content
                            truncated_user_tokens = user_tokens[-available_for_user:]
                            truncated_user_content = encoding.decode(truncated_user_tokens)
                            messages[-1]['content'] = truncated_user_content

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

    def call_model_batch(self, all_messages: List[List[Dict]], max_tokens=2048, temperature=0.1,
                         qwen_batch_size=1024, azure_batch_size=20, memagent_sample_data=None):
        """Call the model in batch mode for better efficiency"""

        # For MemAgent models - use the new sample-based approach
        if self.model.startswith("memagent"):
            if not memagent_sample_data:
                print("Error: MemAgent requires sample_data parameter")
                return []

            model_name = self.model
            return self.call_model_batch_memagent(memagent_sample_data, model_name)

        if not all_messages:
            return []

        # Prepare prompts based on model type
        if self.model in ["qwen3-32b"] or self.model == "qwen3-32b-bm25" or self.model == 'mem1':
            # Convert all messages to prompts for Qwen
            all_prompts = []
            max_input_tokens = 30000

            base_url_obj = getattr(self.client, "base_url", "")
            base_url_str = str(base_url_obj) if base_url_obj else ""
            is_openrouter = self.qwen_is_openrouter or ("openrouter" in base_url_str.lower())
            openrouter_api_key = self.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "EMPTY")

            for messages in tqdm(all_messages, total=len(all_messages)):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Check token length and truncate if necessary (30k token limit)
                prompt_tokens = self.tokenizer(prompt, return_tensors=None, add_special_tokens=False)['input_ids']

                if len(prompt_tokens) > max_input_tokens:
                    print(f"Warning: Input length ({len(prompt_tokens)} tokens) exceeds 30k limit, truncating to last 30k tokens")
                    # Keep the last 30k tokens
                    truncated_tokens = prompt_tokens[-max_input_tokens:]
                    prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)

                all_prompts.append(prompt)

            # Process in mini-batches to avoid API limits
            batch_size = qwen_batch_size if not is_openrouter else 32
            all_results = []

            if len(all_prompts) > batch_size:
                # Process in mini-batches
                print(f"Processing {len(all_prompts)} prompts in mini-batches of {batch_size}")

                for i in range(0, len(all_prompts), batch_size):
                    batch_prompts = all_prompts[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(all_prompts) + batch_size - 1) // batch_size

                    print(f"Processing mini-batch {batch_num}/{total_batches} ({len(batch_prompts)} prompts)")
                    if is_openrouter:
                        cpu_total = os.cpu_count() or 1
                        max_workers = max(1, min(len(batch_prompts), max(cpu_total - 1, 1)))
                        task_args = [
                            (prompt, max_tokens, temperature) for prompt in batch_prompts
                        ]
                        ctx = multiprocessing.get_context("spawn")
                        with ctx.Pool(
                            processes=max_workers,
                            initializer=init_openrouter_worker,
                            initargs=(base_url_str, openrouter_api_key, self.model_name),
                        ) as pool:
                            batch_results = pool.map(run_openrouter_completion, task_args)
                    else:
                        resp = self.client.completions.create(
                            model=self.model_name,
                            prompt=batch_prompts,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            stream=False
                        )
                        # Extract results from this batch
                        batch_results = [choice.text for choice in resp.choices]
                    all_results.extend(batch_results)
            else:
                # Process all at once if batch size is acceptable
                print(f"Processing {len(all_prompts)} prompts in single batch")
                if is_openrouter:
                    cpu_total = os.cpu_count() or 1
                    max_workers = max(1, min(len(all_prompts), max(cpu_total - 1, 1)))
                    task_args = [
                        (prompt, max_tokens, temperature) for prompt in all_prompts
                    ]
                    ctx = multiprocessing.get_context("spawn")
                    with ctx.Pool(
                        processes=max_workers,
                        initializer=init_openrouter_worker,
                        initargs=(base_url_str, openrouter_api_key, self.model_name),
                    ) as pool:
                        all_results = pool.map(run_openrouter_completion, task_args)
                else:
                    resp = self.client.completions.create(
                        model=self.model_name,
                        prompt=all_prompts,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=False
                    )

                    # Extract results from response
                    all_results = [choice.text for choice in resp.choices]

            return all_results


        else:
            # For Azure OpenAI models, process with mini-batching support
            batch_size = azure_batch_size
            all_results = []

            # For gpt-4o-mini and gpt-4o-mini-bm25, apply truncation if needed
            if self.model == "gpt-4o-mini" or self.model == "gpt-4o-mini-bm25":
                max_input_tokens = 126000
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")

                # Truncate messages if needed
                truncated_messages = []
                for messages in all_messages:
                    # Convert messages to string for token counting
                    messages_str = ""
                    for msg in messages:
                        messages_str += f"{msg['role']}: {msg['content']}\n"

                    tokens = encoding.encode(messages_str)

                    if len(tokens) > max_input_tokens:
                        print(f"Warning: Input length ({len(tokens)} tokens) exceeds 126k limit for gpt-4o-mini, truncating to last 126k tokens")
                        # Truncate the content of the last message (usually the user message with context)
                        if messages and messages[-1]['role'] == 'user':
                            user_content = messages[-1]['content']
                            user_tokens = encoding.encode(user_content)

                            # Calculate how many tokens to keep from user content
                            other_tokens = len(tokens) - len(user_tokens)
                            available_for_user = max_input_tokens - other_tokens - 1000  # Leave some buffer

                            if available_for_user > 0:
                                # Keep the last tokens of user content
                                truncated_user_tokens = user_tokens[-available_for_user:]
                                truncated_user_content = encoding.decode(truncated_user_tokens)
                                # Create a copy of messages to avoid modifying original
                                truncated_msg = messages.copy()
                                truncated_msg[-1] = messages[-1].copy()
                                truncated_msg[-1]['content'] = truncated_user_content
                                truncated_messages.append(truncated_msg)
                            else:
                                truncated_messages.append(messages)
                        else:
                            truncated_messages.append(messages)
                    else:
                        truncated_messages.append(messages)

                all_messages = truncated_messages

            if len(all_messages) > batch_size:
                # Process in mini-batches
                print(f"Processing {len(all_messages)} prompts in mini-batches of {batch_size} for Azure OpenAI")

                for i in range(0, len(all_messages), batch_size):
                    batch_messages = all_messages[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    total_batches = (len(all_messages) + batch_size - 1) // batch_size

                    print(f"Processing mini-batch {batch_num}/{total_batches} ({len(batch_messages)} prompts)")

                    batch_results = []
                    for messages in batch_messages:
                        try:
                            response = self.client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            batch_results.append(response.choices[0].message.content)
                        except Exception as e:
                            logger.error(f"Error processing individual prompt in batch {batch_num}: {str(e)}")
                            batch_results.append(f"Error: {str(e)}")

                        # Small delay to avoid rate limiting
                        time.sleep(0.1)

                    all_results.extend(batch_results)
            else:
                # Process all prompts individually if batch size is acceptable
                print(f"Processing {len(all_messages)} prompts individually for Azure OpenAI")
                for messages in all_messages:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        all_results.append(response.choices[0].message.content)
                    except Exception as e:
                        logger.error(f"Error processing individual prompt: {str(e)}")
                        all_results.append(f"Error: {str(e)}")

                    # Small delay to avoid rate limiting
                    time.sleep(0.1)

            return all_results

    def call_model_batch_memagent(self, sample_data_list, model_name, batch_size=20):
        """Call MemAgent model in batch mode with proper sample structure"""
        if not sample_data_list:
            return [], []

        print(f"Processing {len(sample_data_list)} samples with {model_name} in batch mode")

        # Get the appropriate tokenizer
        tokenizer = self.memagent_tokenizers.get(model_name)
        if not tokenizer:
            print(f"Error: Tokenizer not found for {model_name}")
            return [], []

        # Process in mini-batches
        all_results = []
        all_memory_tokens = []

        for i in range(0, len(sample_data_list), batch_size):
            batch_samples = sample_data_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(sample_data_list) + batch_size - 1) // batch_size

            print(f"Processing mini-batch {batch_num}/{total_batches} ({len(batch_samples)} samples)")

            async def process_batch():
                tasks = []
                for sample_data in batch_samples:
                    task = self.async_query_memagent(
                        sample_data['prompt'],
                        sample_data['chunks'],
                        sample_data['questions'],
                        model_name,
                        tokenizer,
                        sample_data['data_source']
                    )
                    tasks.append(task)

                return await asyncio.gather(*tasks)

            batch_results = asyncio.run(process_batch())

            # 分离答案和token数
            batch_answers = [result[0] for result in batch_results]
            batch_tokens = [result[1] for result in batch_results]

            all_results.extend(batch_answers)
            all_memory_tokens.extend(batch_tokens)

            time.sleep(0.5)

        return all_results, all_memory_tokens

    def run_test_evaluation(self, num_samples=10, dataset_name=None):
        """Run evaluation on test data"""

        results = self._load_checkpoint("test")

        # Recompute any missing scores from previous interrupted runs
        results = self._recompute_missing_scores(results)

        processed_questions = {(r['question'], r['data_source']) for r in results}

        if num_samples == -1:
            sample_data = self.test_data
        else:
            sample_data = self.test_data.sample(n=min(num_samples, len(self.test_data)))

        if 'data_source' in sample_data.columns:
            all_sources = sample_data['data_source'].values
            print(np.unique(all_sources, return_counts=True))

        print(f"Running evaluation on {len(sample_data)} test samples...")

        # print the total number of questions
        total_questions = 0
        for questions_and_answers in sample_data['questions_and_answers']:
            questions_and_answers = eval(questions_and_answers)
            total_questions += len(questions_and_answers)
        print(f"Total number of questions: {total_questions}")

        # Handle MemAgent models with sample-based approach
        if self.model in ["memagent-7b", "memagent-14b"]:
            # Collect sample data for batch processing
            sample_data_list = []
            question_metadata = []  # Store metadata for each question

            for idx, row in tqdm(sample_data.iterrows(), total=len(sample_data)):
                prompt = row['prompt']
                chunks = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks'] # list
                qa_pairs = json.loads(row['questions_and_answers']) if isinstance(row['questions_and_answers'], str) else row['questions_and_answers']
                data_source = row.get('data_source', dataset_name)

                # Check if any questions from this sample need processing
                sample_questions = []
                sample_gold_answers = []
                sample_needs_processing = False

                for qa_pair in qa_pairs:
                    question = qa_pair.get('question', '')
                    gold_answer = qa_pair.get('answer', '')

                    if question and (question, data_source) not in processed_questions:
                        existing_answer = self._get_existing_answer(question, data_source)
                        if existing_answer is None:
                            formatted_question = self._format_question_with_data_source_prompt(question, data_source)
                            sample_questions.append(formatted_question)
                            sample_gold_answers.append(gold_answer)
                            sample_needs_processing = True
                        else:
                            # Use existing answer and compute score
                            score = self._compute_score(data_source, existing_answer, gold_answer, question=question)
                            result = {
                                "data_source": data_source,
                                "question": question,
                                "answer": gold_answer,
                                "predicted_answer": existing_answer,
                                "score": score
                            }
                            results.append(result)
                            processed_questions.add((question, data_source))

                # If this sample has questions that need processing, add it to the batch
                if sample_needs_processing and sample_questions:
                    sample_data_list.append({
                        'prompt': prompt,
                        'chunks': chunks,
                        'questions': sample_questions,
                        'data_source': data_source
                    })

                    # Store metadata for each question in this sample
                    for q, gold_a in zip(sample_questions, sample_gold_answers):
                        question_metadata.append({
                            'data_source': data_source,
                            'original_question': q.split('\n')[-1] if '\n' in q else q,  # Extract original question
                            'gold_answer': gold_a
                        })

            # Process all samples in batch
            if sample_data_list:
                print(f"Processing {len(sample_data_list)} samples with {self.model.upper()}")
                sys.stdout.flush()

                sample_results, sample_memory_tokens = self.call_model_batch(
                    all_messages=None,
                    memagent_sample_data=sample_data_list,
                    max_tokens=2048,
                    temperature=0.1
                )

                # Flatten results and memory tokens, match with metadata
                all_predicted_answers = []
                all_memory_token_counts = []
                for sample_idx, (sample_answers, sample_tokens) in enumerate(zip(sample_results, sample_memory_tokens)):
                    all_predicted_answers.extend(sample_answers)
                    all_memory_token_counts.extend(sample_tokens)

                # Save results with memory token information
                for idx, (predicted_answer, metadata) in enumerate(zip(all_predicted_answers, question_metadata)):
                    memory_tokens = all_memory_token_counts[idx] if idx < len(all_memory_token_counts) else 0
                    result = {
                        "data_source": metadata['data_source'],
                        "question": metadata['original_question'],
                        "answer": metadata['gold_answer'],
                        "predicted_answer": predicted_answer,
                        "memory_tokens": memory_tokens,  # Add memory token count
                        "score": None
                    }
                    results.append(result)
                    processed_questions.add((metadata['original_question'], metadata['data_source']))

        else:
            # Original logic for non-MemAgent models
            # Collect all prompts and metadata for batch processing
            all_messages = []
            all_metadata = []

            for idx, row in tqdm(sample_data.iterrows(), total=len(sample_data)):

                prompt = row['prompt']
                chunks = json.loads(row['chunks']) if isinstance(row['chunks'], str) else row['chunks']
                qa_pairs = json.loads(row['questions_and_answers']) if isinstance(row['questions_and_answers'], str) else row['questions_and_answers']
                data_source = row.get('data_source', dataset_name)

                # Evaluate each question for this sample
                for qa_idx, qa_pair in enumerate(qa_pairs):  # Limit to first 5 questions per sample
                    question = qa_pair.get('question', '')
                    gold_answer = qa_pair.get('answer', '')

                    if question:
                        # Check if we already have results for this question
                        existing_answer = self._get_existing_answer(question, data_source)
                        if existing_answer is not None:
                            # Use existing answer and compute score
                            score = self._compute_score(data_source, existing_answer, gold_answer, question=question)
                            result = {
                                'index': idx,
                                "data_source": data_source,
                                "question": question,
                                "answer": gold_answer,
                                "predicted_answer": existing_answer,
                                "score": score
                            }
                            results.append(result)
                            processed_questions.add((question, data_source))
                        else:
                            # Need to get new prediction
                            # Prepare the prompt with data source specific formatting
                            if self.without_chunks:
                                # Without chunks mode: just ask the question directly
                                formatted_question = self._format_question_with_data_source_prompt(question, data_source)
                                full_prompt = formatted_question

                            else:
                                # Standard mode: construct the long-context prompt
                                if self.model == "qwen3-32b-bm25" or self.model == "gpt-4o-mini-bm25":
                                    # BM25 mode: use BM25 to select top 2 chunks
                                    context_parts = self._bm25_search_chunks(chunks, question, top_k=2)
                                elif self.model == 'mem1':
                                    context_parts = self.mem1_memories[idx]
                                elif data_source == 'friends':
                                    context_parts = chunks[:qa_pair['chunk_idx']]
                                else:
                                    context_parts = chunks

                                formatted_question = self._format_question_with_data_source_prompt(question, data_source)
                                if isinstance(context_parts, list):
                                    full_prompt = prompt + "\n\n" + "\n\n".join(context_parts) + "\n\n" + formatted_question
                                else:
                                    full_prompt = prompt + "\n\n" + context_parts + "\n\n" + formatted_question

                            # Get data source specific system prompt
                            system_prompt = self._get_system_prompt_for_data_source(data_source)
                            messages = [{'role': 'system', 'content': system_prompt},
                                    {"role": "user", "content": full_prompt}]
                            all_messages.append(messages)

                            # Store metadata for later processing
                            all_metadata.append({
                                'idx': idx,
                                "data_source": data_source,
                                "question": question,
                                "gold_answer": gold_answer,
                                "prompt": prompt,
                                "chunks": chunks
                            })

            # Process all prompts in batch if we have any
            if all_messages:
                print(f"Processing {len(all_messages)} prompts in batch mode")
                sys.stdout.flush()
                predicted_answers = self.call_model_batch(all_messages, max_tokens=2048, temperature=0.1)

                # First, save all raw results without computing scores
                print(f"Saving {len(predicted_answers)} raw results...")
                for predicted_answer, metadata in zip(predicted_answers, all_metadata):
                    result = {
                        'index': metadata['idx'],
                        "data_source": metadata['data_source'],
                        "question": metadata['question'],
                        "answer": metadata['gold_answer'],
                        "predicted_answer": predicted_answer,
                        "score": None  # Will be computed later
                    }

                    results.append(result)
                    processed_questions.add((metadata['question'], metadata['data_source']))

        # Save all raw results to checkpoint
        self._save_checkpoint(results, "test")
        print(f"Raw results saved: {len(results)} results")

        # Now compute scores for all results that don't have them yet
        results_needing_scores = [r for r in results if r.get('score') is None]
        if results_needing_scores:
            print(f"Computing scores for {len(results_needing_scores)} results...")
            score_count = 0
            for i, result in enumerate(results):
                if result['score'] is None:
                    score = self._compute_score(result['data_source'], result['predicted_answer'], result['answer'], question=result['question'])
                    result['score'] = score
                    score_count += 1

                    # Save checkpoint periodically during score computation
                    if score_count % 100 == 0:
                        self._save_checkpoint(results, "test")
                        print(f"Computed scores for {score_count}/{len(results_needing_scores)} results")

        # Final checkpoint save with all scores computed
        self._save_checkpoint(results, "test")
        print(f"Test evaluation completed: {len(results)} results with scores")
        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate simple evaluation metrics"""
        if not results:
            return {"error": "No results to calculate metrics"}

        # Overall metrics
        total_questions = len(results)
        valid_scores = [r['score'] for r in results if r.get('score') is not None]

        # 计算memory token数的平均值（如果存在）
        memory_tokens = [r.get('memory_tokens', 0) for r in results if r.get('memory_tokens') is not None]
        avg_memory_tokens = np.mean(memory_tokens) if memory_tokens else 0

        if not valid_scores:
            return {"error": "No valid scores found in results"}

        if len(valid_scores) < total_questions:
            print(f"Warning: {total_questions - len(valid_scores)} results missing scores")

        index_to_results = {}
        for result in results:
            if result['index'] not in index_to_results:
                index_to_results[result['index']] = []
            index_to_results[result['index']].append(result)

        index_to_results = {
            k: {
                'score': np.mean([x['score'] for x in v]),
                'source': v[0]['data_source']
            }
            for k,v in index_to_results.items()
        }

        data_sources_to_results = {}
        # for i in range(len(index_to_results)):
        for i in index_to_results:

            if self.model == 'mem1':
                index_to_results[i]['memory_length'] = len(self.tokenizer(self.mem1_memories[i]).input_ids)

            if index_to_results[i]['source'] not in data_sources_to_results:
                data_sources_to_results[index_to_results[i]['source']] = []

            data_sources_to_results[index_to_results[i]['source']].append(index_to_results[i])

        results = {
            source: {
                'score': np.mean([x['score'] for x in data_sources_to_results[source]]),
                # 'count': len(data_sources_to_results[source]),
                'memory_tokens': np.mean([x['memory_length'] for x in data_sources_to_results[source] if 'memory_length' in x])
            }
            for source in data_sources_to_results
        }

        return results

        # # Per data source metrics
        # data_sources = {}
        # for result in results:
        #     source = result['data_source']
        #     if source not in data_sources:
        #         data_sources[source] = {'scores': [], 'count': 0, 'memory_tokens': []}

        #     # Only include results with valid scores
        #     if result.get('score') is not None:
        #         data_sources[source]['scores'].append(result['score'])

        #     # 收集memory token数（如果存在）
        #     if result.get('memory_tokens') is not None:
        #         data_sources[source]['memory_tokens'].append(result['memory_tokens'])

        #     data_sources[source]['count'] += 1

        # # Calculate averages per data source
        # for source in data_sources:
        #     if data_sources[source]['scores']:
        #         data_sources[source]['avg_score'] = np.mean(data_sources[source]['scores'])
        #     else:
        #         data_sources[source]['avg_score'] = None

        #     # 计算每个数据源的平均memory token数
        #     if data_sources[source]['memory_tokens']:
        #         data_sources[source]['avg_memory_tokens'] = np.mean(data_sources[source]['memory_tokens'])
        #     else:
        #         data_sources[source]['avg_memory_tokens'] = 0

        #     del data_sources[source]['scores']  # Remove scores list to keep output clean
        #     del data_sources[source]['memory_tokens']  # Remove memory_tokens list to keep output clean

        # return {
        #     "model": self.model,
        #     "without_chunks": self.without_chunks,
        #     "total_questions": total_questions,
        #     "avg_score": avg_score,
        #     "avg_memory_tokens": avg_memory_tokens,
        #     "data_source_breakdown": data_sources
        # }

    def save_results(self, train_results, test_results, metrics, dataset):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save results
        results_data = {
            "model": self.model,
            "without_chunks": self.without_chunks,
            "dataset": dataset,
            "timestamp": timestamp,
            "train_results": train_results,
            "test_results": test_results,
            "metrics": metrics
        }

        results_file = self._get_results_filename(dataset)
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to {results_file}")

    def run_full_evaluation(self, train_samples=5, test_samples=10, dataset="memalpha"):
        """Run complete evaluation pipeline"""
        print(f"Starting evaluation with model: {self.model}")

        # Clean up any corrupted checkpoints before starting
        self.clean_corrupted_checkpoints()

        # Load datasets
        self.load_datasets(dataset)

        # Load existing results
        self._load_existing_results(dataset)

        # Run evaluations
        train_results = []

        test_results = self.run_test_evaluation(test_samples, dataset)
        metrics = {}

        # Save results
        self.save_results(train_results, test_results, metrics, dataset)

        # Calculate metrics
        metrics = self.calculate_metrics(test_results)
        metrics["dataset"] = dataset

        for source in metrics:
            print(source, metrics[source])

        # Clean up checkpoints
        for data_type in ["train", "test"]:
            checkpoint_file = self._get_checkpoint_filename(data_type)
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)

        return train_results, test_results, metrics


def main():
    """Main function to run the evaluation

    Examples:
        # Evaluate on pumed-rct dataset
        python long_context_eval.py --dataset pubmed-rct --model gpt-4o-mini --test_samples 100

        # Evaluate on booksum dataset
        python long_context_eval.py --dataset booksum --model gpt-4o-mini --test_samples 50

        # Evaluate without chunks (direct question answering)
        python long_context_eval.py --dataset pubmed-rct --without_chunks --test_samples 100

        # Force recompute all scores with updated scoring logic
        python long_context_eval.py --dataset pubmed-rct --force_rescore --test_samples 100

        # Clean corrupted checkpoint files
        python long_context_eval.py --clean_checkpoints
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run long context evaluation')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       choices=['gpt-4o-mini', 'gpt-4o-mini-bm25', 'gpt-4.1-mini', 'qwen3-32b', 'qwen3-32b-bm25', 'memagent-7b', 'memagent-14b', 'mem1'],
                       help='Model to use for evaluation')
    parser.add_argument('--without_chunks', action='store_true',
                       help='Run evaluation without chunks')
    parser.add_argument('--train_samples', type=int, default=0,
                       help='Number of training samples to evaluate (-1 for all)')
    parser.add_argument('--test_samples', type=int, default=-1,
                       help='Number of test samples to evaluate (-1 for all)')
    parser.add_argument('--dataset', type=str, default='memalpha',
                       choices=['memalpha', 'pubmed-rct', 'booksum', 'perltqa', 'long_range_understanding', 'accurate_retrieval', 'test_time_learning', "longmemeval", 'memoryagentbench', 'squad', 'hotpotqa'],
                       help='Dataset to use for evaluation')
    parser.add_argument('--force_rescore', action='store_true',
                       help='Force recomputation of all scores (keeping existing predicted answers)')
    parser.add_argument('--clean_checkpoints', action='store_true',
                       help='Clean corrupted checkpoint files and exit')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = LongContextEvaluator(
        dataset=args.dataset,
        model_name=args.model,
        without_chunks=args.without_chunks,
        force_rescore=args.force_rescore
    )


    # If just cleaning checkpoints, do that and exit
    if args.clean_checkpoints:
        print("Cleaning corrupted checkpoint files...")
        evaluator.clean_corrupted_checkpoints()
        print("Checkpoint cleaning completed.")
        return None, None, None

    print(f"Configuration: model={args.model}, without_chunks={args.without_chunks}, dataset={args.dataset}, force_rescore={args.force_rescore}")

    return evaluator.run_full_evaluation(
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        dataset=args.dataset
    )

if __name__ == "__main__":
    train_results, test_results, metrics = main()
