import json
import os
import time
from typing import Any, Dict, List
import re

import openai
from dotenv import load_dotenv
from json_repair import repair_json

# Conditional imports for Qwen3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt, FN_STOP_WORDS

# Conditional import for vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from memory import Memory
from functions import FUNCTION_IMPLS, MEMORY_TOOL_SCHEMAS, SEARCH_TOOL_SCHEMAS, get_memory_tool_schemas, get_search_tool_schemas

# Function call special tokens (from Qwen-Agent)
FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'

class MemoryAgent:
    """Event‑loop agent that lets the model call memory‑editing tools."""

    DEFAULT_GPT_MODEL = "gpt-4.1-mini"  # replace with actual model name when available
    DEFAULT_QWEN_MODEL = "Qwen/Qwen3-8B"  # replace with actual model name when available
    MAX_CONVERSATION_TURNS = 5  # Maximum number of conversation turns to include in history
    THINKING_BUDGET = 1024  # Maximum tokens for thinking process
    MAX_NEW_TOKENS = 2048  # Maximum tokens for generation
    MAX_MEMORY_ITEMS = 5

    def __init__(self, agent_config: dict = None, save_process: bool = False, out_dir: str = None, is_template: bool = False) -> None:
        load_dotenv()

        # Use agent_config or set defaults
        if agent_config is None:
            agent_config = {}

        # Extract configuration parameters
        model_name = agent_config.get('model_name', self.DEFAULT_GPT_MODEL)
        self.include_conversation_history = agent_config.get('include_conversation_history', True)
        self.enable_thinking = agent_config.get('enable_thinking', True)
        self.use_vllm = agent_config.get('vllm', False)
        # Add thinking budget configuration
        self.thinking_budget = agent_config.get('thinking_budget', self.THINKING_BUDGET)
        self.max_new_tokens = agent_config.get('max_new_tokens', self.MAX_NEW_TOKENS)

        # External model configuration
        self.infer_with_full_memory = agent_config.get('infer_with_full_memory', False)
        self.external_model_url = agent_config.get('external_model_url', None)
        self.api_key = agent_config.get('api_key', None)

        # Determine model type and set default if not provided
        self.model_name = model_name
        self.is_qwen = False if "gpt" in model_name.lower() else True
        self.is_template = is_template

        # Initialize based on model type
        if self.is_qwen and not self.is_template:
            # Check vLLM availability and configuration
            if self.use_vllm and not VLLM_AVAILABLE:
                raise ImportError("vLLM is not available. Please install vLLM or set vllm=false in config.")

            # Initialize Qwen3 model and tokenizer
            qwen_model = model_name or self.DEFAULT_QWEN_MODEL

            # Check if we're in offline mode
            import os
            tokenizer_kwargs = {}
            if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
                tokenizer_kwargs["local_files_only"] = True

            self.tokenizer = AutoTokenizer.from_pretrained(qwen_model, **tokenizer_kwargs)

            if self.use_vllm:
                # Initialize vLLM model
                self.model = LLM(model=qwen_model, dtype="bfloat16")
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=self.max_new_tokens,
                    stop_token_ids=[self.tokenizer.eos_token_id]
                )
            else:
                # Initialize Hugging Face model
                model_kwargs = {
                    "torch_dtype": 'bfloat16',
                }
                if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1":
                    model_kwargs["local_files_only"] = True

                self.model = AutoModelForCausalLM.from_pretrained(
                    qwen_model,
                    **model_kwargs
                ).cuda()
        else:
            # Initialize OpenAI client
            # openai.api_key = os.getenv("OPENAI_API_KEY")
            # self.client = openai.OpenAI()
            pass

        # Initialize external model client if configured
        self.external_client = None
        if self.infer_with_full_memory and self.external_model_url and self.api_key:
            self.external_client = openai.OpenAI(
                base_url=self.external_model_url,
                api_key=self.api_key
            )

        # Get including_core parameter from agent_config, default to False
        including_core = agent_config.get('including_core', False)
        self.memory = Memory(including_core=including_core)
        self.conversation_history: List[Dict[str, Any]] = []
        self.step = 0

        # Process saving configuration
        self.save_process = save_process and self.is_qwen  # Only save for Qwen
        self.out_dir = out_dir
        self.step_data: List[Dict[str, Any]] = []  # Store all step data

    def initialize(self, out_dir: str) -> int:
        """
        Initialize agent by loading existing state from files if available.

        Args:
            out_dir: Directory path where agent state files are stored

        Returns:
            current_step: The current step number (-1 if no existing state)
        """
        # Load existing state if available
        if os.path.exists(f"{out_dir}/agent_state.json"):
            with open(f"{out_dir}/agent_state.json", "r") as f:
                state = json.load(f)
                self.step = state.get('step', 0)
                # Only load core memory if it's available in the memory object
                if self.memory.including_core and self.memory.core is not None:
                    self.memory.core = state.get('core', [])
                self.memory.semantic = state.get('semantic', [])
                self.memory.episodic = state.get('episodic', [])
                self.conversation_history = state.get('conversation_history', [])

                # Load embedding IDs
                self.memory.semantic_embedding_ids = state.get('semantic_embedding_ids', [])
                self.memory.episodic_embedding_ids = state.get('episodic_embedding_ids', [])

                # Load embeddings from npz file if available
                embeddings_file = f"{out_dir}/embeddings.npz"
                if os.path.exists(embeddings_file):
                    import numpy as np
                    try:
                        embeddings_data = np.load(embeddings_file)
                        self.memory.semantic_embedding_matrix = embeddings_data.get('semantic_matrix', np.empty((0, 1536)))
                        self.memory.episodic_embedding_matrix = embeddings_data.get('episodic_matrix', np.empty((0, 1536)))
                        print(f"Loaded embeddings from {embeddings_file}")
                    except Exception as e:
                        print(f"Error loading embeddings from {embeddings_file}: {e}")
                        # Recalculate if loading fails
                        self._recalculate_embeddings('semantic')
                        self._recalculate_embeddings('episodic')
                        self._save_embeddings_to_npz(out_dir)
                else:
                    # Recalculate embeddings if file doesn't exist
                    print(f"Embeddings file not found, recalculating...")
                    self._recalculate_embeddings('semantic')
                    self._recalculate_embeddings('episodic')
                    self._save_embeddings_to_npz(out_dir)

        # Load existing step data for process tracking if save_process is enabled
        if self.save_process and out_dir:
            process_file = os.path.join(out_dir, "qwen_process_log.json")
            if os.path.exists(process_file):
                try:
                    with open(process_file, "r", encoding="utf-8") as f:
                        self.step_data = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    self.step_data = []

        if os.path.exists(f"{out_dir}/current_step.txt"):
            with open(f"{out_dir}/current_step.txt", "r") as f:
                current_step = int(f.read().strip())
        else:
            current_step = -1

        return current_step

    def _generate_query(self, user_msg: str) -> str:
        """
        Generate a query according to the user messages.
        """
        if self.is_qwen:
            prompt = f"""Given the following user input, generate a query to search the memory.

```user_input
{user_msg}
```

Please only return the query, no other text."""
            response, _ = self._complete_qwen([{"role": "user", "content": prompt}], functions=None)
            query = self._parse_response(response)[0]['content']
            return query

        else:
            raise NotImplementedError("Only Qwen is supported for now.")

    def save_step_data(self, step_info: Dict[str, Any]) -> None:
        """
        Save step data to JSON file for Qwen model process tracking.

        Args:
            step_info: Dictionary containing step information including messages, tool calls, etc.
        """
        if not self.save_process or not self.out_dir:
            return

        # Add step info to our collection
        self.step_data.append(step_info)

        # Ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        # Save to JSON file
        process_file = os.path.join(self.out_dir, "qwen_process_log.json")
        with open(process_file, "w", encoding="utf-8") as f:
            json.dump(self.step_data, f, indent=2, ensure_ascii=False)

    # ---------------------------------------------
    # Public API
    # ---------------------------------------------
    def chat(self, user_msg: str='', status: str='chat', verbose: bool=False, return_step_info: bool=False):
        """Single chat turn including possible tool sub‑turns.

        Args:
            user_msg: The user message to process
            status: The status of the chat ('chat', 'memorie', 'rethink')
            verbose: Whether to print verbose output
            return_step_info: Whether to return step info along with the response

        Returns:
            If return_step_info is False: str (the response content)
            If return_step_info is True: tuple (response_content, step_info_dict)
        """
        # Initialize step tracking for Qwen
        step_info = None
        if self.is_qwen and (self.save_process or return_step_info):
            step_info = {
                "step": self.step,
                "timestamp": time.time(),
                "status": status,
                "user_message": user_msg,
                "function_calls": [],
                "thoughts": []
            }

        # Check if external model inference with full memory is enabled
        if self.external_client and self.infer_with_full_memory and status == 'chat':
            return self._chat_with_external_model(user_msg, verbose, return_step_info, step_info)

        # Store user message and conversation history once (they don't change within the loop)
        user_messages = []
        if status != 'rethink':
            # Add conversation history if enabled, limited to MAX_CONVERSATION_TURNS
            if self.include_conversation_history:
                # Each turn consists of 2 messages (user + assistant), so multiply by 2
                recent_history = self.conversation_history[-(self.MAX_CONVERSATION_TURNS * 2):]
                user_messages.extend(recent_history)
            user_messages.append({"role": "user", "content": user_msg})

        # generate a query according to `user_messages`
        query = self._generate_query(user_msg)

        # Track conversation context that accumulates during this chat turn
        conversation_context = []

        while True:
            # Render system prompt with current memory state at each iteration
            messages = self.memory.render_system_prompt(status=status, query=query, max_num_of_recent_chunks=MemoryAgent.MAX_MEMORY_ITEMS) # When memorizing, system_prompt includes self.MEMORY_CONSOLIDATE_STEP memories; When answering, system_prompt includes all memories;

            # Add user messages (conversation history + current user message)
            messages.extend(user_messages)

            # Add accumulated conversation context from this chat turn
            messages.extend(conversation_context)

            if self.is_qwen:
                if status == 'memorie' or status == 'rethink':
                    functions = [tool["function"] for tool in get_memory_tool_schemas(self.memory)]
                elif status == 'chat':
                    functions = [tool["function"] for tool in get_search_tool_schemas(self.memory)]
                else:
                    raise ValueError(f"Invalid status: {status}")

            else:
                if status == 'memorie' or status == 'rethink':
                    functions = get_memory_tool_schemas(self.memory)
                elif status == 'chat':
                    functions = get_search_tool_schemas(self.memory)
                else:
                    raise ValueError(f"Invalid status: {status}")

            response, functions = self._complete(messages, functions=functions)

            if self.is_qwen:

                # Parse the response to check for function calls (Qwen format)
                assistant_messages = self._parse_response(response)

                if len(assistant_messages) > 0 and "function_call" not in assistant_messages[0]:
                    if step_info is not None:
                        step_info["thoughts"].append(assistant_messages[0]["reasoning_content"])

                # Add all assistant messages to the conversation context
                conversation_context.extend(assistant_messages)

                # Find messages with function calls and execute them
                function_call_messages = [msg for msg in assistant_messages if msg.get("function_call")]

                num_tool_calls = 0

                if function_call_messages:

                    if step_info is not None:
                        step_info["function_calls"].append([])

                    # Execute all function calls
                    for assistant_msg in function_call_messages:
                        tool_result = self._run_tool_from_function_call(assistant_msg["function_call"])

                        # Save tool call info for step tracking
                        if step_info is not None:
                            step_info["function_calls"][-1].append({
                                'function_name': assistant_msg["function_call"]["name"],
                                'arguments': assistant_msg["function_call"]["arguments"],
                                'result': tool_result
                            })

                        conversation_context.append({
                            "role": "function",
                            "name": assistant_msg["function_call"]["name"],
                            "content": json.dumps(tool_result)
                        })
                        num_tool_calls += 1

                    print(f"num_tool_calls: {num_tool_calls}")

                    # If we're memorizing, return the content from the first message
                    if status == 'memorie':
                        # if num_tool_calls == 1:
                        #     # need to continue the loop
                        #     continue
                        # else:
                        final_response = assistant_messages[0].get("content", "")
                        if step_info is not None:
                            step_info["final_response"] = final_response
                            if self.save_process:
                                self.save_step_data(step_info)

                        if return_step_info:
                            return final_response, step_info
                        else:
                            return final_response

                    # Loop again only when the agent is not memorizing
                    continue

            else:
                # OpenAI format
                assistant_msg = response.choices[0].message
                conversation_context.append(assistant_msg)

                # If LLM decided to call tool(s) we must answer each with role="tool"
                if assistant_msg.tool_calls:
                    for call in assistant_msg.tool_calls:
                        tool_result = self._run_tool(call)  # returns dict
                        conversation_context.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "name": call.function.name,
                                "content": json.dumps(tool_result),
                            }
                        )

                    if status == 'memorie':
                        final_response = assistant_msg.content
                        if return_step_info:
                            # For OpenAI, create a basic step_info structure if needed
                            if step_info is None:
                                step_info = {
                                    "step": self.step,
                                    "timestamp": time.time(),
                                    "status": status,
                                    "user_message": user_msg,
                                    "function_calls": [],
                                    "thoughts": []
                                }
                            step_info["final_response"] = final_response
                            return final_response, step_info
                        else:
                            return final_response

                    # Loop again only when the agent is not memorizing
                    continue

            # Get the main assistant message (first one for Qwen, or the single message for OpenAI)
            main_assistant_msg = assistant_messages[0] if self.is_qwen else assistant_msg

            # Only print if verbose mode is enabled (for debugging)
            if verbose:
                content = main_assistant_msg.get("content", "") if self.is_qwen else main_assistant_msg.content
                print(f"Assistant: {content}\n")

            if status != 'rethink':
                # Store the conversation turn in history if enabled
                if self.include_conversation_history:
                    content = main_assistant_msg.get("content", "") if self.is_qwen else main_assistant_msg.content
                    self.conversation_history.extend([
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": content}
                    ])

                self.step += 1

            # Return the assistant's response instead of just printing
            content = main_assistant_msg.get("content", "") if self.is_qwen else main_assistant_msg.content

            # Save step info for Qwen before returning
            if step_info is not None:
                step_info["final_response"] = content
                step_info["total_messages_after"] = len(messages)
                if self.save_process:
                    self.save_step_data(step_info)

            # For OpenAI models, create step_info if needed and return_step_info is True
            if not self.is_qwen and return_step_info and step_info is None:
                step_info = {
                    "step": self.step,
                    "timestamp": time.time(),
                    "status": status,
                    "user_message": user_msg,
                    "function_calls": [],
                    "thoughts": [],
                    "final_response": content,
                    "total_messages_after": len(messages)
                }

            if return_step_info:
                return content, step_info
            else:
                return content

    def _chat_with_external_model(self, user_msg: str, verbose: bool = False, return_step_info: bool = False, step_info: Dict[str, Any] = None):
        """
        Chat using external model with full memory context.

        Args:
            user_msg: The user message to process
            verbose: Whether to print verbose output
            return_step_info: Whether to return step info along with the response
            step_info: Step info dictionary for tracking

        Returns:
            If return_step_info is False: str (the response content)
            If return_step_info is True: tuple (response_content, step_info_dict)
        """
        try:
            # Create system prompt with full memory (use large number to get all memories)
            total_memory_count = len(self.memory.semantic) + len(self.memory.episodic)
            max_chunks = max(total_memory_count, 1000)  # Use large number to get all memories

            # Render system prompt with full memory
            messages = self.memory.render_system_prompt(
                status='chat',
                query=user_msg,  # Use user message as query
                max_num_of_recent_chunks=max_chunks
            )

            # Add conversation history if enabled
            if self.include_conversation_history:
                recent_history = self.conversation_history[-(self.MAX_CONVERSATION_TURNS * 2):]
                for history_msg in recent_history:
                    messages.append(history_msg)

            # Add current user message
            messages.append({"role": "user", "content": user_msg})

            # Call external model
            response = self.external_client.chat.completions.create(
                model="qwen3-32b",
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0.7
            )

            # Extract response content
            content = response.choices[0].message.content

            # Only print if verbose mode is enabled
            if verbose:
                print(f"Assistant: {content}\n")

            # Store conversation turn in history if enabled
            if self.include_conversation_history:
                self.conversation_history.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": content}
                ])

            # Update step counter
            self.step += 1

            # Update step info
            if step_info is not None:
                step_info["final_response"] = content
                step_info["total_messages_after"] = len(messages)
                step_info["external_model_used"] = True
                if self.save_process:
                    self.save_step_data(step_info)

            # Create step_info if needed for return
            if return_step_info and step_info is None:
                step_info = {
                    "step": self.step,
                    "timestamp": time.time(),
                    "status": "chat",
                    "user_message": user_msg,
                    "function_calls": [],
                    "thoughts": [],
                    "final_response": content,
                    "total_messages_after": len(messages),
                    "external_model_used": True
                }

            if return_step_info:
                return content, step_info
            else:
                return content

        except Exception as e:
            error_msg = f"Error calling external model: {str(e)}"
            print(f"[ERROR] {error_msg}")

            if return_step_info:
                if step_info is None:
                    step_info = {
                        "step": self.step,
                        "timestamp": time.time(),
                        "status": "chat",
                        "user_message": user_msg,
                        "function_calls": [],
                        "thoughts": [],
                        "error": error_msg,
                        "external_model_used": True
                    }
                else:
                    step_info["error"] = error_msg
                    step_info["external_model_used"] = True
                return error_msg, step_info
            else:
                return error_msg

    # ---------------------------------------------
    # Model-specific helpers
    # ---------------------------------------------
    def _complete(self, messages, functions=None):
        """Invoke LLM for reasoning - supports both OpenAI and Qwen3"""
        if self.is_qwen:
            response, functions = self._complete_qwen(messages, functions=functions)
            return response, functions
        else:
            return self._complete_openai(messages, functions=functions), None

    def _complete_openai(self, messages, functions=None):
        """OpenAI completion"""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=functions,
            tool_choice="auto",
            stream=False,
        )

    def _complete_qwen(self, messages, functions):
        """
        Qwen3-8B completion with function calling support using QwenFnCallPrompt and thinking_budget control
        """

        # Get processed text using the new method
        text = self._process_messages_to_text(messages, functions)

        # just for test
        test_messages = [
            {'role': 'user', "content": "abc"},
            {"role": "assistant", "content": "", "reasoning_content": "Okay, the user is asking for the current temperature in San Francisco and the temperature for tomorrow. Let me check the available tools.\n\nFirst, there's the get_current_temperature function. It requires the location and optionally the unit. Since the user didn't specify the unit, I'll default to celsius. The location should be \"San Francisco, State, Country\". Wait, the example format is \"City, State, Country\", but San Francisco is a city in California, USA. So the location parameter would be \"San Francisco, California, United States\".\n\nThen, for tomorrow's temperature, the user mentioned the current date is 2024-09-30, so tomorrow would be 2024-10-01. The get_temperature_date function requires location, date, and unit. Again, using the same location and default unit. I need to format the date as \"Year-Month-Day\", which is 2024-10-01.\n\nWait, the current date given is 2024-09-30. If today is September 30, then tomorrow is October 1st. So the date parameter for the second function call should be \"2024-10-01\".\n\nI should make two separate function calls: one for the current temperature and another for tomorrow's date. Let me structure the JSON for both tool calls accordingly."},
            {"role": "assistant", "content": "", "function_call": {"name": "get_current_temperature", "arguments": "{\"location\": \"San Francisco, California, United States\", \"unit\": \"celsius\"}"}},
            {"role": "assistant", "content": "", "function_call": {"name": "get_temperature_date", "arguments": "{\"location\": \"San Francisco, California, United States\", \"date\": \"2024-10-01\", \"unit\": \"celsius\"}"}},
            {"role": "function", "name": "get_current_temperature", "content": '{"temperature": 26.1, "location": "San Francisco, California, United States", "unit": "celsius"}'},
            {"role": "function", "name": "get_temperature_date", "content": '{"temperature": 25.9, "location": "San Francisco, California, United States", "date": "2024-10-01", "unit": "celsius"}'},
        ]

        if self.use_vllm:
            # Use vLLM for generation with thinking budget support
            if self.enable_thinking:
                # First generation until thinking budget
                thinking_sampling_params = SamplingParams(
                    temperature=0.7,
                    max_tokens=self.thinking_budget,
                    stop_token_ids=[self.tokenizer.eos_token_id]
                )
                outputs = self.model.generate([text], thinking_sampling_params)
                first_response = outputs[0].outputs[0].text

                # Check if the generation has already finished or thinking process is complete
                if self.tokenizer.eos_token_id not in self.tokenizer(first_response).input_ids and "</think>" not in first_response:
                    print("thinking budget is reached")
                    # Add early stopping text and continue generation
                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                    continued_text = text + first_response + early_stopping_text

                    # Second generation for the actual response
                    remaining_sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=self.max_new_tokens - self.thinking_budget - len(self.tokenizer(early_stopping_text).input_ids),
                        stop_token_ids=[self.tokenizer.eos_token_id]
                    )
                    outputs = self.model.generate([continued_text], remaining_sampling_params)
                    response = first_response + early_stopping_text + outputs[0].outputs[0].text.strip()
                else:
                    # Thinking completed within budget or generation finished
                    if "</think>" in first_response and self.tokenizer.eos_token_id not in self.tokenizer(first_response).input_ids:
                        # Continue generation after thinking
                        continued_text = text + first_response
                        remaining_sampling_params = SamplingParams(
                            temperature=0.7,
                            max_tokens=self.max_new_tokens - self.thinking_budget,
                            stop_token_ids=[self.tokenizer.eos_token_id]
                        )
                        outputs = self.model.generate([continued_text], remaining_sampling_params)
                        response = first_response + outputs[0].outputs[0].text.strip()
                    else:
                        response = first_response.strip()
            else:
                # Single generation without thinking budget
                outputs = self.model.generate([text], self.sampling_params)
                response = outputs[0].outputs[0].text.strip()
        else:
            # Use Hugging Face for generation
            # Tokenize and prepare model inputs
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                # First generation until thinking budget
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.thinking_budget,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id, # Padding token ID
                )

                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() # Portion after the length of the input sequence

                # Check if the generation has already finished (151645 is <|im_end|>)
                if 151645 not in output_ids:
                    # Check if the thinking process has finished (151668 is </think>)
                    # and prepare the second model input
                    if 151668 not in output_ids:
                        print("thinking budget is reached")
                        early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                        early_stopping_ids = self.tokenizer([early_stopping_text], return_tensors="pt", return_attention_mask=False).input_ids.to(self.model.device)
                        input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
                        early_stopping_length = early_stopping_ids.size(-1)
                    else:
                        input_ids = generated_ids
                        early_stopping_length = 0

                    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

                    # Second generation for the actual response
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=self.max_new_tokens - self.thinking_budget - early_stopping_length,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() # Get the full output after original input

            # Decode the complete response
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip() # Ignore the special tokens added by the tokenizer during the generation process and only keep the actual text content
        return response, functions

    def _process_messages_to_text(self, messages, functions):
        """
        Process messages to text format suitable for Qwen model input.
        Extracted from _complete_qwen for reuse in generation.py
        """
        # Convert messages to proper format for Qwen processing
        from qwen_agent.llm.schema import Message, ContentItem
        qwen_messages = []
        for msg in messages: # [{"role": ..., "content": ...}, ...]
            content = msg["content"]
            if isinstance(content, str):
                content = [ContentItem(text=content)]
            elif isinstance(content, list):
                # Handle mixed content types
                content_items = []
                for item in content:
                    if isinstance(item, str):
                        content_items.append(ContentItem(text=item))
                    elif isinstance(item, dict) and "text" in item: # If the list item is a dictionary and has a "text" key, then extract value
                        content_items.append(ContentItem(text=item["text"]))
                    else:
                        content_items.append(ContentItem(text=str(item)))
                content = content_items
            else:
                content = [ContentItem(text=str(content))]

            qwen_msg = Message(role=msg["role"], content=content)
            qwen_messages.append(qwen_msg)

        # Preprocess messages with function calling format
        if functions: # If tool functions are available
            processed_messages = QwenFnCallPrompt.preprocess_fncall_messages(
                messages=qwen_messages,
                functions=functions,
                lang='en',  # Using English
                parallel_function_calls=True,
                function_choice='auto'
            )
        else:
            processed_messages = qwen_messages

        # Convert back to dict format for tokenizer
        dict_messages = []
        for msg in processed_messages:
            content_text = ""
            for content_item in msg.content:
                content_text += content_item.text
            dict_messages.append({
                "role": msg.role,
                "content": content_text
            })

        # Apply chat template
        text = self.tokenizer.apply_chat_template( # Convert the list of dialogue messages into a single, suitable string as model input
            dict_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking # Determine whether to enable think prompt in the chat template
        )

        return text

    def _process_messages_to_token_ids(self, messages, functions, max_length=None):
        """
        Process messages to token IDs format suitable for Qwen model input.
        This method provides the full text-to-token-IDs pipeline for use in generation.py
        """
        # Get processed text
        text = self._process_messages_to_text(messages, functions)

        # Tokenize the text
        token_ids = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        # Apply max_length constraint if specified
        if max_length is not None and token_ids.shape[1] > max_length:
            print(f"[WARNING] PROCESSED TEXT TOO LONG, TRUNCATING: {token_ids.shape[1]} -> {max_length}")
            token_ids = token_ids[:, :max_length]

        return token_ids

    def _parse_response(self, response):
        """
        Parse the response from Qwen model to extract function calls and text content.
        Returns a list of assistant messages with either content or function_call.
        """
        function_calls, remaining_text = self._parse_function_calls_from_text(response)

        messages = []

        if function_calls:
            # Add main content message if there's remaining text
            if remaining_text.strip():
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": remaining_text.strip().strip("<think>").strip("</think>").strip()
                })

            # Add function call messages
            for func_call in function_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": func_call["name"],
                        "arguments": func_call["arguments"]
                    }
                })

        else:
            # No function calls - parse thinking and final response
            # Format: <think>...</think>\n\nfinal_response
            thinking_pattern = r'<think>(.*?)</think>\s*(.*)'
            match = re.search(thinking_pattern, response, re.DOTALL)

            if match:
                thinking_content = match.group(1).strip()
                final_content = match.group(2).strip()

                messages.append({
                    "role": "assistant",
                    "content": final_content,
                    "reasoning_content": thinking_content
                })

            else:
                # No thinking tags found, treat entire response as content
                # Remove any incomplete thinking tags
                clean_response = response.strip()
                if clean_response.startswith('<think>') and '</think>' not in clean_response:
                    # Incomplete thinking tag at start
                    clean_response = clean_response[7:].strip()  # Remove '<think>'

                messages.append({
                    "role": "assistant",
                    "content": clean_response,
                    "reasoning_content": ""
                })

        # If no content and no function calls, add an empty content message
        if not messages:
            messages.append({
                "role": "assistant",
                "content": ""
            })

        return messages

    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def _recalculate_embeddings(self, memory_type: str):
        """Recalculate embeddings for a specific memory type"""
        import numpy as np

        memory_list = getattr(self.memory, memory_type)
        if not memory_list:
            return

        print(f"Recalculating embeddings for {memory_type} memory ({len(memory_list)} items)...")

        # Reset embedding data
        setattr(self.memory, f"{memory_type}_embedding_matrix", np.empty((0, 1536)))
        setattr(self.memory, f"{memory_type}_embedding_ids", [])

        # Recalculate embeddings for each memory item
        for memory_item in memory_list:
            for memory_id, content in memory_item.items():
                embedding = self.memory._get_embedding(content)

                # Add to embedding matrix
                embedding_matrix = getattr(self.memory, f"{memory_type}_embedding_matrix")
                embedding_ids = getattr(self.memory, f"{memory_type}_embedding_ids")

                new_matrix = np.vstack([embedding_matrix, embedding.reshape(1, -1)])
                setattr(self.memory, f"{memory_type}_embedding_matrix", new_matrix)
                embedding_ids.append(memory_id)

        print(f"Finished recalculating {memory_type} embeddings")

    def _save_embeddings_to_npz(self, out_dir: str):
        """Save current embeddings to npz file"""
        import numpy as np

        embeddings_file = f"{out_dir}/embeddings.npz"
        np.savez_compressed(embeddings_file,
                          semantic_matrix=self.memory.semantic_embedding_matrix,
                          episodic_matrix=self.memory.episodic_embedding_matrix)

        print(f"Saved embeddings to {embeddings_file}")

        # Also update the embedding IDs in the JSON state file
        import json
        state_file = f"{out_dir}/agent_state.json"
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)

            # Update embedding IDs
            state['semantic_embedding_ids'] = self.memory.semantic_embedding_ids
            state['episodic_embedding_ids'] = self.memory.episodic_embedding_ids

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

    def _parse_function_calls_from_text(self, text: str):
        """
        Parse function calls from raw model output text.
        Returns (function_calls, remaining_text).
        Adapted from raw_token_function_calling.py
        """
        function_calls = []

        # Find all function call patterns
        pattern = f'{re.escape(FN_NAME)}:\\s*([^\\n]+)\\s*{re.escape(FN_ARGS)}:\\s*([^✿]+?)(?={re.escape(FN_RESULT)}|{re.escape(FN_EXIT)}|{re.escape(FN_NAME)}|$)'

        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            func_name = match.group(1).strip()
            func_args = match.group(2).strip()

            # Clean up arguments
            func_args = self._remove_trailing_comment_of_fn_args(func_args)

            function_calls.append({
                'name': func_name,
                'arguments': func_args
            })

        # Extract text before first function call
        first_fn_pos = text.find(FN_NAME)
        if first_fn_pos >= 0:
            remaining_text = text[:first_fn_pos].strip()
        else:
            remaining_text = text.strip()

        # Remove incomplete special tokens
        remaining_text = self._remove_incomplete_special_tokens(remaining_text)

        return function_calls, remaining_text

    def _remove_incomplete_special_tokens(self, text: str) -> str:
        """Remove incomplete special tokens from the end of text."""
        special_tokens = (FN_NAME, FN_ARGS, FN_RESULT, FN_EXIT)
        text = text.rstrip()

        if text.endswith(special_tokens):
            for s in special_tokens:
                if text.endswith(s):
                    text = text[:-len(s)]
                    break
        else:
            trail_start = text.rfind('✿')
            if trail_start >= 0:
                trail_token = text[trail_start:]
                for s in special_tokens:
                    if s.startswith(trail_token):
                        text = text[:trail_start]
                        break

        text = text.lstrip('\n').rstrip()
        return text

    def _remove_trailing_comment_of_fn_args(self, fn_args: str) -> str:
        """Remove trailing comments from function arguments."""
        fn_args = fn_args.strip()

        if fn_args.startswith('{'):
            k = fn_args.rfind('}')
            if k > 0:
                fn_args = fn_args[:k + 1]

        if fn_args.startswith('```'):
            k = fn_args.rfind('\n```')
            if k > 0:
                fn_args = fn_args[:k + 4]

        return fn_args

    def _extract_json_object(self, text):
        """Extract a complete JSON object from text starting with {"""
        if not text.startswith('{'):
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[:i+1]

        return None

    def _fix_json_issues(self, json_str):
        """Fix common JSON formatting issues"""
        # Fix unquoted values that look like identifiers (alphanumeric with possible special chars)
        # Pattern: "key": unquoted_value -> "key": "unquoted_value"
        # But avoid numbers, booleans, and null
        def fix_unquoted_value(match):
            key = match.group(1)
            value = match.group(2)

            # Don't quote if it's already a number, boolean, or null
            if (value.isdigit() or
                value in ['true', 'false', 'null'] or
                (value.startswith('-') and value[1:].replace('.', '').isdigit())):
                return match.group(0)

            # Quote the value
            return f'"{key}": "{value}"'

        # Pattern to match unquoted values after colons
        unquoted_pattern = r'"([^"]+)":\s*([a-zA-Z0-9_\-\.]+)(?=\s*[,\}])'
        fixed_json = re.sub(unquoted_pattern, fix_unquoted_value, json_str)

        return fixed_json

    # ---------------------------------------------
    # Tool plumbing
    # ---------------------------------------------
    def _run_tool(self, call):
        """Run tool from OpenAI tool call format"""
        name = call.function.name
        try:
            arguments: Dict[str, Any] = json.loads(call.function.arguments)
            result = FUNCTION_IMPLS[name](self.memory, arguments)
            return f"[tool {name} executed successfully] → {result}"
        except json.JSONDecodeError as e:
            return f"[tool {name} error] Invalid JSON arguments: {str(e)}"
        except KeyError as e:
            return f"[tool {name} error] Missing required argument: {str(e)}"
        except ValueError as e:
            return f"[tool {name} error] Invalid argument value: {str(e)}"
        except Exception as e:
            return f"[tool {name} error] Unexpected error: {str(e)}"

    def _run_tool_from_function_call(self, function_call, memory=None, return_arguments=False):
        """Run tool from Qwen function call format"""
        if memory is None:
            memory = self.memory

        success = False
        name = function_call["name"]
        arguments = None
        try:
            if isinstance(function_call["arguments"], str):
                # Repair JSON before parsing
                repaired_json = repair_json(function_call["arguments"])
                arguments = json.loads(repaired_json)

                # Debug: Check if parsed arguments contain numeric content
                if name in ["new_memory_insert", "memory_update"]:
                    content_key = "content" if name == "new_memory_insert" else "new_content"
                    if content_key in arguments and not isinstance(arguments[content_key], str):
                        print(f"!!!! WARNING: {name} parsed JSON with non-string {content_key}: {repr(arguments[content_key])} (type: {type(arguments[content_key])})")
                        print(f"!!!! Original function_call arguments: {function_call['arguments']}")
                        print(f"!!!! Repaired JSON: {repaired_json}")

            else:
                arguments = function_call["arguments"]
                # Debug: Check if direct arguments contain numeric content
                if name in ["new_memory_insert", "memory_update"]:
                    content_key = "content" if name == "new_memory_insert" else "new_content"
                    if content_key in arguments and not isinstance(arguments[content_key], str):
                        print(f"!!!! WARNING: {name} direct arguments with non-string {content_key}: {repr(arguments[content_key])} (type: {type(arguments[content_key])})")
                        print(f"!!!! Direct arguments: {arguments}")

            result = FUNCTION_IMPLS[name](memory, arguments)
            result_str = f"[tool {name} executed successfully] → {result}"
        except json.JSONDecodeError as e:
            result_str = f"[tool {name} error] Invalid JSON arguments: {str(e)}"
        except KeyError as e:
            result_str = f"[tool {name} error] Missing required argument: {str(e)}"
        except ValueError as e:
            result_str = f"[tool {name} error] Invalid argument value: {str(e)}"
        except Exception as e:
            result_str = f"[tool {name} error] Unexpected error: {str(e)}"

        if return_arguments:
            return name, arguments, result_str
        else:
            return result_str

    @staticmethod
    def process_text_with_qwen_pipeline(text: str, tokenizer, functions=None, status='memorie', enable_thinking=True, max_length=None, memory=None, device=None, max_num_of_recent_chunks=None, return_text=False):
        """
        Static method to process text using Qwen pipeline without requiring a full MemoryAgent instance.
        This is useful for external modules like generation.py
        """
        # Create messages starting with memory system prompt if available
        messages = []

        if memory is not None:
            # Use memory system prompt similar to agent.py chat() method
            # For memory operations, we use status='memorie' and generate a simple query
            query = text[:100] + "..." if len(text) > 100 else text  # Simple query from text
            max_num_of_recent_chunks = max_num_of_recent_chunks if max_num_of_recent_chunks is not None else MemoryAgent.MAX_MEMORY_ITEMS
            system_messages = memory.render_system_prompt(status=status, query=query, max_num_of_recent_chunks=max_num_of_recent_chunks)
            messages.extend(system_messages)
        else:
            print(f"[DEBUG] No memory provided, using text-only processing")

        # Add user message
        messages.append({"role": "user", "content": text})

        # Convert messages to proper format for Qwen processing
        from qwen_agent.llm.schema import Message, ContentItem
        qwen_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [ContentItem(text=content)]
            elif isinstance(content, list):
                content_items = []
                for item in content:
                    if isinstance(item, str):
                        content_items.append(ContentItem(text=item))
                    elif isinstance(item, dict) and "text" in item:
                        content_items.append(ContentItem(text=item["text"]))
                    else:
                        content_items.append(ContentItem(text=str(item)))
                content = content_items
            else:
                content = [ContentItem(text=str(content))]

            qwen_msg = Message(role=msg["role"], content=content)
            qwen_messages.append(qwen_msg)

        # Preprocess messages with function calling format if functions are provided
        if functions:
            processed_messages = QwenFnCallPrompt.preprocess_fncall_messages(
                messages=qwen_messages,
                functions=functions,
                lang='en',
                parallel_function_calls=True,
                function_choice='auto'
            )
        else:
            processed_messages = qwen_messages

        # Convert back to dict format for tokenizer
        dict_messages = []
        for msg in processed_messages:
            content_text = ""
            for content_item in msg.content:
                content_text += content_item.text
            dict_messages.append({
                "role": msg.role,
                "content": content_text
            })

        # Apply chat template
        processed_text = tokenizer.apply_chat_template(
            dict_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        if return_text:
            return processed_text

        # Tokenize the processed text
        token_ids = tokenizer(
            processed_text,
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        # Move to appropriate device
        if device is not None:
            token_ids = token_ids.to(device)
        elif hasattr(tokenizer, 'device') and tokenizer.device is not None:
            token_ids = token_ids.to(tokenizer.device)
        else:
            # Try to infer device from CUDA availability
            import torch
            if torch.cuda.is_available():
                token_ids = token_ids.cuda()

        # Apply max_length constraint if specified
        if max_length is not None and token_ids.shape[1] > max_length:
            token_ids = token_ids[:, :max_length]

        return token_ids
