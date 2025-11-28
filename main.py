import os
from datetime import datetime
import json
import vllm
import yaml
import argparse
import numpy as np
import requests
import openai
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from conversation_creator import ConversationCreator
from agent import MemoryAgent
from memory import Memory
from functions import get_memory_tool_schemas

def load_agent_config(config_path):
    """Load agent configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate required fields
    required_fields = ['agent_name', 'model_name']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file: {config_path}")

    return config

def get_results_filename(agentic_search=False):
    """Get the appropriate results filename based on search method."""
    return "agentic_results.json" if agentic_search else "results.json"

def process_chunk_with_gpt4_mini(chunk_data):
    """Process a single chunk using GPT-4.1-mini via OpenAI API"""
    chunk, memory, agent_config, memory_agent_template = chunk_data

    # Build messages similar to process_text_with_qwen_pipeline but for chat.completions
    messages = []

    # Add memory system prompt if available
    if memory is not None:
        query = chunk[:100] + "..." if len(chunk) > 100 else chunk
        max_num_of_recent_chunks = getattr(MemoryAgent, 'MAX_MEMORY_ITEMS', 10)
        messages = memory.render_system_prompt(status='memorie', query=query, max_num_of_recent_chunks=max_num_of_recent_chunks)

    # Add user message with the chunk content
    messages.append({"role": "user", "content": chunk})

    # Get memory tools directly
    tools = get_memory_tool_schemas(memory)

    # Initialize Azure OpenAI client
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-01-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Generate response using GPT-4.1-mini with tools
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=agent_config.get('max_new_tokens', 2048),
        temperature=0.6
    )

    message = response.choices[0].message
    final_response = message.content or ""

    # Parse function calls directly from the structured response
    function_calls = []
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.type == "function":
                function_call = {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }

                # Execute the function call immediately
                tool_result = memory_agent_template._run_tool_from_function_call(
                    function_call,
                    memory
                )

                function_calls.append({
                    'function_call': function_call,
                    'tool_result': tool_result,
                    'timestamp': time.time()
                })

    return final_response, function_calls

def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Memory Agent Evaluation")
    parser.add_argument("--agent_config", type=str, required=True, help="Path to agent configuration YAML file")
    parser.add_argument("--dataset", type=str, default="LOCOMO", choices=['squad', 'squad_test', 'hotpotqa', 'booksum', 'friends', 'wos46985', 'pubmed-rct', 'arxiv-classification', 'eurlex', 'accurate_retrieval', 'long_range_understanding', 'conflict_resolution', 'test_time_learning', "LOCOMO", "LongMemEval", "MemAgent_Bench", "memalpha", "memalpha_train", 'memalpha_sample', "detectiveqa", 'memoryagentbench', 'perltqa', 'narrativeqa', 'accurate_retrieval', 'test_time_learning', 'cr_train']) # Restricted choices
    parser.add_argument("--load_db_from", type=str, default=None) # Memory databse
    parser.add_argument("--chunk_size", type=int, default=4096, help="Chunk size for MemAgent_Bench dataset")  # add parameter chunk_size
    parser.add_argument("--save_process", action="store_true", help="Enable process tracking for Qwen models (saves detailed logs)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batch processing")
    parser.add_argument("--agentic_search", action="store_true", help="Use agentic memory search instead of simple batch processing")
    parser.add_argument("--rollout_label", type=str, default=None, help="Label to append to output directory path, e.g., rollout_1")
    parser.add_argument("--force_reanswer_questions", action="store_true", help="Force reanswering all questions even if results file already exists")
    parser.add_argument(
        "--exclude_memory",
        nargs='*',
        default=[],
        help="Space or comma separated list of memory components to disable. Choose from: core, episodic, semantic."
    )

    args = parser.parse_args()
    allowed_memory_types = {"core", "semantic", "episodic"}
    normalized_exclusions = []
    for entry in args.exclude_memory:
        # Allow comma separated values in addition to whitespace separation
        parts = [part.strip().lower() for part in entry.split(",") if part.strip()]
        normalized_exclusions.extend(parts)

    invalid = sorted(set(normalized_exclusions) - allowed_memory_types)
    if invalid:
        parser.error(f"Invalid memory types for --exclude_memory: {', '.join(invalid)}. Allowed values: core, semantic, episodic.")

    args.exclude_memory = set(normalized_exclusions)
    return args

def run_with_chunks_and_questions_batch(
        args,
        agent_config,
        batch_indices,
        batch_chunks,
        batch_queries_and_answers,
        batch_sources):

    with open('config/prompts_wrt_datasource.yaml', 'r') as f:
        prompts_wrt_datasource = yaml.safe_load(f)

    batch_size = len(batch_chunks)

    # Get including_core parameter from agent_config, default to False
    # including_core = agent_config.get('including_core', False)
    batch_memories = [
        Memory(
            including_core=prompts_wrt_datasource[batch_sources[idx]]['including_core'],
            disabled_memory_types=args.exclude_memory
        )
        for idx in range(batch_size)
    ]
    memory_agent_template = MemoryAgent(agent_config=agent_config)

    # Check if agent_state.json exists for all batch items
    batch_out_dirs = []
    all_states_exist = True

    for i in range(batch_size):

        batch_idx = batch_indices[i]

        # Create output directory path
        if agent_config.get("model_name") is not None:
            out_dir = f"./agents/{agent_config['agent_name']}_{agent_config['model_name'].replace('/', '_')}_{args.dataset}"
        else:
            out_dir = f"./agents/{agent_config['agent_name']}_{args.dataset}"

        # Add external model info if using external model for question answering
        if agent_config.get('infer_with_full_memory', False) and agent_config.get('external_model_url'):
            external_model_name = agent_config.get('external_model_name', 'qwen3-32b').replace('/', '_')
            out_dir = out_dir + f"_ext_{external_model_name}"

        if not agent_config['enable_thinking']:
            out_dir = out_dir + "_no_thinking"

        if args.exclude_memory:
            out_dir = out_dir + "_exclude_" + "_".join(args.exclude_memory)

        # Add max_new_tokens to the directory name
        max_new_tokens = agent_config.get('max_new_tokens', 2048)
        out_dir = out_dir + f"_tokens_{max_new_tokens}"

        # Add rollout label if provided
        if args.rollout_label is not None:
            out_dir = out_dir + f"_rollout_{args.rollout_label}"

        out_dir += f"/{batch_idx}"
        batch_out_dirs.append(out_dir)

    for i in range(batch_size):
        out_dir = batch_out_dirs[i]
        # Check if agent_state.json exists for this batch item
        if not os.path.exists(f"{out_dir}/agent_state.json"):
            all_states_exist = False
            break

    # Load existing states if all exist, otherwise process chunks
    if all_states_exist:
        print(f"[DEBUG] Loading existing agent states for all batch items, skipping chunk processing...")

        for i in range(batch_size):
            out_dir = batch_out_dirs[i]

            # Load agent state
            with open(f"{out_dir}/agent_state.json", "r") as f:
                state = json.load(f)

            # Restore memory state
            memory = batch_memories[i]
            # Only restore core memory if it's available in the memory object
            if memory.including_core and memory.core is not None:
                memory.core = state.get('core', [])

            if memory.is_memory_type_enabled('semantic'):
                memory.semantic = state.get('semantic', [])
                memory.semantic_embedding_ids = state.get('semantic_embedding_ids', [])
            else:
                memory.semantic = []
                memory.semantic_embedding_ids = []

            if memory.is_memory_type_enabled('episodic'):
                memory.episodic = state.get('episodic', [])
                memory.episodic_embedding_ids = state.get('episodic_embedding_ids', [])
            else:
                memory.episodic = []
                memory.episodic_embedding_ids = []

            # Load embeddings if available
            embeddings_file = f"{out_dir}/embeddings.npz"
            if os.path.exists(embeddings_file):
                embeddings = np.load(embeddings_file)
                if memory.is_memory_type_enabled('semantic'):
                    memory.semantic_embedding_matrix = embeddings['semantic_matrix']
                else:
                    memory.semantic_embedding_matrix = np.array([])

                if memory.is_memory_type_enabled('episodic'):
                    memory.episodic_embedding_matrix = embeddings['episodic_matrix']
                else:
                    memory.episodic_embedding_matrix = np.array([])
            else:
                memory.semantic_embedding_matrix = np.array([])
                memory.episodic_embedding_matrix = np.array([])

        max_chunks = max(len(chunk_list) for chunk_list in batch_chunks) if len(batch_chunks) > 0 else 0
        print(f"[DEBUG] Loaded existing states, proceeding directly to question answering...")

    else:
        print(f"[DEBUG] Not all agent states exist, proceeding with chunk processing...")

        max_chunks = max(len(chunk_list) for chunk_list in batch_chunks) if len(batch_chunks) > 0 else 0

        # Initialize function calls storage for each batch item
        batch_function_calls_log = [[] for _ in range(batch_size)]
        batch_final_responses = {i: [] for i in range(batch_size)}

        # Check if we're using gpt-4.1-mini (detected by model_name containing "4.1-mini")
        use_gpt4_mini = agent_config.get('model_name', '').lower().find('4.1-mini') != -1

        for chunk_idx in range(max_chunks):

            print(f"[DEBUG] Processing chunk {chunk_idx + 1}/{max_chunks}")

            max_new_tokens = agent_config.get('max_new_tokens', 2048)

            remaining_indices = []
            current_chunks = []
            for i, chunk_list in enumerate(batch_chunks):
                if chunk_idx < len(chunk_list):
                    # current_chunks.append(f"\n\nHere is new information to process:\n{chunk_list[chunk_idx]}\n\nPlease analyze this information and decide what memory operations to perform. Maximum response length is {max_new_tokens}, the above chunk has {len(memory_agent_template.tokenizer(chunk_list[chunk_idx]))} tokens.")
                    # current_chunks.append(prompts_wrt_datasource[batch_sources[i]]['prompt'].format(context=chunk_list[chunk_idx], max_new_tokens=int(max_new_tokens * 0.8)))
                    current_chunks.append(prompts_wrt_datasource['unified_prompt'].format(context=chunk_list[chunk_idx], max_new_tokens=int(max_new_tokens * 0.8)))
                    # current_chunks.append(prompts_wrt_datasource['unified_prompt'].format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), context=chunk_list[chunk_idx],
                                # max_new_tokens=int(max_new_tokens * 0.8)))
                    remaining_indices.append(i)

            if len(remaining_indices) == 0:
                break

            if use_gpt4_mini:

                print(f"[DEBUG] Using GPT-4.1-mini with fake batch processing (multiprocessing) for chunk {chunk_idx + 1}")

                # Prepare data for multiprocessing
                chunk_data_list = []
                for i, chunk in enumerate(current_chunks):
                    memory_idx = remaining_indices[i]
                    memory = batch_memories[memory_idx]
                    chunk_data_list.append((chunk, memory, agent_config, memory_agent_template))

                # Use ThreadPoolExecutor for fake batch processing (simulating multiprocessing)
                final_responses = []
                gpt4_function_calls_list = []
                with ThreadPoolExecutor(max_workers=min(len(chunk_data_list), 16)) as executor:
                    futures = [executor.submit(process_chunk_with_gpt4_mini, data) for data in chunk_data_list]
                    for future in tqdm(futures, desc="Processing chunks", unit="chunk", total=len(futures)):
                        response, function_calls = future.result()
                        final_responses.append(response)
                        gpt4_function_calls_list.append(function_calls)

            else:

                prompts = []
                for chunk, memory in zip(current_chunks, [batch_memories[i] for i in remaining_indices]):
                    processed_text = MemoryAgent.process_text_with_qwen_pipeline(
                        text=chunk,
                        tokenizer=memory_agent_template.tokenizer,
                        functions=[tool["function"] for tool in get_memory_tool_schemas(memory)],
                        status='memorie',
                        enable_thinking=agent_config['enable_thinking'],
                        return_text=True,
                        memory=memory
                    )
                    prompts.append(processed_text)

                assert agent_config['vllm']

                # Import SamplingParams from vLLM for batch processing
                from vllm import SamplingParams

                if agent_config['enable_thinking']:
                    # First generation until thinking budget
                    thinking_budget = agent_config.get('thinking_budget', 1024)
                    max_new_tokens = agent_config.get('max_new_tokens', 2048)

                    thinking_sampling_params = SamplingParams(
                        temperature=0.7,
                        max_tokens=thinking_budget,
                        stop_token_ids=[memory_agent_template.tokenizer.eos_token_id]
                    )

                    outputs = memory_agent_template.model.generate(prompts, thinking_sampling_params)
                    first_responses = [output.outputs[0].text for output in outputs]

                    # Collect all texts that need second generation
                    second_gen_indices = []
                    second_gen_texts = []
                    has_early_stopping = []  # Track which ones have early stopping text
                    finished_indices = []

                    early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"

                    for i, (first_response, prompt) in enumerate(zip(first_responses, prompts)):
                        # Check if the generation has already finished or thinking process is complete
                        if (memory_agent_template.tokenizer.eos_token_id not in memory_agent_template.tokenizer(first_response).input_ids
                            and "</think>" not in first_response):
                            print(f"thinking budget is reached for prompt {i}")
                            # Add early stopping text and prepare for batch second generation
                            continued_text = prompt + first_response + early_stopping_text
                            second_gen_indices.append(i)
                            second_gen_texts.append(continued_text)
                            has_early_stopping.append(True)
                        elif ("</think>" in first_response
                              and memory_agent_template.tokenizer.eos_token_id not in memory_agent_template.tokenizer(first_response).input_ids):
                            # Thinking completed, continue generation after thinking
                            continued_text = prompt + first_response
                            second_gen_indices.append(i)
                            second_gen_texts.append(continued_text)
                            has_early_stopping.append(False)
                        else:
                            # Generation finished or no continuation needed
                            finished_indices.append(i)

                    # Batch second generation for all texts that need it
                    second_gen_responses = []
                    if second_gen_texts:
                        remaining_sampling_params = SamplingParams(
                            temperature=0.7,
                            max_tokens=max_new_tokens - thinking_budget,
                            stop_token_ids=[memory_agent_template.tokenizer.eos_token_id]
                        )
                        second_outputs = memory_agent_template.model.generate(second_gen_texts, remaining_sampling_params)
                        second_gen_responses = [output.outputs[0].text.strip() for output in second_outputs]

                    # Combine all responses in correct order
                    final_responses = [None] * len(first_responses)

                    # Fill in second generation responses
                    for i, idx in enumerate(second_gen_indices):
                        if has_early_stopping[i]:
                            # Budget reached case: include early stopping text
                            final_responses[idx] = first_responses[idx] + early_stopping_text + second_gen_responses[i]
                        else:
                            # Thinking complete case: no early stopping text
                            final_responses[idx] = first_responses[idx] + second_gen_responses[i]

                    # Fill in finished responses
                    for idx in finished_indices:
                        final_responses[idx] = first_responses[idx].strip()
                else:
                    # Single generation without thinking budget
                    max_new_tokens = agent_config.get('max_new_tokens', 2048)
                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=max_new_tokens,
                        stop_token_ids=[memory_agent_template.tokenizer.eos_token_id]
                    )
                    outputs = memory_agent_template.model.generate(prompts, sampling_params)
                    final_responses = [output.outputs[0].text.strip() for output in outputs]

            # Batch process responses and execute function calls
            batch_assistant_messages = []
            batch_function_calls = []

            # Parse all responses in batch
            if use_gpt4_mini:
                # For GPT-4.1-mini, function calls are already executed, just store them
                for i, (response, memory_idx) in enumerate(zip(final_responses, remaining_indices)):
                    batch_final_responses[memory_idx].append(response)

                    # Store the already executed function calls
                    function_calls = gpt4_function_calls_list[i]
                    for function_call_record in function_calls:
                        function_call_record['chunk_idx'] = chunk_idx
                        batch_function_calls_log[memory_idx].append(function_call_record)

            else:
                # For qwen, parse responses and execute function calls
                for i, (response, memory_idx) in enumerate(zip(final_responses, remaining_indices)):
                    assistant_messages = memory_agent_template._parse_response(response)
                    batch_assistant_messages.append((assistant_messages, memory_idx))
                    batch_final_responses[memory_idx].append(response)

                    # Collect function calls for batch execution
                    function_calls_messages = [msg for msg in assistant_messages if msg.get("function_call")]
                    if function_calls_messages:
                        for assistant_msg in function_calls_messages:
                            batch_function_calls.append((assistant_msg["function_call"], memory_idx))

            # Execute function calls in batch and collect results (only for qwen)
            if batch_function_calls and not use_gpt4_mini:
                for function_call, memory_idx in batch_function_calls:
                    tool_result = memory_agent_template._run_tool_from_function_call(
                        function_call,
                        batch_memories[memory_idx]
                    )
                    # Store the function call and result in the appropriate batch item log
                    function_call_record = {
                        'function_call': function_call,
                        'tool_result': tool_result,
                        'chunk_idx': chunk_idx,
                        'timestamp': time.time()
                    }
                    batch_function_calls_log[memory_idx].append(function_call_record)

        # Save all memory states after chunk processing
        for i in range(batch_size):
            batch_idx = batch_indices[i]
            memory = batch_memories[i]
            out_dir = batch_out_dirs[i]

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Save memory state
            state = {
                'semantic': memory.semantic,
                'episodic': memory.episodic,
                'conversation_history': [],
                'step': max_chunks,
                'semantic_embedding_ids': memory.semantic_embedding_ids,
                'episodic_embedding_ids': memory.episodic_embedding_ids
            }

            # Only save core memory if it's available
            if memory.including_core and memory.core is not None:
                state['core'] = memory.core

            with open(f"{out_dir}/agent_state.json", "w") as f:
                json.dump(state, f, indent=2)

            # Save data instance info
            data_instance_info = {
                'data_source': batch_sources[i],
                'global_idx': batch_idx
            }
            with open(f"{out_dir}/data_instance_info.json", "w") as f:
                json.dump(data_instance_info, f, indent=2)

            # Save chunks with their corresponding function calls in a single file
            chunks_with_function_calls = []
            for chunk_idx, chunk in enumerate(batch_chunks[i]):
                # Get function calls for this specific chunk
                chunk_function_calls = [
                    fc for fc in batch_function_calls_log[i]
                    if fc.get('chunk_idx') == chunk_idx
                ]

                chunks_with_function_calls.append({
                    'chunk_idx': chunk_idx,
                    'raw_chunk': chunk,
                    'function_calls': chunk_function_calls
                })

            with open(f"{out_dir}/chunks_and_function_calls.json", "w") as f:
                json.dump(chunks_with_function_calls, f, indent=2)

            with open(f"{out_dir}/final_responses.json", "w") as f:
                json.dump(batch_final_responses[i], f, indent=2)

            # Save embeddings if available
            if (memory.semantic_embedding_matrix.size > 0 or
                memory.episodic_embedding_matrix.size > 0):
                np.savez_compressed(f"{out_dir}/embeddings.npz",
                                  semantic_matrix=memory.semantic_embedding_matrix,
                                  episodic_matrix=memory.episodic_embedding_matrix)


    # TODO: check if the results file exists for all batch items
    results_filename = get_results_filename(args.agentic_search)
    all_results_exist = True
    if not args.force_reanswer_questions:
        for i in range(batch_size):
            out_dir = batch_out_dirs[i]
            if not os.path.exists(f"{out_dir}/{results_filename}"):
                all_results_exist = False
                break
        if all_results_exist:
            all_results = []
            for i in range(batch_size):
                out_dir = batch_out_dirs[i]
                with open(f"{out_dir}/{results_filename}", "r") as f:
                    all_results.extend(json.load(f))
            return all_results
    else:
        all_results_exist = False

    # First, save all memory states in parallel
    all_results = []

    for i in range(batch_size):
        batch_idx = batch_indices[i]
        memory = batch_memories[i]
        out_dir = batch_out_dirs[i]

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Save memory state (in case it wasn't saved during chunk processing)
        state = {
            'semantic': memory.semantic,
            'episodic': memory.episodic,
            'conversation_history': [],
            'step': max_chunks,
            'semantic_embedding_ids': memory.semantic_embedding_ids,
            'episodic_embedding_ids': memory.episodic_embedding_ids
        }

        # Only save core memory if it's available
        if memory.including_core and memory.core is not None:
            state['core'] = memory.core

        with open(f"{out_dir}/agent_state.json", "w") as f:
            json.dump(state, f, indent=2)

        # Save data instance info
        data_instance_info = {
            'data_source': batch_sources[i],
            'global_idx': batch_idx
        }
        with open(f"{out_dir}/data_instance_info.json", "w") as f:
            json.dump(data_instance_info, f, indent=2)

        # Save embeddings if available
        if (memory.semantic_embedding_matrix.size > 0 or
            memory.episodic_embedding_matrix.size > 0):
            np.savez_compressed(f"{out_dir}/embeddings.npz",
                              semantic_matrix=memory.semantic_embedding_matrix,
                              episodic_matrix=memory.episodic_embedding_matrix)

    # Collect all questions for batch processing
    all_questions = []
    question_metadata = []  # Store metadata for each question

    for i in range(batch_size):
        batch_idx = batch_indices[i]
        queries_and_answers = batch_queries_and_answers[i]
        memory = batch_memories[i]

        for item in queries_and_answers:
            if args.dataset == "LOCOMO":
                question_idx, question, answer, category = item
                question_metadata.append({
                    'batch_idx': batch_idx,
                    'memory_idx': i,
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'dataset_type': 'LOCOMO'
                })
            elif args.dataset == "MemAgent_Bench":
                question_idx, question, answer, category, source = item
                question_metadata.append({
                    'batch_idx': batch_idx,
                    'memory_idx': i,
                    'question': question,
                    'answer': answer,
                    'category': category,
                    'source': source,
                    'dataset_type': 'MemAgent_Bench'
                })
            else:
                question_idx, question, answer, data_source = item
                question_metadata.append({
                    'batch_idx': batch_idx,
                    'memory_idx': i,
                    'question': question,
                    'answer': answer,
                    'data_source': data_source,
                    'dataset_type': args.dataset
                })

            all_questions.append(question)

    # Process questions in batches using temporary agents or external model
    question_responses = []
    question_step_infos = []

    # Check if we should use external model for batch inference
    if agent_config.get('infer_with_full_memory', False) and agent_config.get('external_model_url'):

        # Prepare questions grouped by memory (batch item)
        questions_by_memory = {}
        for i, meta in enumerate(question_metadata):
            memory_idx = meta['memory_idx']
            question = meta['question']

            if meta.get("data_source", None):
                data_source = meta['data_source']
                query_prompt = prompts_wrt_datasource[data_source]['query_prompt']
                if query_prompt is not None:
                    question = query_prompt + "\n\n" + question

            if memory_idx not in questions_by_memory:
                questions_by_memory[memory_idx] = []
            questions_by_memory[memory_idx].append({'question': question, 'metadata_idx': i})

        # Prepare payload for memory server
        batch_memories_for_server = []
        questions_for_server = []

        for memory_idx in sorted(questions_by_memory.keys()):
            memory = batch_memories[memory_idx]
            # Prepare memory dict for server
            memory_dict = {
                'episodic': memory.episodic,
                'semantic': memory.semantic
            }
            # Only include core if it exists
            if memory.including_core and memory.core is not None:
                memory_dict['core'] = memory.core

            batch_memories_for_server.append(memory_dict)
            questions_for_server.append([q['question'] for q in questions_by_memory[memory_idx]])

        # Make request to memory server
        payload = {
            "memories": batch_memories_for_server,
            'questions': questions_for_server
        }

        # Choose endpoint based on agentic_search flag
        base_url = agent_config['external_model_url']
        if base_url.endswith('/batch_process'):
            base_url = base_url[:-len('/batch_process')]

        if args.agentic_search:
            endpoint = f"{base_url}/agentic_process"
        else:
            endpoint = f"{base_url}/batch_process"

        response = requests.post(endpoint, json=payload)

        if response.status_code != 200:
            raise Exception(f"Memory server request failed with status {response.status_code}: {response.text}")

        server_results = response.json().get('result', [])

        # Process server results and map back to original question order
        question_responses = [None] * len(all_questions)
        question_step_infos = [None] * len(all_questions)

        result_idx = 0
        for memory_idx in sorted(questions_by_memory.keys()):
            memory_questions = questions_by_memory[memory_idx]
            memory_results = server_results[result_idx] if result_idx < len(server_results) else []

            for i, q_info in enumerate(memory_questions):
                metadata_idx = q_info['metadata_idx']
                response_text = memory_results[i] if i < len(memory_results) else "No response from server"

                question_responses[metadata_idx] = response_text
                step_info = {
                    "step": max_chunks,
                    "final_response": response_text,
                    "memory_server_used": True,
                    "batch_processed": True,
                    "agentic_search_used": args.agentic_search
                }
                question_step_infos[metadata_idx] = step_info

            result_idx += 1

    elif agent_config.get('external_model_url'):
        # Handle case without infer_with_full_memory flag
        # This case would need similar implementation based on your requirements
        raise NotImplementedError("Memory server without infer_with_full_memory not yet implemented")

    else:
        raise NotImplementedError("Only memory server inference is supported for batch processing")

    # Group results by batch item and save
    batch_results_dict = {}
    for i, meta in enumerate(question_metadata):
        batch_idx = meta['batch_idx']
        if batch_idx not in batch_results_dict:
            batch_results_dict[batch_idx] = []

        # Format result based on dataset type
        if meta['dataset_type'] == 'LOCOMO':
            result = {
                'question': meta['question'],
                'response': question_responses[i],
                'answer': meta['answer'],
                'category': meta['category'],
                'step_info': question_step_infos[i]
            }
        elif meta['dataset_type'] == 'MemAgent_Bench':
            result = {
                'question': meta['question'],
                'response': question_responses[i],
                'answer': meta['answer'],
                'category': meta['category'],
                'source': meta['source'],
                'step_info': question_step_infos[i]
            }
        else:
            result = {
                'question': meta['question'],
                'response': question_responses[i],
                'answer': meta['answer'],
                'step_info': question_step_infos[i]
            }

        batch_results_dict[batch_idx].append(result)

    # Save results for each batch item
    for i, batch_idx in enumerate(batch_indices):
        batch_results = batch_results_dict[batch_idx]
        out_dir = batch_out_dirs[i]

        results_filename = get_results_filename(args.agentic_search)
        with open(f"{out_dir}/{results_filename}", "w") as f:
            json.dump(batch_results, f, indent=2)

        all_results.extend(batch_results)

    return all_results

def main():

    args = parse_args()

    # Load agent configuration
    agent_config = load_agent_config(args.agent_config)

    # Print loaded configuration
    print(f"Loaded agent configuration:")
    print(f"  Agent name: {agent_config['agent_name']}")
    print(f"  Model name: {agent_config['model_name']}")
    if 'enable_thinking' in agent_config:
        print(f"  Enable thinking: {agent_config['enable_thinking']}")
    print(f"  Save process (Qwen only): {args.save_process}")
    if args.exclude_memory:
        print(f"  Disabled memories: {', '.join(sorted(args.exclude_memory))}")
    else:
        print(f"  Disabled memories: None")

    conversation_creator = ConversationCreator(args.dataset, args.chunk_size)

    all_chunks = conversation_creator.chunks() # TODO: Note we don't skip already completed conversations in chunking process of conversation_creator.py since it's easy to mess up the index sequence in eval.py, but we can fix it later (return empty chunks instead of skipping)

    all_queries_and_answers = conversation_creator.get_query_and_answer()

    # Handle cases where some instances might have empty Q&A lists
    all_sources = []
    for item in all_queries_and_answers:
        if len(item) > 0:
            all_sources.append(item[0][-1])
        else:
            # Default source for empty Q&A lists based on dataset
            all_sources.append(args.dataset)

    # just for debug
    for item in all_queries_and_answers:
        if len(item) > 0:
            assert len(np.unique([x[-1] for x in item])) == 1, "all sources should be the same"

    print(f"Processing {len(all_chunks)} conversations for dataset {args.dataset}...")

    # Process all conversations using batch processing
    all_indices = list(range(len(all_chunks)))
    for i in range(0, len(all_indices), args.batch_size):
        batch_indices = all_indices[i:i+args.batch_size]
        batch_chunks = [all_chunks[idx] for idx in batch_indices]
        batch_sources = [all_sources[idx] for idx in batch_indices]
        batch_queries_and_answers = [all_queries_and_answers[idx] for idx in batch_indices]
        run_with_chunks_and_questions_batch(args, agent_config, batch_indices, batch_chunks, batch_queries_and_answers, batch_sources)

if __name__ == '__main__':
    main()
