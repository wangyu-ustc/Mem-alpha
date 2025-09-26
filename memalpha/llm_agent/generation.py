import os
import re
import yaml
import time
import torch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import requests
from memory import Memory
from memalpha.utils import count_tokens


@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3

@dataclass
class MemoryGenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    num_rollouts: int = 1
    search_url: str = None  # URL for search service
    topk: int = 3  # Number of search results to return
    respond_url: str = None
    analyze_function_url: str = None
    enable_thinking: bool = True
    including_core: bool = False

class MemoryGenerationManager:
    """Generation manager for memory agent that processes chunks and performs memory operations."""

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: MemoryGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        with open("/home/wangyu/work/Mem-alpha/config/prompts_wrt_datasource.yaml", "r") as f:
            self.prompts_wrt_datasource = yaml.safe_load(f)

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract memory operations."""
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )

        # Extract memory operations (memory_insert, memory_update, memory_delete)
        # or final completion signal
        responses_str = [resp.split('</memory_insert>')[0] + '</memory_insert>'
                 if '</memory_insert>' in resp
                 else resp.split('</memory_update>')[0] + '</memory_update>'
                 if '</memory_update>' in resp
                 else resp.split('</memory_delete>')[0] + '</memory_delete>'
                 if '</memory_delete>' in resp
                 else resp.split('</done>')[0] + '</done>'
                 if '</done>' in resp
                 else resp
                 for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_chunk(self, chunks: List[str], batch_memories: List[Memory]) -> torch.Tensor:
        """Process next chunks of information using memory agent template logic."""

        # Import the memory agent to use its processing logic
        from agent import MemoryAgent
        from functions import MEMORY_TOOL_SCHEMAS, get_memory_tool_schemas

        # Get memory tool functions for processing
        # Use the first memory instance for schema generation (all should have same config)
        functions = [tool["function"] for tool in get_memory_tool_schemas(batch_memories[0])]

        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        else:
            device = "cpu"

        # Process each chunk using the memory agent's full pipeline
        processed_token_ids_list = []
        for chunk, memory in zip(chunks, batch_memories):
            if chunk.strip():  # Only process non-empty chunks with full pipeline
                # Use the memory agent's static method for efficient processing
                token_ids = MemoryAgent.process_text_with_qwen_pipeline(
                    text=chunk,
                    tokenizer=self.tokenizer,
                    functions=functions,
                    status='memorie',
                    enable_thinking=self.config.enable_thinking,
                    memory=memory,
                    device=device,
                    # max_length=self.config.max_obs_length
                )
                max_num_of_recent_chunks = MemoryAgent.MAX_MEMORY_ITEMS
                while token_ids.shape[-1] > 6144 and max_num_of_recent_chunks > 0:
                    max_num_of_recent_chunks = max_num_of_recent_chunks // 2

                    token_ids = MemoryAgent.process_text_with_qwen_pipeline(
                        text=chunk,
                        tokenizer=self.tokenizer,
                        functions=functions,
                        status='memorie',
                        enable_thinking=self.config.enable_thinking,
                        memory=memory,
                        device=device,
                        max_num_of_recent_chunks=max_num_of_recent_chunks
                    )

            else:
                # For empty chunks, create minimal token sequence
                token_ids = self.tokenizer(
                    "",
                    return_tensors='pt',
                    add_special_tokens=False,
                )['input_ids']
                if token_ids.shape[1] == 0:
                    # Ensure we have at least one token
                    token_ids = torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long)

                # Move to same device
                if device is not None:
                    token_ids = token_ids.to(device)

            processed_token_ids_list.append(token_ids)

        for token_ids in processed_token_ids_list:
            assert token_ids.shape[1] <= 6144, f"token_ids.shape[1]: {token_ids.shape[1]}"

        # Batch the processed token IDs with padding
        if processed_token_ids_list:
            # Pad to the same length for batching - remove truncation
            max_len = max(ids.shape[1] for ids in processed_token_ids_list)

            padded_ids_list = []
            for token_ids in processed_token_ids_list:
                if token_ids.shape[1] < max_len:
                    # Pad to max_len
                    padding = torch.full(
                        (token_ids.shape[0], max_len - token_ids.shape[1]),
                        self.tokenizer.pad_token_id,
                        dtype=token_ids.dtype,
                        device=token_ids.device
                    )
                    padded_ids = torch.cat([token_ids, padding], dim=1)
                elif token_ids.shape[1] > max_len:
                    # Truncate to max_len
                    padded_ids = token_ids[:, :max_len]
                else:
                    padded_ids = token_ids
                padded_ids_list.append(padded_ids)

            # Stack into batch
            chunk_ids = torch.cat(padded_ids_list, dim=0)
        else:
            # Empty case - return empty tensor on the correct device
            chunk_ids = torch.zeros((len(chunks), 1), dtype=torch.long, device=device)

        assert chunk_ids.shape[1] <= 6144, f"chunk_ids.shape[1]: {chunk_ids.shape[1]}"

        return chunk_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor,
                            next_chunk_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and chunks."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_chunk_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        # max_len = min(self.config.max_prompt_length, effective_len)
        max_len = effective_len

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        # Copy meta_info from original batch to ensure eos_token_id, pad_token_id etc. are available
        padded_active_batch.meta_info = active_batch.meta_info.copy() if hasattr(active_batch, 'meta_info') else {}
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        padded_result = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from result
        result_dict = {}
        for k, v in padded_result.batch.items():
            result_dict[k] = v[:batch_size]

        result = DataProto.from_dict(result_dict)
        result.meta_info = padded_result.meta_info
        return result

    def _process_question_with_memory_using_server(self, batch_memories, questions_and_answers):

        payload = {
            "memories": [{'core': memory.core, 'episodic': memory.episodic, 'semantic': memory.semantic} for memory in batch_memories],
            'questions': [[qa['question'] for qa in qa_list] for qa_list in questions_and_answers]
        }
        import requests

        print("Number of questions:", [len(qa_list) for qa_list in questions_and_answers])
        print("Number of questions in total:", sum([len(qa_list) for qa_list in questions_and_answers]))

        start_time = time.time()

        # Retry up to 3 times if results are empty
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(self.config.respond_url, json=payload)
            results = response.json().get('result', [])

            if len(results) > 0:
                break

            if attempt < max_retries - 1:
                print(f"Got empty results on attempt {attempt + 1}, retrying...")
            else:
                raise AssertionError(f"Got empty results after {max_retries} attempts")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        questions_list = [[qa['question'] for qa in qa_list] for qa_list in questions_and_answers]

        def clean_pred(pred):
            if '</think>' in pred:
                return pred.split("</think>")[1].strip()
            elif "<think>" in pred:
                return "Empty"
            else:
                return pred

        results = [
            [clean_pred(pred) for pred in result]
            for result in results
        ]
        return questions_list, results

    def _analyze_function_call_content(self, function_calls: List[Dict]) -> List[float]:

        for function_call in function_calls:
            for fc in function_call:
                if 'arguments' in fc:
                    if 'memory_type' in fc['arguments']:
                        fc['arguments']['memory_type'] = fc['arguments']['memory_type'].replace("_memory", "")

        response = requests.post(
            self.config.analyze_function_url,
            json={'batch': function_calls},
            headers={'Content-Type': 'application/json'},
        )

        if response.status_code == 200:
            result = response.json()
            return result['scores']
        else:
            print(f"[WARNING] Analyze function endpoint returned status {response.status_code}: {response.text}")
            return [0.0] * len(function_calls)

    def run_memory_loop(self, gen_batch, chunks: List[List[str]], questions_and_answers: List[List[dict]], data_sources: List[str], num_gpus: int = 1) -> DataProto:
        """Run memory agent loop to process chunks and perform memory operations."""

        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)

        # Initialize separate memory instance for each instance in the batch
        batch_memories = [Memory(including_core=self.prompts_wrt_datasource[data_sources[i % batch_size]]['including_core']) for i in range(batch_size * self.config.num_rollouts)]

        # Initialize meta_info to avoid UnboundLocalError
        meta_info = {}
        last_chunk_meta_info = {}

        # Initialize rollings - will be set in first iteration
        rollings = None

        # Main memory processing loop - iterate through chunks
        max_chunks = max(len(chunk_list) for chunk_list in chunks) if len(chunks) > 0 else 0

        chunk_input_ids_list = []
        chunk_responses_ids_list = []
        chunk_response_masks_list = []
        chunk_function_call_rewards_list = []  # Track function call rewards for each chunk
        chunk_function_calls_list = []

        total_chunk_length = [0] * batch_size

        for chunk_idx in range(max_chunks):

            print(f"[DEBUG] Processing chunk {chunk_idx + 1}/{max_chunks}")

            remaining_indices = []
            # Get current chunks for active sequences
            current_chunks = []
            for i, chunk_list in enumerate(chunks):
                if chunk_idx < len(chunk_list):
                    current_chunks.append(self.prompts_wrt_datasource['unified_prompt'].format(context=chunk_list[chunk_idx], max_new_tokens=int(self.config.max_response_length * 0.8)))
                    remaining_indices.append(i)
                    chunk_content = chunk_list[chunk_idx]
                    total_chunk_length[i] += count_tokens(chunk_content)
                else:
                    active_mask[i] = 0

            if not active_mask.sum():
                break

            # Start fresh with empty context for all chunks
            device = gen_batch.batch['input_ids'].device
            empty_ids = torch.zeros((len(remaining_indices), 0), dtype=torch.long, device=device)
            rollings = DataProto.from_dict({
                'input_ids': empty_ids,
                'attention_mask': torch.ones_like(empty_ids),
                'position_ids': torch.zeros_like(empty_ids)
            })
            rollings.meta_info = gen_batch.meta_info.copy()

            # Process current chunks following agent.py chat() logic for memory operations
            active_chunk_input_ids, active_chunk_responses_ids, active_chunk_response_mask, chunk_meta_info = self._process_chunk_with_memory_operations(
                rollings, current_chunks, [batch_memories[i] for i in remaining_indices]
            )
            last_chunk_meta_info = chunk_meta_info  # Keep track of last chunk's meta_info

            chunk_input_ids = [None] * batch_size
            chunk_responses_ids = [None] * batch_size
            chunk_response_masks = [None] * batch_size
            chunk_function_call_rewards = [None] * batch_size  # Initialize with 0.0 for all sequences
            chunk_function_calls = [None] * batch_size

            for reordered_idx, idx in enumerate(remaining_indices):
                chunk_input_ids[idx] = active_chunk_input_ids[reordered_idx]
                chunk_responses_ids[idx] = active_chunk_responses_ids[reordered_idx]
                chunk_response_masks[idx] = active_chunk_response_mask[reordered_idx]
                # Get function call reward for this active sequence
                assert reordered_idx < len(chunk_meta_info['function_call_rewards'])
                if 'function_call_rewards' in chunk_meta_info:
                    chunk_function_call_rewards[idx] = chunk_meta_info['function_call_rewards'][reordered_idx]
                if 'all_function_calls' in chunk_meta_info:
                    chunk_function_calls[idx] = chunk_meta_info['all_function_calls'][reordered_idx]

            chunk_input_ids_list.append(chunk_input_ids)
            chunk_responses_ids_list.append(chunk_responses_ids)
            chunk_response_masks_list.append(chunk_response_masks)
            chunk_function_call_rewards_list.append(chunk_function_call_rewards)
            chunk_function_calls_list.append(chunk_function_calls)

        total_memory_length = [memory.total_length() for memory in batch_memories]

        questions_and_answers_with_query_prompt = []
        for idx, qa_list in enumerate(questions_and_answers):
            if self.prompts_wrt_datasource[data_sources[idx]]['query_prompt'] is not None:
                # qa_list = [self.prompts_wrt_datasource[data_sources[idx]]['query_prompt'] + "\n\n" + qa for qa in qa_list]
                qa_list = [{'question': self.prompts_wrt_datasource[data_sources[idx]]['query_prompt'] + "\n\n" + qa['question'], 'answer': qa['answer']} for qa in qa_list]
            questions_and_answers_with_query_prompt.append(qa_list)

        questions_list, predicted_answers_list = self._process_question_with_memory_using_server(
            batch_memories, questions_and_answers_with_query_prompt
        )

        ground_truth_answers_list = [[qa['answer'] for qa in qa_list] for qa_list in questions_and_answers]

        all_input_ids = []
        all_response_ids = []
        all_response_masks = []
        indices_in_batch = []
        all_function_call_rewards = []
        all_function_calls = []

        ### we can now prepare the final output

        for x in chunk_input_ids_list:
            assert x is not None

        for idx in range(batch_size):

            current_input_ids = [x[idx] for x in chunk_input_ids_list if x[idx] is not None]
            current_response_ids = [x[idx] for x in chunk_responses_ids_list if x[idx] is not None]
            current_response_masks = [x[idx] for x in chunk_response_masks_list if x[idx] is not None]
            current_function_call_rewards = [x[idx] for x in chunk_function_call_rewards_list if x[idx] is not None]
            current_function_calls = [x[idx] for x in chunk_function_calls_list if x[idx] is not None]

            assert len(current_input_ids) == len(current_response_ids) == len(current_response_masks) == len(current_function_call_rewards) == len(current_function_calls)

            indices_in_batch.extend([idx] * len(current_input_ids))
            all_input_ids.extend(current_input_ids)
            all_response_ids.extend(current_response_ids)
            all_response_masks.extend(current_response_masks)
            all_function_call_rewards.extend(current_function_call_rewards)
            all_function_calls.extend(current_function_calls)

        # pad to the left
        max_input_length = max(len(input_ids) for input_ids in all_input_ids)
        new_all_input_ids = []
        for input_ids in all_input_ids:
            if len(input_ids) < max_input_length:
                new_all_input_ids.append(torch.cat([torch.tensor([self.tokenizer.pad_token_id] * (max_input_length - len(input_ids))),input_ids]))
            else:
                new_all_input_ids.append(input_ids)
        all_input_ids = new_all_input_ids

        # pad to the right
        max_response_length = max(len(response_ids) for response_ids in all_response_ids)
        new_all_response_ids = []
        new_all_response_masks = []
        for response_ids, response_mask in zip(all_response_ids, all_response_masks):
            if len(response_ids) < max_response_length:
                new_all_response_ids.append(torch.cat([
                    response_ids,
                    torch.tensor([self.tokenizer.pad_token_id] * (max_response_length - len(response_ids)))
                ]))
                # Pad response_mask with True (unmasked) for padding tokens
                new_all_response_masks.append(torch.cat([
                    response_mask,
                    torch.tensor([False] * (max_response_length - len(response_mask)))
                ]))
            else:
                new_all_response_ids.append(response_ids)
                new_all_response_masks.append(response_mask)

        all_response_ids = new_all_response_ids
        all_response_masks = new_all_response_masks

        final_output = {'prompts': torch.stack(all_input_ids),
                        'responses': torch.stack(all_response_ids),
                        'response_mask': torch.stack(all_response_masks)}

        final_output['input_ids'] = torch.cat([final_output['prompts'], final_output['responses']], dim=1)
        final_output['attention_mask'] = torch.where(final_output['input_ids'] != self.tokenizer.pad_token_id, 1, 0)
        final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])
        final_output['attention_mask'][:, -final_output['response_mask'].shape[1]:] = final_output['response_mask']

        if self.config.analyze_function_url is not None:
            # we need to add function_call_content_rewards
            all_function_call_content_rewards = self._analyze_function_call_content(all_function_calls)

        # Repeat tensors to match the number of GPUs for distributed processing
        if num_gpus > 1:
            current_batch_size = final_output['input_ids'].shape[0]
            padding_needed = ((current_batch_size + num_gpus - 1) // num_gpus) * num_gpus - current_batch_size

            if padding_needed > 0:
                # Calculate how many times to repeat each tensor to reach the padding
                repeat_indices = torch.arange(current_batch_size)[:padding_needed]

                # Repeat tensors by adding the first few samples again
                final_output['input_ids'] = torch.cat([
                    final_output['input_ids'],
                    final_output['input_ids'][repeat_indices]
                ], dim=0)
                final_output['attention_mask'] = torch.cat([
                    final_output['attention_mask'],
                    final_output['attention_mask'][repeat_indices]
                ], dim=0)
                final_output['position_ids'] = torch.cat([
                    final_output['position_ids'],
                    final_output['position_ids'][repeat_indices]
                ], dim=0)
                final_output['prompts'] = torch.cat([
                    final_output['prompts'],
                    final_output['prompts'][repeat_indices]
                ], dim=0)
                final_output['responses'] = torch.cat([
                    final_output['responses'],
                    final_output['responses'][repeat_indices]
                ], dim=0)
                final_output['response_mask'] = torch.cat([
                    final_output['response_mask'],
                    final_output['response_mask'][repeat_indices]
                ], dim=0)

                repeated_all_function_call_rewards = [all_function_call_rewards[i] for i in repeat_indices.tolist()]
                all_function_call_rewards.extend(repeated_all_function_call_rewards)

                repeated_all_function_calls = [all_function_calls[i] for i in repeat_indices.tolist()]
                all_function_calls.extend(repeated_all_function_calls)

                if self.config.analyze_function_url is not None:
                    repeated_all_function_call_content_rewards = [all_function_call_content_rewards[i] for i in repeat_indices.tolist()]
                    all_function_call_content_rewards.extend(repeated_all_function_call_content_rewards)

                # Also repeat the indices_in_batch to match
                repeated_indices = [indices_in_batch[i] for i in repeat_indices.tolist()]
                indices_in_batch.extend(repeated_indices)

        # need to add attention_mask to avoid the gradients on "current_thinking_prompt"
        every_chunk_length = []
        for response_mask in final_output['response_mask']:
            response_length = response_mask.sum().item()
            every_chunk_length.append(response_length)

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update({
            'questions_list': questions_list,
            'predicted_answers_list': predicted_answers_list,
            'ground_truth_answers_list': ground_truth_answers_list,
            'indices_in_batch': indices_in_batch,
            'total_chunk_length': total_chunk_length,
            'total_memory_length': total_memory_length,
            'every_chunk_length': every_chunk_length,
            'batch_memories': [{'core': memory.core if memory.core is not None else "", 'episodic': memory.episodic, 'semantic': memory.semantic} for memory in batch_memories],
            'all_function_call_rewards': all_function_call_rewards,
            "all_function_calls": all_function_calls
        })
        if self.config.analyze_function_url is not None:
            final_output.meta_info['all_function_call_content_rewards'] = all_function_call_content_rewards
        final_output.meta_info.update(last_chunk_meta_info)
        return final_output

    def _process_chunk_with_memory_operations(self, rollings: DataProto, current_chunks: List[str], batch_memories: List[Memory]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Process chunks following agent.py chat() logic with status='memorie'.
        This implements the same loop structure as agent.chat() for memory operations.
        """
        from agent import MemoryAgent

        # Initialize a single memory agent template for function execution
        memory_agent_template = MemoryAgent(agent_config={'model_name': 'Qwen/Qwen3-4B', 'vllm': False, 'including_core': self.config.including_core}, is_template=True)

        # Add chunks to rolling state
        chunk_ids = self._process_next_chunk(current_chunks, batch_memories)

        # Move rollings to same device as chunk_ids to avoid device mismatch
        device = chunk_ids.device
        rollings.batch = rollings.batch.to(device)

        # Use empty response tensor with shape (batch_size, 0)
        empty_response = chunk_ids[:, :0]  # Creates tensor with shape (batch_size, 0)

        rollings = self._update_rolling_state(rollings,
                                            empty_response,  # Empty response with correct batch size
                                            chunk_ids)

        print("prompt length in rollings:", rollings.batch['input_ids'].shape)

        chunk_input_ids = rollings.batch['input_ids']

        # TODO: Now let's simply do one turn of memory operation
        gen_output = self._generate_with_gpu_padding(rollings)
        responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

        if self.config.enable_thinking:

            # Define the sentence to be masked
            sentence_to_mask = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
            sentence_to_mask_ids = self.tokenizer.encode(sentence_to_mask, add_special_tokens=False)

            new_responses_ids = []
            new_responses_str = []
            original_response_ids = []  # Keep track of original response IDs for masking
            needs_masking = []

            # check if there is unstopped thinking
            for response_ids, response_str in zip(responses_ids, responses_str):
                if "</think>" not in response_str:
                    # Store original response IDs for masking comparison
                    original_response_ids.append(response_ids)
                    response_str += sentence_to_mask
                    new_response_ids = self.tokenizer.encode(response_str, return_tensors="pt")[0]
                    new_responses_ids.append(new_response_ids)
                    new_responses_str.append(response_str)
                    needs_masking.append(True)
                else:
                    original_response_ids.append(response_ids)
                    new_responses_ids.append(response_ids)
                    new_responses_str.append(response_str)
                    needs_masking.append(False)

            # pad to the same length
            max_length = max(len(response_ids) for response_ids in new_responses_ids)
            new_responses_ids = [torch.cat([response_ids, torch.tensor([self.tokenizer.pad_token_id] * (max_length - len(response_ids)))]) for response_ids in new_responses_ids]
            new_responses_ids = torch.stack(new_responses_ids).long()
            rollings = self._update_rolling_state(rollings,
                                                empty_response,  # Empty response with correct batch size
                                                new_responses_ids)

            gen_output_2 = self._generate_with_gpu_padding(rollings)
            responses_ids, responses_str = self._postprocess_responses(gen_output_2.batch['responses'])
            responses_str = [new_rs + rs for new_rs, rs in zip(new_responses_str, responses_str)]

        # Cut the responses_str if if starts generating "RETURN" and "RESULTS"
        new_responses_str = []
        for response_str in responses_str:
            if "✿RESULT✿:" in response_str:
                response_str = response_str.split("✿RESULT✿:")[0].strip()
            new_responses_str.append(response_str)

        responses_str = new_responses_str
        responses_ids = self._batch_tokenize(new_responses_str)

        responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, torch.ones(responses_ids.shape[0], dtype=torch.bool))

        # Generate response mask by comparing original and new response IDs
        response_mask = self.tensor_fn.create_attention_mask(responses_ids)

        if self.config.enable_thinking:
            # Create mask for the tokens that were added
            for i, (original_ids, new_ids, need_masking) in enumerate(zip(original_response_ids, responses_ids, needs_masking)):
                original_len = len(original_ids)
                if need_masking:
                    response_mask[i, original_len: original_len+len(sentence_to_mask_ids)] = 0

        # Track function call success rates for reward calculation
        function_call_rewards = []
        all_function_calls = []

        for i, prediction in enumerate(responses_str):

            # Parse memory operation from prediction
            # operation, params = self.parse_memory_operation(prediction)
            assistant_messages = memory_agent_template._parse_response(prediction)

            function_calls_messages = [msg for msg in assistant_messages if msg.get("function_call")]
            if function_calls_messages:
                # Track function call success for this batch item
                total_calls = len(function_calls_messages)

                successful_calls = 0
                current_function_calls = []

                for assistant_msg in function_calls_messages:

                    name, arguments, tool_result = memory_agent_template._run_tool_from_function_call(assistant_msg["function_call"], batch_memories[i], return_arguments=True)

                    # Check if the function call was successful based on the result string
                    if "executed successfully" in tool_result and not "'status': 'skipped'" in tool_result:
                        successful_calls += 1

                    current_function_calls.append({
                        "name": name,
                        "arguments": arguments,
                        "result": tool_result,
                        'success': "executed successfully" in tool_result
                    })

                # Calculate reward as success rate
                if total_calls > 0:
                    success_rate = successful_calls / total_calls
                    function_call_rewards.append(success_rate)

                else:
                    function_call_rewards.append(0.0)

                all_function_calls.append(current_function_calls)

            else:
                # No function calls in this response
                function_call_rewards.append(0.0)
                all_function_calls.append([])

        # Add function call rewards to meta_info
        updated_meta_info = gen_output.meta_info.copy()
        updated_meta_info['function_call_rewards'] = function_call_rewards
        updated_meta_info['all_function_calls'] = all_function_calls

        return chunk_input_ids, responses_ids, response_mask, updated_meta_info

