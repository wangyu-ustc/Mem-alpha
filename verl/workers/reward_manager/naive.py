# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import torch
import numpy as np
import re
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from openai import OpenAI

# Add import for memory agent bench evaluation
from memalpha.llm_agent.metrics import evaluate_wrt_source, _extract_answer_from_response


SYSTEM_PROMPT = """
Your are a helpful assistant that can label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The input format is:
<Question>: {question}
<Gold answer>: {gold_answer}
<Generated answer>: {generated_answer}

You will need to label the generated answer as either CORRECT or WRONG based on the following criteria:

CORRECT: Label as CORRECT if and only if **the generated answer contains the same factual information as the gold answer** (Paraphrasing, rewording, or different phrasing is acceptable)

WRONG: Label as WRONG if:
- The generated answer provides different factual information (different dates, locations, names, numbers, etc.)
- The generated answer contradicts or conflicts with the gold answer
- The generated answer is completely unrelated to the question
- The generated answer is too vague or incomplete to verify correctness
- The generated answer says "I don't know" or equivalent when a gold answer exists

Response format: Provide a brief one-sentence explanation of your reasoning, then end with exactly <label>CORRECT</label> or <label>WRONG</label>.
CRITICAL: Use only ONE label in your response - never include both CORRECT and WRONG as this will break the evaluation.
"""

ACCURACY_PROMPT = """
Give me the label <label>CORRECT</label> or <label>WRONG</label> for the following question, gold answer, and generated answer.
<Question>: {question}
<Gold answer>: {gold_answer}
<Generated answer>: {generated_answer}
"""


def batch_process_questions_with_qwen32b(questions, batch_size=32, system_prompt=None, model="qwen3-32b", no_thinking=False, generative_reward=False):
    """
    Process a list of questions using Qwen32B model in batches

    Args:
        questions: List of questions to process
        batch_size: Number of questions to process in each batch
        model: Qwen model to use

    Returns:
        List of responses corresponding to each question
    """

    # Import and setup Qwen client
    from openai import OpenAI
    from transformers import AutoTokenizer
    import time

    # Setup Qwen client
    client = OpenAI(
        base_url=os.getenv("QWEN_URL"),
        api_key="EMPTY"
    )

    # Initialize tokenizer for prompt conversion
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    print(f"Starting batch processing of {len(questions)} questions with Qwen32B, batch size {batch_size}")

    all_responses = []

    # Process questions in batches
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(questions) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")

        # Convert all questions in batch to prompts
        batch_prompts = []
        for question in batch_questions:
            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]

            # Convert to prompt using tokenizer
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            if no_thinking:
                prompt += "<think></think>\n\n"

            batch_prompts.append(prompt)

        # Process the entire batch at once using completions API
        response = client.completions.create(
            model=model,
            prompt=batch_prompts,
            max_tokens=1024,
            temperature=0.0,
            stream=False
        )

        # Extract responses
        batch_responses = [choice.text for choice in response.choices]
        all_responses.extend(batch_responses)
        print(f"Completed batch {batch_num}/{total_batches}")
        # Delay between batches to avoid overloading the server
        if i + batch_size < len(questions):
            time.sleep(0.5)

    print(f"Batch processing complete. Generated {len(all_responses)} responses.")
    return all_responses


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", return_separate_scores=False, compression_ratio_weight=1.0,
                 function_content_reward_weight=1.0, generative_reward=False, threshold=None) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.return_separate_scores = return_separate_scores
        self.compression_ratio_weight = compression_ratio_weight
        self.function_content_reward_weight = function_content_reward_weight
        self.generative_reward = generative_reward
        self.threshold = threshold

        self.client = OpenAI(
            base_url=os.getenv("QWEN_URL"),
            api_key="EMPTY"
        )

    def _compute_score_for_data_source(self, data_source, predicted_answer, gold_answer, question=None):
        """Compute evaluation score based on data source using the same logic as long_context_eval.py."""

        # Handle thinking tags in predicted answer
        if "<think>" in predicted_answer and "</think>" in predicted_answer:
            predicted_answer = predicted_answer.split("</think>")[1].strip()
        if "<think>" in predicted_answer:
            predicted_answer = "Empty"

        if data_source == 'booksum':
            keywords = gold_answer.split(",")
            keywords = [x.strip() for x in keywords]
            hit = 0
            for keyword in keywords:
                if keyword.lower() in predicted_answer.lower():
                    hit += 1
            return hit / len(keywords)

        elif data_source == 'pubmed-rct' or 'ttl_train' in data_source or 'icl' in data_source:
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

        elif data_source == 'lme_train':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, gold_answer, predicted_answer)

            response = self.client.chat.completions.create(
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

        elif data_source in ['squad', 'hotpotqa']:
            # Default: containment score for QA datasets
            if isinstance(gold_answer, list):
                answer_text = str(gold_answer[0]) if gold_answer else ""
            else:
                answer_text = gold_answer.get('text', gold_answer) if isinstance(gold_answer, dict) else str(gold_answer)

            return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

        else:
            # Memory agent bench evaluation for other datasets
            return evaluate_wrt_source({'output': predicted_answer}, gold_answer, data_source)

    def __call__(self, data: DataProto, data_sources: list, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        predicted_answers_list = data.meta_info['predicted_answers_list']
        ground_truth_answers_list = data.meta_info['ground_truth_answers_list']
        total_chunk_length = data.meta_info['total_chunk_length']
        total_memory_length = data.meta_info['total_memory_length']
        questions_list = data.meta_info['questions_list']
        all_function_call_rewards = data.meta_info['all_function_call_rewards']
        all_function_call_content_rewards = data.meta_info['all_function_call_content_rewards'] if "all_function_call_content_rewards" in data.meta_info else None

        compression_ratio_reward_scores = [1 - memory_length / chunk_length for memory_length, chunk_length in zip(total_memory_length, total_chunk_length)]

        reward_scores = []

        # First, collect all prompts that need to be processed by qwen32b (only when generative_reward is True)
        all_qwen_prompts = []
        qwen_prompt_mapping = []  # Track which batch item and question index each prompt belongs to

        if self.generative_reward:
            for i in range(len(ground_truth_answers_list)):
                if data_sources[i] in ['squad', 'hotpotqa']:
                    # Collect prompts for this batch item
                    for j, (question, predicted_answer, ground_truth_answer) in enumerate(zip(questions_list[i], predicted_answers_list[i], ground_truth_answers_list[i])):
                        prompt = ACCURACY_PROMPT.format(question=question, gold_answer=ground_truth_answer, generated_answer=predicted_answer)
                        all_qwen_prompts.append(prompt)
                        qwen_prompt_mapping.append((i, j))  # Store batch item index and question index

        # Process all qwen prompts in a single batch call if there are any
        if all_qwen_prompts:
            print(f"Processing {len(all_qwen_prompts)} prompts with qwen32b in a single batch call")
            all_qwen_responses = batch_process_questions_with_qwen32b(all_qwen_prompts, batch_size=1024, no_thinking=True, generative_reward=self.generative_reward)
        else:
            all_qwen_responses = []

        # Create a mapping from batch item index to qwen responses
        qwen_responses_by_batch_item = defaultdict(list)
        for response, (batch_idx, question_idx) in zip(all_qwen_responses, qwen_prompt_mapping):
            qwen_responses_by_batch_item[batch_idx].append(response)

        # Now process each batch item and calculate reward scores
        for i in range(len(ground_truth_answers_list)):
            data_source = data_sources[i]

            # Handle special case for generative reward on squad/hotpotqa
            if data_source in ['squad', 'hotpotqa'] and self.generative_reward:
                # Use the pre-computed qwen32b responses
                batch_responses = qwen_responses_by_batch_item[i]
                all_scores = []
                for response in batch_responses:
                    if "<label>CORRECT</label>" in response and "<label>WRONG</label>" in response:
                        score = 0  # Default to wrong if both tags present
                    elif "<label>CORRECT</label>" in response:
                        score = 1
                    elif "<label>WRONG</label>" in response:
                        score = 0
                    else:
                        score = 0  # Default to wrong if we can't parse
                    all_scores.append(score)
                reward_scores.append(np.mean(all_scores))
            else:
                # Use the comprehensive scoring method for all other cases
                all_scores = []
                for pred, answer, question in zip(predicted_answers_list[i], ground_truth_answers_list[i], questions_list[i]):
                    # Use the new comprehensive scoring method
                    score = self._compute_score_for_data_source(data_source, pred, answer, question)
                    all_scores.append(score)
                reward_scores.append(np.mean(all_scores))

        # Save original reward scores:
        original_reward_scores = reward_scores
        original_compression_ratio_reward_scores = compression_ratio_reward_scores
        original_all_function_call_rewards = all_function_call_rewards
        original_all_function_call_content_rewards = all_function_call_content_rewards

        if not self.return_separate_scores:
            # meaning we are doing training
            acc_reward_scores = reward_scores
            if self.threshold is not None:
                reward_scores = [0 if r < self.threshold else 1 for r in reward_scores]
            reward_scores = [r + self.compression_ratio_weight * c for r, c in zip(reward_scores, compression_ratio_reward_scores)]

        all_reward_scores = []
        for i in data.meta_info['indices_in_batch']:
            all_reward_scores.append(reward_scores[i])

        if not self.return_separate_scores:
            if all_function_call_content_rewards is not None:
                all_reward_scores = [r + f + (c * self.function_content_reward_weight) for r, f, c in zip(all_reward_scores, all_function_call_rewards, all_function_call_content_rewards)]
            else:
                all_reward_scores = [r + f for r, f in zip(all_reward_scores, all_function_call_rewards)]

        for i in range(len(all_reward_scores)):

            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            # valid_response_ids = response_ids[:valid_response_length]
            reward_tensor[i, valid_response_length - 1] = all_reward_scores[i]

        if not self.return_separate_scores:
            # Training
            if return_dict:
                # return {"reward_tensor": reward_tensor, "reward_extra_info": {"acc_reward_scores": acc_reward_scores, "compression_ratio_reward_scores": np.mean(compression_ratio_reward_scores)}}
                return {"reward_tensor": reward_tensor, "reward_extra_info": {"acc_reward_scores": original_reward_scores,
                    "compression_ratio_reward_scores": original_compression_ratio_reward_scores,
                    "all_function_call_rewards": original_all_function_call_rewards,
                    "all_function_call_content_rewards": original_all_function_call_content_rewards}}
            else:
                return reward_tensor
        else:
            if return_dict:
                if all_function_call_content_rewards is not None:
                    return {"reward_tensor": reward_tensor, "reward_extra_info": {"compression_ratio_reward_scores": np.mean(compression_ratio_reward_scores), "all_function_call_rewards": all_function_call_rewards, "all_function_call_content_rewards": all_function_call_content_rewards}}
                else:
                    return {"reward_tensor": reward_tensor, "reward_extra_info": {"compression_ratio_reward_scores": np.mean(compression_ratio_reward_scores), "all_function_call_rewards": all_function_call_rewards}}
            else:
                if all_function_call_content_rewards is not None:
                    return reward_tensor, compression_ratio_reward_scores, all_function_call_rewards, all_function_call_content_rewards
                else:
                    return reward_tensor, compression_ratio_reward_scores, all_function_call_rewards


        # reward_extra_info = defaultdict(list)

        # already_print_data_sources = {}

        # for i in range(len(data)):
        #     data_item = data[i]  # DataProtoItem

        #     prompt_ids = data_item.batch["prompts"]

        #     prompt_length = prompt_ids.shape[-1]

        #     valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        #     response_ids = data_item.batch["responses"]
        #     valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        #     valid_response_ids = response_ids[:valid_response_length]

        #     # decode
        #     prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        #     response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        #     ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        #     data_source = data_item.non_tensor_batch[self.reward_fn_key]
        #     extra_info = data_item.non_tensor_batch.get("extra_info", None)

        #     score = self.compute_score(
        #         data_source=data_source,
        #         solution_str=response_str,
        #         ground_truth=ground_truth,
        #         extra_info=extra_info,
        #     )

        #     if isinstance(score, dict):
        #         reward = score["score"]
        #         # Store the information including original reward
        #         for key, value in score.items():
        #             reward_extra_info[key].append(value)
        #     else:
        #         reward = score

        #     reward_tensor[i, valid_response_length - 1] = reward

        #     if data_source not in already_print_data_sources:
        #         already_print_data_sources[data_source] = 0

        #     if already_print_data_sources[data_source] < self.num_examine:
        #         already_print_data_sources[data_source] += 1
        #         print("[prompt]", prompt_str)
        #         print("[response]", response_str)
        #         print("[ground_truth]", ground_truth)
        #         if isinstance(score, dict):
        #             for key, value in score.items():
        #                 print(f"[{key}]", value)
        #         else:
        #             print("[score]", score)

        # if return_dict:
        #     return {
        #         "reward_tensor": reward_tensor,
        #         "reward_extra_info": reward_extra_info,
        #     }
        # else:
        #     return reward_tensor
