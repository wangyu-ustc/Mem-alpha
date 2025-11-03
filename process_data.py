# Standard library imports
import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List, Any, Tuple

# Third-party imports
import dotenv
import nltk
import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset
from openai import AzureOpenAI
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

# Azure OpenAI configuration
api_key = os.getenv("AZURE_OPENAI_API_KEY")
client = AzureOpenAI(
    api_key=api_key,
    api_version="2025-01-01-preview",
    azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
)

qwen32b_server_url = os.getenv("QWEN_URL")

# User message templates for conversational format
USER_MESSAGE_TEMPLATES = [
    "Here are some new facts I learnt just now:",
    "Here are some news I learnt earlier:",
    "The following are some more news I learnt:",
    "The following are some new knowledge:",
    "I just discovered some interesting information:",
    "Let me share some additional facts I found:",
    "Here's some more information I came across:",
    "I want to tell you about some new discoveries:",
    "There are some important details I learned:",
    "I have some fresh insights to share:",
    "I'd like to tell you about some recent findings:",
    "Let me update you with some new information:",
    "I've gathered some additional data to share:",
    "Here's some valuable information I collected:",
    "I want to share some knowledge I just acquired:",
    "Let me provide you with some new details:",
    "I have some interesting updates for you:",
    "Here are some facts I recently discovered:",
    "I'd like to share some important information:",
    "Let me tell you about some new developments:"
]

# User message templates for classification tasks
CLASSIFICATION_USER_TEMPLATES = [
    "Here are some classification examples to learn from. Please pay attention to the labels:",
    "I have some labeled classification examples for you to study:",
    "The following are classification examples with their corresponding labels:",
    "Please observe these classification examples and their associated labels:",
    "Here are training examples for classification. Note the labels carefully:",
    "I'm sharing some classification data with labels for you to learn:",
    "These are labeled examples for classification tasks:",
    "Please study these classification instances and their labels:",
    "Here are some examples with classification labels to remember:",
    "The following classification examples include important label information:",
    "I want you to learn from these classified examples:",
    "Here are categorized examples with their respective labels:",
    "Please memorize these classification examples and their labels:",
    "These labeled training examples are for classification:",
    "I'm providing classification data with labels for your reference:",
    "Study these classification examples and pay attention to the categories:",
    "Here are some annotated classification examples:",
    "Please learn from these labeled classification instances:",
    "The following are classification training examples with labels:",
    "These examples show different classes - please note the labels:"
]

# Assistant response templates
ASSISTANT_RESPONSE_TEMPLATES = [
    "Sure I will remember them.",
    "Got it. I will remember them.",
    "Thank you for sharing. I've noted this information.",
    "I understand. I'll keep this in mind.",
    "Thanks for the update. I've recorded these facts.",
    "Received. I'll store this information.",
    "Noted. I'll remember these details.",
    "I've processed this information and will remember it.",
    "Thanks for letting me know. I'll keep track of this.",
    "Perfect. I've stored this information in my memory.",
    "Understood. I'll keep these facts for future reference.",
    "Excellent. I've documented all of this information.",
    "Thanks for the information. I've saved it.",
    "Appreciated. I'll retain these important details.",
    "Great! I've added this to my knowledge base.",
    "I've successfully recorded all of this data.",
    "Wonderful. I'll remember these key points.",
    "Thanks for sharing. I've committed this to memory.",
    "I've captured all of this valuable information."
]

ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The input format is:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""

def judge_answer_with_token_logic(ground_truth_answer, predicted_answer, debug=False):
    """
    Judge answer based on token count and string containment logic

    Args:
        ground_truth_answer: The expected correct answer
        predicted_answer: The model's predicted answer
        debug: If True, print debug information

    Returns:
        int: 0 if we should keep the example (judge as incorrect), 1 if we should remove it
    """
    # Convert to strings and handle edge cases
    ground_truth = str(ground_truth_answer).strip()
    predicted = str(predicted_answer).strip()

    if not ground_truth or not predicted:
        if debug:
            print(f"Empty answer detected - keeping example (GT: '{ground_truth}', Pred: '{predicted}')")
        return 0  # Keep examples with empty answers for safety

    # Count tokens in ground truth answer
    ground_truth_tokens = count_tokens(ground_truth)

    # Check condition (1): ground truth answer is less than 5 tokens
    condition_1 = ground_truth_tokens < 5

    # Check condition (2): ground truth answer (lowercase) is NOT in predicted answer (lowercase)
    condition_2 = ground_truth.lower() not in predicted.lower()

    # If both conditions are satisfied, set judge as 0 (keep the example)
    if condition_1 and condition_2:
        if debug:
            print(f"KEEP: Short answer ({ground_truth_tokens} tokens) not found in prediction")
            print(f"  GT: '{ground_truth}'")
            print(f"  Pred: '{predicted[:100]}...'")
        return 0  # Keep the example
    else:
        if debug:
            reason = []
            if not condition_1:
                reason.append(f"answer too long ({ground_truth_tokens} >= 5 tokens)")
            if not condition_2:
                reason.append("answer found in prediction")
            print(f"REMOVE: {', '.join(reason)}")
            print(f"  GT: '{ground_truth}'")
            print(f"  Pred: '{predicted[:100]}...'")
        return 1  # Remove the example


# VARIABLE CHUNK SIZE FEATURE:
# For SQuAD, HotpotQA, WOS46985, PubMed-RCT, ArXiv-Classification, and EurLex datasets,
# chunks now have variable sizes between 100 and 4096 tokens instead of fixed sizes around 2000 tokens.
# This is controlled by the variable_size parameter in chunking functions.
# Booksum dataset is excluded from this feature as requested.

def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)

    # Convert input to string if it's not already a string
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        else:
            text = str(text)

    return len(encoding.encode(text))

def create_chunks_use_sent_tokenizer(text, max_tokens=10000):
    """Create chunks from text using sentence tokenization"""
    # Make sure we have the punkt tokenizer downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:

        if '<|endoftext|>' in sentence:
            sentence = sentence.replace('<|endoftext|>', '\n')

        sentence_tokens = count_tokens(sentence)

        # If adding this sentence would exceed max_tokens, start a new chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            if current_chunk:
                # Add space between sentences
                current_chunk += " " + sentence
                current_tokens += sentence_tokens + count_tokens(" ")
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def create_chunks(contexts, max_tokens=2000, min_tokens=None, variable_size=False):
    """Create chunks from contexts, ensuring each chunk is less than max_tokens

    Args:
        contexts: List of context strings to chunk
        max_tokens: Maximum tokens per chunk (default: 2000)
        min_tokens: Minimum tokens per chunk (used when variable_size=True, default: max_tokens/20)
        variable_size: If True, randomly vary chunk size between min_tokens and max_tokens for each chunk
    """
    chunks = []
    current_chunk = ""
    current_tokens = 0

    # Set default min_tokens if not provided and variable_size is enabled
    if variable_size and min_tokens is None:
        min_tokens = max(100, max_tokens // 20)

    # Set target tokens for first chunk
    if variable_size:
        target_tokens = random.randint(min_tokens, max_tokens)
    else:
        target_tokens = max_tokens

    for context in contexts:
        context_tokens = count_tokens(context)

        # If adding this context would exceed target_tokens, start a new chunk
        if current_tokens + context_tokens > target_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = context
            current_tokens = context_tokens

            # Set new target for next chunk if using variable size
            if variable_size:
                target_tokens = random.randint(min_tokens, max_tokens)
            else:
                target_tokens = max_tokens
        else:
            if current_chunk:
                current_chunk += "\n\n" + context
                current_tokens += context_tokens + count_tokens("\n\n")
            else:
                current_chunk = context
                current_tokens = context_tokens

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def batch_process_questions_with_qwen32b(questions, batch_size=32, system_prompt=None, model="qwen3-32b", no_thinking=False):
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
        base_url=qwen32b_server_url,
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
            temperature=0.7,
            stream=False
        )

        # Extract responses
        batch_responses = [choice.text for choice in response.choices]

        if not no_thinking:
            # need to remove the <think></think> tags
            batch_responses = [(x.split("</think>")[1] if "</think>" in x else x) for x in batch_responses]

        all_responses.extend(batch_responses)
        print(f"Completed batch {batch_num}/{total_batches}")
        # Delay between batches to avoid overloading the server
        if i + batch_size < len(questions):
            time.sleep(0.5)

    print(f"Batch processing complete. Generated {len(all_responses)} responses.")
    return all_responses


def process_squad_dataset(split='train', num_chunks=10, max_chunks_allowed=20):
    """Process SQuAD dataset into the desired format"""

    if not os.path.exists("./data/squad/raw_instances.json"):
        ds = load_dataset("rajpurkar/squad")
        context_to_qas = defaultdict(list)

        for item in ds[split]:
            context = item['context']
            qa = {
                'question': item['question'],
                'answer': item['answers']['text'][0],
            }
            context_to_qas[context].append(qa)

        print(f"Found {len(context_to_qas)} unique contexts")
        # Get unique contexts
        unique_contexts = list(context_to_qas.keys())

        chunks = create_chunks(unique_contexts, max_tokens=2048, min_tokens=100, variable_size=True)
        # Print chunk size statistics
        chunk_sizes = [count_tokens(chunk) for chunk in chunks]
        print(f"Chunk size statistics - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.1f}")

        # Create data instances with 10 chunks each
        print("Creating data instances...")
        processed_data = []

        # Import datetime for timestamp generation

        base_date = datetime(2024, 1, 1)

        for i in range(0, len(chunks), num_chunks):
            # Get 10 chunks (or remaining chunks if less than 10)
            chunk_batch_raw = chunks[i:i+num_chunks]

            # Format chunks with conversational templates
            chunk_batch = []
            for chunk_idx, chunk_content in enumerate(chunk_batch_raw):
                # Format using conversational template with random selection
                user_template = random.choice(USER_MESSAGE_TEMPLATES)
                assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

                # Create a timestamp for each chunk (incrementing by days)
                chunk_date = base_date + timedelta(days=chunk_idx)
                timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

                # Format the chunk with conversational template
                formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n{chunk_content}\n<Assistant>{assistant_template}"
                chunk_batch.append(formatted_chunk)

            # Collect all questions and answers for these chunks
            all_qas = []
            for chunk_idx, chunk in enumerate(chunk_batch_raw):  # Use raw chunks for Q&A matching
                # For each chunk, find all contexts within it and collect their Q&As
                for context in unique_contexts:
                    if context in chunk:
                        cur_qas = context_to_qas[context]
                        for qa in cur_qas:
                            qa['evidence_idx'] = chunk_idx
                        all_qas.extend(cur_qas)

            # filter out oversized questions
            all_qas = [qa for qa in all_qas if count_tokens(qa['question']) < 2048]

            # Filter out instances with less than 50 questions
            if len(all_qas) < 50:
                print(f"Skipping instance with only {len(all_qas)} questions (< 50)")
                continue

            # Create the data instance
            data_instance = {
                'prompt': 'I will provide you with sequential information chunks. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
                'chunks': chunk_batch,
                'questions_and_answers': all_qas,
                'data_source': 'squad',
            }

            processed_data.append(data_instance)

            print(f"Created instance {len(processed_data)} with {len(chunk_batch)} chunks and {len(all_qas)} Q&As")

        with open("./data/squad/raw_instances.json", "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    else:
        with open("./data/squad/raw_instances.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)

    question_to_groundtruth_answer = {}
    for instance in processed_data:
        for qa in instance['questions_and_answers']:
            question_to_groundtruth_answer[qa['question']] = qa['answer']

    if not os.path.exists("./data/qwen32b-answers/squad_results.json"):
        # Filter questions to only process those with short answers (< 5 tokens) to save compute
        print("Pre-filtering questions with short answers (< 5 tokens) to save compute...")
        all_questions = []
        question_to_answer = {}
        original_count = 0

        for instance in processed_data:
            for qa in instance['questions_and_answers']:
                original_count += 1
                answer_tokens = count_tokens(str(qa['answer']).strip())
                if answer_tokens < 5:
                    all_questions.append(qa['question'])
                    question_to_answer[qa['question']] = qa['answer']

        print(f"Filtered questions: {original_count} → {len(all_questions)} ({len(all_questions)/original_count*100:.1f}% kept)")
        print(f"Saved {original_count - len(all_questions)} API calls by pre-filtering long answers")

        # Batch process filtered questions using Qwen32B API
        batch_responses = batch_process_questions_with_qwen32b(all_questions,
            batch_size=1024,
            system_prompt="You are a helpful assistant. Answer the question as accurately as possible. Be brief, only output the answer without any other text.", no_thinking=False)
        assert len(batch_responses) == len(all_questions)

        # Create results for all questions (including filtered ones)
        results = []
        processed_questions = set(all_questions)

        for instance in processed_data:
            for qa in instance['questions_and_answers']:
                question = qa['question']
                if question in processed_questions:
                    # Question was processed - use API response
                    question_idx = all_questions.index(question)
                    results.append({'question': question, 'answer': batch_responses[question_idx]})
                else:
                    # Question was filtered out - use placeholder to indicate it wasn't processed
                    results.append({'question': question, 'answer': '[FILTERED_LONG_ANSWER]'})

        # Save batch results for analysis
        os.makedirs('./data/qwen32b-answers', exist_ok=True)
        with open('./data/qwen32b-answers/squad_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Batch processing results saved to ./data/qwen32b-answers/squad_results.json")

    else:
        with open("./data/qwen32b-answers/squad_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)

    if not os.path.exists("./data/qwen32b-answers/squad_results_with_judge.json"):

        # Use token-based logic instead of LLM judge
        question_to_predicted_answer = {qa['question']: qa['answer'] for qa in results}

        question_to_score = {}
        keep_count = 0
        remove_count = 0
        short_answer_count = 0

        print("Applying new token-based filtering logic...")
        for q in question_to_groundtruth_answer:
            ground_truth = question_to_groundtruth_answer[q]
            predicted = question_to_predicted_answer[q]

            # Count short answers for statistics
            if count_tokens(str(ground_truth).strip()) < 5:
                short_answer_count += 1

            # Handle filtered questions (long answers that weren't processed)
            if predicted == '[FILTERED_LONG_ANSWER]':
                score = 1  # Remove questions with long answers automatically
            else:
                # Use our new token logic: 0 means keep (incorrect), 1 means remove (correct)
                score = judge_answer_with_token_logic(ground_truth, predicted)

            question_to_score[q] = score

            if score == 0:
                keep_count += 1
            else:
                remove_count += 1

        total_questions = len(question_to_groundtruth_answer)
        filtered_questions = len([q for q in question_to_predicted_answer.values() if q == '[FILTERED_LONG_ANSWER]'])
        print(f"Token-based filtering results for SQuAD:")
        print(f"  Total questions: {total_questions}")
        print(f"  Pre-filtered (long answers): {filtered_questions} ({filtered_questions/total_questions*100:.1f}%)")
        print(f"  Short answers (< 5 tokens): {short_answer_count} ({short_answer_count/total_questions*100:.1f}%)")
        print(f"  Questions to keep: {keep_count} ({keep_count/total_questions*100:.1f}%)")
        print(f"  Questions to remove: {remove_count} ({remove_count/total_questions*100:.1f}%)")

        # Save the predicted results to "results"
        for idx, result in enumerate(results):
            result['score'] = question_to_score[result['question']]

        # Save the results to "results"
        with open('./data/qwen32b-answers/squad_results_with_judge.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Batch processing results saved to ./data/squad_results_with_judge.json")

    else:
        with open("./data/qwen32b-answers/squad_results_with_judge.json", "r", encoding="utf-8") as f:
            results = json.load(f)

    # Now start filtering the data
    question_to_score = {qa['question']: qa['score'] for qa in results}
    for idx, instance in enumerate(processed_data):
        original_len = len(instance['questions_and_answers'])
        filtered_qas = []
        for qa in instance['questions_and_answers']:
            # Use .get() to handle missing questions gracefully
            if question_to_score.get(qa['question'], 1) == 0:  # Default to 1 if question not found
                filtered_qas.append(qa)
        if len(filtered_qas) > 100:
            filtered_qas = random.sample(filtered_qas, 100)
        instance['questions_and_answers'] = filtered_qas
        print(f"Filtered out {original_len - len(filtered_qas)} out of {original_len} questions for Instance {idx}")

    print(f"Obtained {len(processed_data)} instances")
    processed_data = [f for f in processed_data if len(f['questions_and_answers']) >= 30]
    print(f"Filtered to {len(processed_data)} instances with more than 30 questions")

    # Filter out instances with too many chunks (over 20)
    max_chunks_allowed = max_chunks_allowed
    filtered_instances = []
    instances_with_too_many_chunks = 0

    for instance in processed_data:
        num_chunks = len(instance['chunks'])
        if num_chunks <= max_chunks_allowed:
            filtered_instances.append(instance)
        else:
            instances_with_too_many_chunks += 1
            print(f"Removing SQuAD instance with {num_chunks} chunks (> {max_chunks_allowed})")

    print(f"\n📊 SQuAD Chunk Filtering Summary:")
    print(f"  • Removed {instances_with_too_many_chunks} instances with > {max_chunks_allowed} chunks")
    if processed_data:
        print(f"  • Instances: {len(processed_data):,} → {len(filtered_instances):,} (kept {len(filtered_instances)/len(processed_data)*100:.1f}%)")
    else:
        print(f"  • Instances: 0 → 0")

    return filtered_instances

def process_hotpotqa_dataset():
    """Process HotpotQA dataset into the desired format"""

    os.makedirs("./data/hotpotqa", exist_ok=True)

    if not os.path.exists("./data/hotpotqa/processed_data.json"):
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki")
        print(f"HotpotQA dataset loaded with {len(ds['train'])} training examples")

        unique_articles = {}  # title -> full_text
        questions_data = []

        for item in ds['train']:
            # Extract context information
            titles = item['context']['title']
            sentences_lists = item['context']['sentences']

            # Create unique articles
            for title, sentences in zip(titles, sentences_lists):
                if title not in unique_articles:
                    # Combine all sentences for this title into one article
                    full_text = ' '.join(sentences)
                    unique_articles[title] = full_text

            # Extract question with evidence requirements
            question_data = {
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'evidence_requirements': []  # list of (title, sentence_text) tuples
            }

            # Extract supporting facts (evidence)
            for support_title, sent_id in zip(item['supporting_facts']['title'], item['supporting_facts']['sent_id']):
                # Find the sentence text
                title_idx = titles.index(support_title) if support_title in titles else -1
                if title_idx >= 0 and sent_id < len(sentences_lists[title_idx]):
                    sentence_text = sentences_lists[title_idx][sent_id]
                    question_data['evidence_requirements'].append((support_title, sentence_text))

            questions_data.append(question_data)

        print(f"Found {len(unique_articles)} unique articles")
        print(f"Found {len(questions_data)} questions")

        print("Calculating article statistics...")
        article_tokens = []
        for title, text in unique_articles.items():
            tokens = count_tokens(f"Title: {title}\n{text}")
            article_tokens.append(tokens)

        avg_article_tokens = sum(article_tokens) / len(article_tokens)
        print(f"Average article length: {avg_article_tokens:.1f} tokens")

        # Target: ~20k tokens per data instance, with chunks of ~2k tokens each (so ~10 chunks)
        # Calculate how many articles we can fit in one data instance
        target_tokens_per_instance = 10000
        k = max(1, int(target_tokens_per_instance / avg_article_tokens))
        print(f"Target articles per data instance: {k}")

        # Step 3: Process questions sequentially and group articles
        print("Processing questions and grouping articles...")
        processed_data = []
        current_articles = []  # list of titles
        current_questions = []  # list of question_data

        for question in questions_data:
            # Extract evidence titles for this question
            evidence_titles = set(title for title, _ in question['evidence_requirements'])

            # Check if we need to add new articles
            new_articles_needed = evidence_titles - set(current_articles)

            # If adding new articles would exceed k, process current batch
            if current_articles and len(current_articles) + len(new_articles_needed) > k:
                # Process current batch
                if current_articles and current_questions:  # Make sure we have both articles and questions
                    data_instance = create_data_instance(current_articles, current_questions, unique_articles)
                    if data_instance:  # Only add if successfully created
                        processed_data.append(data_instance)
                        print(f"Created instance {len(processed_data)} with {len(current_articles)} articles and {len(current_questions)} questions")

                # Start new batch with current question's requirements
                current_articles = list(evidence_titles)
                current_questions = [question]
            else:
                # Add new articles to current batch
                current_articles.extend(new_articles_needed)
                current_questions.append(question)

        # Process the last batch
        if current_articles and current_questions:
            data_instance = create_data_instance(current_articles, current_questions, unique_articles)
            if data_instance:
                processed_data.append(data_instance)
                print(f"Created instance {len(processed_data)} with {len(current_articles)} articles and {len(current_questions)} questions")

        with open("./data/hotpotqa/processed_data.json", "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

    else:
        with open("./data/hotpotqa/processed_data.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)

    print(f"Obtained {len(processed_data)} instances")
    processed_data = [f for f in processed_data if len(f['questions_and_answers']) >= 30]
    print(f"Filtered to {len(processed_data)} instances with more than 30 questions")

    question_to_groundtruth_answer = {}
    for instance in processed_data:
        for qa in instance['questions_and_answers']:
            question_to_groundtruth_answer[qa['question']] = qa['answer']

    if not os.path.exists("./data/qwen32b-answers/hotpotqa_results.json"):
        # Filter questions to only process those with short answers (< 5 tokens) to save compute
        print("Pre-filtering questions with short answers (< 5 tokens) to save compute...")
        all_questions = []
        question_to_answer = {}
        original_count = 0

        for instance in processed_data:
            for qa in instance['questions_and_answers']:
                original_count += 1
                answer_tokens = count_tokens(str(qa['answer']).strip())
                if answer_tokens < 5:
                    all_questions.append(qa['question'])
                    question_to_answer[qa['question']] = qa['answer']

        print(f"Filtered questions: {original_count} → {len(all_questions)} ({len(all_questions)/original_count*100:.1f}% kept)")
        print(f"Saved {original_count - len(all_questions)} API calls by pre-filtering long answers")

        batch_responses = batch_process_questions_with_qwen32b(all_questions, batch_size=1024, system_prompt="You are a helpful assistant. Answer the question as accurately as possible. Be brief, only output the answer without any other text.", no_thinking=False)

        # Create results for all questions (including filtered ones)
        results = []
        processed_questions = set(all_questions)

        for instance in processed_data:
            for qa in instance['questions_and_answers']:
                question = qa['question']
                if question in processed_questions:
                    # Question was processed - use API response
                    question_idx = all_questions.index(question)
                    results.append({'question': question, 'answer': batch_responses[question_idx]})
                else:
                    # Question was filtered out - use placeholder to indicate it wasn't processed
                    results.append({'question': question, 'answer': '[FILTERED_LONG_ANSWER]'})

        with open("./data/qwen32b-answers/hotpotqa_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Batch processing results saved to ./data/qwen32b-answers/hotpotqa_results.json")
    else:
        with open("./data/qwen32b-answers/hotpotqa_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)

    if not os.path.exists("./data/qwen32b-answers/hotpotqa_results_with_judge.json"):

        # Use token-based logic instead of LLM judge
        question_to_predicted_answer = {qa['question']: qa['answer'] for qa in results}

        question_to_score = {}
        keep_count = 0
        remove_count = 0
        short_answer_count = 0

        print("Applying new token-based filtering logic...")
        for q in question_to_groundtruth_answer:
            ground_truth = question_to_groundtruth_answer[q]
            predicted = question_to_predicted_answer[q]

            # Count short answers for statistics
            if count_tokens(str(ground_truth).strip()) < 5:
                short_answer_count += 1

            # Handle filtered questions (long answers that weren't processed)
            if predicted == '[FILTERED_LONG_ANSWER]':
                score = 1  # Remove questions with long answers automatically
            else:
                # Use our new token logic: 0 means keep (incorrect), 1 means remove (correct)
                score = judge_answer_with_token_logic(ground_truth, predicted)

            question_to_score[q] = score

            if score == 0:
                keep_count += 1
            else:
                remove_count += 1

        total_questions = len(question_to_groundtruth_answer)
        filtered_questions = len([q for q in question_to_predicted_answer.values() if q == '[FILTERED_LONG_ANSWER]'])
        print(f"Token-based filtering results for HotpotQA:")
        print(f"  Total questions: {total_questions}")
        print(f"  Pre-filtered (long answers): {filtered_questions} ({filtered_questions/total_questions*100:.1f}%)")
        print(f"  Short answers (< 5 tokens): {short_answer_count} ({short_answer_count/total_questions*100:.1f}%)")
        print(f"  Questions to keep: {keep_count} ({keep_count/total_questions*100:.1f}%)")
        print(f"  Questions to remove: {remove_count} ({remove_count/total_questions*100:.1f}%)")

        for idx, result in enumerate(results):
            result['score'] = question_to_score[result['question']]

        with open("./data/qwen32b-answers/hotpotqa_results_with_judge.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("Batch processing results saved to ./data/hotpotqa_results_with_judge.json")
    else:
        with open("./data/qwen32b-answers/hotpotqa_results_with_judge.json", "r", encoding="utf-8") as f:
            results = json.load(f)

    # Now start filtering the data
    question_to_score = {qa['question']: qa['score'] for qa in results}
    for idx, instance in enumerate(processed_data):
        filtered_qas = []
        original_len = len(instance['questions_and_answers'])
        for qa in instance['questions_and_answers']:
            if question_to_score[qa['question']] == 0:
                filtered_qas.append(qa)
        if len(filtered_qas) > 100:
            filtered_qas = random.sample(filtered_qas, 100)
        instance['questions_and_answers'] = filtered_qas
        print(f"Filtered out {original_len - len(filtered_qas)} out of {original_len} questions for Instance {idx}")

    print(f"Obtained {len(processed_data)} instances")
    processed_data = [f for f in processed_data if len(f['questions_and_answers']) >= 10 and len(f['questions_and_answers']) <= 200]
    print(f"Filtered to {len(processed_data)} instances with 10-200 questions")

    # Filter out instances with too many chunks (over 20)
    max_chunks_allowed = 20
    filtered_instances = []
    instances_with_too_many_chunks = 0

    for instance in processed_data:
        num_chunks = len(instance['chunks'])
        if num_chunks <= max_chunks_allowed:
            filtered_instances.append(instance)
        else:
            instances_with_too_many_chunks += 1
            print(f"Removing HotpotQA instance with {num_chunks} chunks (> {max_chunks_allowed})")

    print(f"\n📊 HotpotQA Chunk Filtering Summary:")
    print(f"  • Removed {instances_with_too_many_chunks} instances with > {max_chunks_allowed} chunks")
    if processed_data:
        print(f"  • Instances: {len(processed_data):,} → {len(filtered_instances):,} (kept {len(filtered_instances)/len(processed_data)*100:.1f}%)")
    else:
        print(f"  • Instances: 0 → 0")

    return filtered_instances

def create_data_instance(article_titles, questions, unique_articles):
    """Create a data instance from a list of article titles and questions"""

    # Get article texts
    article_texts = []
    for title in article_titles:
        if title in unique_articles:
            article_texts.append(unique_articles[title])
        else:
            print(f"Warning: Article '{title}' not found in unique_articles")

    if not article_texts:
        return None

    # Create chunks from articles with variable sizes between 100 and 4k tokens
    chunks_with_titles = create_chunks_with_titles(article_titles, article_texts, max_tokens=2048, min_tokens=100, variable_size=True)

    if not chunks_with_titles:
        return None

    # Import datetime for timestamp generation
    base_date = datetime(2024, 1, 1)

    # Extract chunk texts and format with conversational templates
    chunk_texts = []
    chunk_title_mapping = []  # List of lists, each containing titles in that chunk

    for chunk_idx, (chunk_text, titles_in_chunk) in enumerate(chunks_with_titles):
        # Format using conversational template with random selection
        user_template = random.choice(USER_MESSAGE_TEMPLATES)
        assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

        # Create a timestamp for each chunk (incrementing by days)
        chunk_date = base_date + timedelta(days=chunk_idx)
        timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

        # Format the chunk with conversational template
        formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n{chunk_text}\n<Assistant>{assistant_template}"

        chunk_texts.append(formatted_chunk)
        chunk_title_mapping.append(titles_in_chunk)

    # Print chunk size statistics for this instance
    chunk_sizes = [count_tokens(chunk_text) for chunk_text in chunk_texts]
    print(f"Instance chunks - Count: {len(chunk_texts)}, Sizes: Min={min(chunk_sizes)}, Max={max(chunk_sizes)}, Avg={sum(chunk_sizes)/len(chunk_sizes):.1f}")

    # Format questions and answers with evidence_idx
    formatted_qas = []
    for q in questions:
        # Find which chunk(s) contain the evidence for this question
        evidence_chunk_indices = set()

        for evidence_title, evidence_sentence in q['evidence_requirements']:
            # Find which chunk contains this evidence title
            for chunk_idx, titles_in_chunk in enumerate(chunk_title_mapping):
                if evidence_title in titles_in_chunk:
                    evidence_chunk_indices.add(chunk_idx)
                    break

        # Convert to sorted list for consistency
        evidence_idx = sorted(list(evidence_chunk_indices)) if evidence_chunk_indices else [0]  # Default to first chunk if no evidence found

        qa = {
            'question': q['question'],
            'answer': q['answer'],
            'evidence': q['evidence_requirements'],
            'evidence_idx': evidence_idx  # Add evidence chunk indices
        }
        formatted_qas.append(qa)

    # Create the data instance
    data_instance = {
        'prompt': 'I will provide you with sequential information chunks. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
        'chunks': chunk_texts,
        'questions_and_answers': formatted_qas
    }

    return data_instance

def create_chunks_with_titles(titles, texts, max_tokens=2000, min_tokens=None, variable_size=False):
    """Create chunks from articles, keeping track of which titles are in each chunk

    Args:
        titles: List of article titles
        texts: List of article texts
        max_tokens: Maximum tokens per chunk (default: 2000)
        min_tokens: Minimum tokens per chunk (used when variable_size=True, default: max_tokens/20)
        variable_size: If True, randomly vary chunk size between min_tokens and max_tokens for each chunk
    """
    chunks_with_titles = []  # list of (chunk_text, [titles_in_chunk])
    current_chunk = ""
    current_titles = []
    current_tokens = 0

    # Set default min_tokens if not provided and variable_size is enabled
    if variable_size and min_tokens is None:
        min_tokens = max(100, max_tokens // 20)

    # Set target tokens for first chunk
    if variable_size:
        target_tokens = random.randint(min_tokens, max_tokens)
    else:
        target_tokens = max_tokens

    for title, text in zip(titles, texts):
        text_with_title = f"Title: {title}\n{text}"
        text_tokens = count_tokens(text_with_title)

        # CRITICAL FIX: Truncate oversized articles to prevent massive chunks
        if text_tokens > target_tokens:
            # Truncate text to fit within target_tokens (reserve ~100 tokens for title)
            max_text_chars = min(len(text), (target_tokens - 100) * 4)  # ~4 chars per token estimate
            truncated_text = text[:max_text_chars]
            text_with_title = f"Title: {title}\n{truncated_text}"
            text_tokens = count_tokens(text_with_title)

            # If still too large after truncation, skip this article
            if text_tokens > target_tokens:
                print(f"    Warning: Skipping oversized article '{title}' ({text_tokens} tokens > {target_tokens} target)")
                continue

        # If adding this article would exceed target_tokens, start a new chunk
        if current_tokens + text_tokens > target_tokens and current_chunk:
            chunks_with_titles.append((current_chunk.strip(), current_titles.copy()))
            current_chunk = text_with_title
            current_titles = [title]
            current_tokens = text_tokens

            # Set new target for next chunk if using variable size
            if variable_size:
                target_tokens = random.randint(min_tokens, max_tokens)
            else:
                target_tokens = max_tokens
        else:
            if current_chunk:
                current_chunk += "\n\n" + text_with_title
                current_tokens += text_tokens + count_tokens("\n\n")
            else:
                current_chunk = text_with_title
                current_tokens = text_tokens
            current_titles.append(title)

    # Add the last chunk if it exists
    if current_chunk:
        chunks_with_titles.append((current_chunk.strip(), current_titles.copy()))

    return chunks_with_titles



def save_processed_data(data, filename="processed_data.json"):
    """Save processed data to JSON file"""
    print(f"Saving {len(data)} instances to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Data saved to {filename}")

def convert_json_to_parquet(json_filename, dataset_name):
    """Convert JSON file to parquet format and save in ./data/memalpha/"""

    # Create output directory
    output_dir = "./data/memalpha"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting {json_filename} to parquet format...")

    # Load JSON data
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to DataFrame-friendly format
    rows = []
    for i, instance in enumerate(data):
        # Handle both 'chunks' and 'context_chunks' keys
        chunks_key = 'chunks' if 'chunks' in instance else 'context_chunks'
        # Convert chunks list to string representation
        chunks_str = json.dumps(instance[chunks_key])

        if len(instance['questions_and_answers']) > 100:
            # random sample 100 questions_and_answers
            instance['questions_and_answers'] = random.sample(instance['questions_and_answers'], 100)

        # Convert questions_and_answers to string representation
        qa_str = json.dumps(instance['questions_and_answers'])

        row = {
            'instance_id': i,
            'prompt': instance.get('prompt', ''),  # Use get() with default for lme_train
            'chunks': chunks_str,
            'questions_and_answers': qa_str,
            'num_chunks': len(instance[chunks_key]),
            'num_questions': len(instance['questions_and_answers']),
        }

        if 'sub_source' in instance:
            row['sub_source'] = instance['sub_source']
        elif 'data_source' in instance:
            row['sub_source'] = instance['data_source']
        elif 'metadata' in instance:
            row['sub_source'] = instance['metadata']['source']

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save as parquet
    parquet_filename = os.path.join(output_dir, f"processed_{dataset_name}_data.parquet")
    df.to_parquet(parquet_filename, index=False)

    print(f"Parquet file saved to {parquet_filename}")
    print(f"Parquet file contains {len(df)} instances")

    return parquet_filename



def process_pubmed_rct_dataset():
    """Process pubmed-200k-rct dataset into the desired format"""

    def format_sample_for_chunk(sample: Dict[str, Any]) -> str:
        """Format a single sample for chunk creation"""
        text = sample['text']
        labels = sample['labels']

        # Convert labels list to readable format
        if isinstance(labels, list):
            labels_str = ", ".join(labels)
        else:
            labels_str = str(labels)

        return f"Sample: {text}\nLabels: {labels_str}"

    def format_sample_for_qa(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single sample for QA pair creation"""
        text = sample['text']
        labels = sample['labels']

        # Convert labels list to readable format
        if isinstance(labels, list):
            labels_str = ", ".join(labels)
        else:
            labels_str = str(labels)

        question = f"Sentence: {text}\nWhat are the labels for the above medical abstract sentence? Put your final answer as \\boxed{{label}}." # TODO: Prompt engineering
        answer = labels_str

        return {"question": question, "answer": answer}

    def create_chunk_from_samples(samples: List[Dict[str, Any]], target_tokens: int = 2000) -> str:
        """Create a chunk from given samples"""
        chunk_parts = []
        total_tokens = 0

        for sample in samples:
            sample_text = format_sample_for_chunk(sample)
            sample_tokens = count_tokens(sample_text)

            # Check if adding this sample would exceed the token limit
            if total_tokens + sample_tokens > target_tokens:
                break

            chunk_parts.append(sample_text)
            total_tokens += sample_tokens

        return "\n\n".join(chunk_parts)

    def create_single_chunk(available_samples: List[Dict[str, Any]], max_tokens: int = 2048, min_tokens: int = 100, variable_size: bool = False) -> tuple:
        """Create a single chunk and return the chunk text and used samples

        Args:
            available_samples: List of samples to choose from
            max_tokens: Maximum tokens per chunk
            min_tokens: Minimum tokens per chunk (used when variable_size=True)
            variable_size: If True, randomly vary chunk size between min_tokens and max_tokens
        """
        if not available_samples:
            return "", []

        # Set target tokens for this chunk
        if variable_size:
            target_tokens = random.randint(min_tokens, max_tokens)
        else:
            target_tokens = max_tokens

        # Try different strategies to create a chunk
        random.shuffle(available_samples)

        chunk_samples = []
        total_tokens = 0

        for sample in available_samples:
            sample_text = format_sample_for_chunk(sample)
            sample_tokens = count_tokens(sample_text)

            # If this is the first sample and it's too big, try to truncate or skip
            if len(chunk_samples) == 0 and sample_tokens > target_tokens:
                # Try with truncated text
                truncated_sample = {
                    'text': sample['text'][:1000],  # Truncate to first 1000 chars
                    'labels': sample['labels']
                }
                truncated_text = format_sample_for_chunk(truncated_sample)
                truncated_tokens = count_tokens(truncated_text)

                if truncated_tokens <= target_tokens:
                    chunk_samples.append(truncated_sample)
                    total_tokens = truncated_tokens
                continue

            # Check if adding this sample would exceed the token limit
            if total_tokens + sample_tokens > target_tokens:
                break

            chunk_samples.append(sample)
            total_tokens += sample_tokens

        if chunk_samples:
            chunk_text = create_chunk_from_samples(chunk_samples, target_tokens)
            return chunk_text, chunk_samples
        else:
            return "", []

    def load_pubmed_train_data(dataset_name: str) -> List[Dict[str, Any]]:
        """Load all training data from the pubmed-200k-rct dataset"""
        all_train_data = []

        print(f"Loading {dataset_name} dataset...")

        try:
            # Try loading via Hugging Face datasets
            dataset = load_dataset(dataset_name)
            if 'train' in dataset:
                train_data = list(dataset['train'])
                all_train_data.extend(train_data)
                print(f"Loaded {len(train_data)} samples from train split")
            else:
                print("No 'train' split found in HF dataset")
        except Exception as e:
            print(f"Failed to load via Hugging Face: {e}")
            print("Trying to load from local parquet files...")

            try:
                # Load from local parquet files for pubmed-200k-rct
                parquet_files = {
                    "train": [
                        "train-00000-of-00001-12ed98a863dcf4b2.parquet"
                    ]
                }
                dataset = load_dataset("parquet", data_files=parquet_files)
                if 'train' in dataset:
                    train_data = list(dataset['train'])
                    all_train_data.extend(train_data)
                    print(f"Loaded {len(train_data)} samples from local parquet")
            except Exception as e2:
                print(f"Failed to load local parquet: {e2}")

        print(f"Total loaded samples: {len(all_train_data)}")

        # Print sample data structure for debugging
        if all_train_data:
            print(f"\nSample data structure:")
            sample = all_train_data[0]
            print(f"Keys: {list(sample.keys())}")
            print(f"Sample text (first 100 chars): {str(sample.get('text', 'N/A'))[:100]}")
            print(f"Sample labels: {sample.get('labels', 'N/A')}")

        return all_train_data

    def create_pubmed_dataset_instances(train_data: List[Dict[str, Any]],
                               num_instances: int = 50, chunks_per_instance: int = 10, qa_pairs_per_instance: int = 100) -> List[Dict[str, Any]]:
        """Create the complete dataset with specified number of instances"""
        dataset_instances = []
        used_indices = set()

        print(f"Creating up to {num_instances} dataset instances, each with {chunks_per_instance} chunks and {qa_pairs_per_instance} QA pairs...")

        for i in tqdm(range(num_instances)):
            print(f"\nCreating instance {i+1}/{num_instances}")

            # Check if we have enough unused samples for this instance
            available_indices = [idx for idx in range(len(train_data)) if idx not in used_indices]

            # Estimate needed samples (conservative estimate)
            estimated_samples_needed = chunks_per_instance * 5 + qa_pairs_per_instance  # 5 samples per chunk + QA pairs

            if len(available_indices) < estimated_samples_needed:
                print(f"Not enough unused samples for instance {i+1}. Stopping at {i} instances.")
                break

            # Create chunks for this instance
            instance_chunks = []
            instance_used_indices = set()

            for chunk_idx in range(chunks_per_instance):
                # Get available samples for this chunk
                chunk_available_indices = [idx for idx in available_indices
                                         if idx not in instance_used_indices]

                if not chunk_available_indices:
                    print(f"No more samples available for chunk {chunk_idx+1}")
                    break

                chunk_available_samples = [train_data[idx] for idx in chunk_available_indices]

                # Create the chunk with variable sizes between 100 and 4k tokens
                chunk_text, used_samples = create_single_chunk(chunk_available_samples, max_tokens=2048, min_tokens=1024, variable_size=True)

                if chunk_text and used_samples:
                    # Format chunk with dialogue header using classification-specific templates
                    # Use classification-specific user templates for pubmed-rct
                    user_template = random.choice(CLASSIFICATION_USER_TEMPLATES)
                    assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

                    base_date = datetime(2024, 1, 1)
                    chunk_date = base_date + timedelta(days=chunk_idx)
                    timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

                    # Format the chunk with classification template
                    formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n\n{chunk_text}\n<Assistant>{assistant_template}"

                    instance_chunks.append(formatted_chunk)
                    # Mark these samples as used in this instance
                    for sample in used_samples:
                        for idx in chunk_available_indices:
                            if train_data[idx] == sample:
                                instance_used_indices.add(idx)
                                break

                    print(f"  Chunk {chunk_idx+1}: {count_tokens(chunk_text)} tokens, {len(used_samples)} samples")
                else:
                    print(f"  Failed to create Chunk {chunk_idx+1}")
                    break

            # Only proceed if we have all required chunks
            if len(instance_chunks) != chunks_per_instance:
                print(f"Could not create all {chunks_per_instance} chunks for instance {i+1}. Created {len(instance_chunks)} chunks.")
                print(f"Stopping at {i} complete instances.")
                break

            # Update global used indices
            used_indices.update(instance_used_indices)

            # Create QA pairs for this instance
            qa_available_indices = [idx for idx in range(len(train_data)) if idx not in used_indices]

            if len(qa_available_indices) < qa_pairs_per_instance:
                print(f"Not enough samples for {qa_pairs_per_instance} QA pairs in instance {i+1}")
                print(f"Stopping at {i} complete instances.")
                break

            # Sample indices for QA pairs
            qa_indices = random.sample(qa_available_indices, qa_pairs_per_instance)
            qa_pairs = []

            for idx in qa_indices:
                sample = train_data[idx]
                qa_pair = format_sample_for_qa(sample)
                qa_pairs.append(qa_pair)
                used_indices.add(idx)

            # Create instance
            instance = {
                # "prompt": 'I will provide you with sequential information chunks from medical abstracts. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
                # "prompt": "You are a medical abstract classifier. I will provide you with chunks of text from medical research abstracts. Each chunk belongs to one of the following categories:\n\n0 = Backgrounds\n1 = Conclusions\n2 = Methods\n3 = Objective\n4 = Results\n\nYour task has two steps:\n\n1. **Memorize the label meanings**: As you read each chunk, understand how the content reflects its associated label. Learn to associate common phrases, structure, and intent with each label (e.g., objectives often begin with 'To determine...', methods describe experimental setup, etc.).\n\n2. **Classify**: After seeing several examples, I will give you a new abstract chunk. Based on the label definitions you've learned, classify the new chunk by returning its label number (0–4).\n\nOutput only the number corresponding to the predicted label.\n\nLet's begin with a few chunks to help you understand the label definitions.",
                "prompt": "You are a scientific document classifier. I will provide you with chunks of medical abstracts. Each chunk is labeled with a number. The meaning of each label must be learned from context — you will not be told what each number represents explicitly.\n\nYour task has two steps:\n\n1. **Memorize label mappings**: As you read each labeled chunk, learn to associate the label number with the type of content it represents. Memorize the relationship between the chunk content and its numeric label. **Classify**: After seeing enough labeled examples, I will give you a new chunk. Based on your memory of label meanings, classify the new chunk and output only the number (0–4) that best represents it.\n\nOutput format: A single number from 0 to 4.\n\nLet's begin with the labeled examples.",
                "chunks": instance_chunks,
                "questions_and_answers": qa_pairs
            }

            dataset_instances.append(instance)

            print(f"Instance {i+1} created successfully with {len(instance_chunks)} chunks and {len(qa_pairs)} QA pairs")

        return dataset_instances

    # Set random seed for reproducibility
    random.seed(42)

    # Load all training data from the pubmed-200k-rct dataset
    train_data = load_pubmed_train_data("pietrolesci/pubmed-200k-rct")

    if not train_data:
        print("Failed to load training data. Exiting.")
        return []

    print(f"\nLoaded {len(train_data)} total training samples")

    # Create dataset instances
    dataset_instances = create_pubmed_dataset_instances(
        train_data,
        num_instances=100,
        chunks_per_instance=10,
        qa_pairs_per_instance=100
    )

    print(f"\nSuccessfully created {len(dataset_instances)} complete dataset instances")
    return dataset_instances

def process_booksum_dataset():
    """Process BOOKSUM dataset into the desired format"""

    if not os.path.exists("./data/booksum/raw_instances.json"):

        # Load the datasets
        print("Loading BOOKSUM chapter dataset...")
        chapter_data = load_dataset("ubaada/booksum-complete-cleaned", "chapters")

        print(f"Chapters dataset loaded: {len(chapter_data['train'])} train, {len(chapter_data['test'])} test")

        processed_data = []

        # Process each row individually to maintain 1:1 correspondence
        print("Processing chapters data...")
        all_chapters = []
        all_chapters.extend(chapter_data['train'])
        all_chapters.extend(chapter_data['test'])

        processed_count = 0
        skipped_count = 0

        for idx, chapter in enumerate(all_chapters):

            num_tokens = count_tokens(chapter['text'])

            print(f"Processing chapter {idx+1}/{len(all_chapters)}: Book {chapter['book_title']}, Chapter {chapter['chapter_id']}, Length {num_tokens}")

            # Create chunks for this chapter
            chapter_chunks_raw = []
            if num_tokens > 2048:
                chunks = create_chunks_use_sent_tokenizer(chapter['text'], max_tokens=2048)
                print("Created chunks with lengths: ", [count_tokens(chunk) for chunk in chunks])
                chapter_chunks_raw = chunks
            else:
                chapter_chunks_raw = [chapter['text']]

            # Filter out chapters with fewer than 5 chunks
            if len(chapter_chunks_raw) < 5:
                print(f"Skipping chapter {chapter['chapter_id']} from {chapter['book_title']} - only {len(chapter_chunks_raw)} chunks (minimum 5 required)")
                skipped_count += 1
                continue

            # Format chunks with dialogue headers
            base_date = datetime(2024, 1, 1)

            chapter_chunks = []
            chapter_dates = []  # Store dates for question generation
            current_date = base_date

            for chunk_idx, chunk_content in enumerate(chapter_chunks_raw):
                # Create progressive dates (incrementing by random 1-3 days)
                if chunk_idx > 0:  # Don't add days for the first chunk
                    days_to_add = random.randint(1, 3)
                    current_date = current_date + timedelta(days=days_to_add)
                date_str = current_date.strftime("%Y-%m-%d")
                chapter_dates.append(date_str)

                # Format with the specific BookSum template
                formatted_chunk = f"[Event happened on {date_str} The user is reading a book]\n<User> {chunk_content}\n\n<System> Please remember what the user reads on {date_str}, save the details within the book, and retain a summary of the book the user has read so far."
                chapter_chunks.append(formatted_chunk)

            # Create a data instance for this chapter with date-based question
            first_date = chapter_dates[0] if chapter_dates else "2024-01-01"
            last_date = chapter_dates[-1] if chapter_dates else "2024-01-01"

            data_instance = {
                'prompt': 'I will provide you with sequential chunks from a book chapter. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
                'chunks': chapter_chunks,
                'questions_and_answers': [{'question': f"Summarize the content of the book I read from {first_date} to {last_date}", 'answer': chapter['summary'][0]['text']}],
                'book_title': chapter['book_title'],
                'chapter_id': chapter['chapter_id'],
                'reading_dates': {'start': first_date, 'end': last_date}
            }

            processed_data.append(data_instance)
            processed_count += 1

        print(f"Processed {processed_count} chapters, skipped {skipped_count} chapters (too short)")

        os.makedirs("./data/booksum", exist_ok=True)
        with open("./data/booksum/raw_instances.json", "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    else:
        with open("./data/booksum/raw_instances.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)


    # Now start filtering the data

    # 1. Filter out instances with more than 50 chunks
    print("Original number of chapter instances: ", len(processed_data))
    processed_data = [x for x in processed_data if len(x['chunks']) <= 50]
    print("Number of instances after filtering out long chapters: ", len(processed_data))

    # Filter out instances with too many chunks (over 20)
    max_chunks_allowed = 20
    filtered_instances = []
    instances_with_too_many_chunks = 0

    for instance in processed_data:
        num_chunks = len(instance['chunks'])
        if num_chunks <= max_chunks_allowed:
            filtered_instances.append(instance)
        else:
            instances_with_too_many_chunks += 1
            print(f"Removing BOOKSUM instance with {num_chunks} chunks (> {max_chunks_allowed})")

    print(f"\n📊 BOOKSUM Chunk Filtering Summary:")
    print(f"  • Removed {instances_with_too_many_chunks} instances with > {max_chunks_allowed} chunks")
    if processed_data:
        print(f"  • Instances: {len(processed_data):,} → {len(filtered_instances):,} (kept {len(filtered_instances)/len(processed_data)*100:.1f}%)")
    else:
        print(f"  • Instances: 0 → 0")

    print(f"Final processed dataset: {len(filtered_instances)} chapters")
    return filtered_instances

def print_statistics(processed_data, dataset_name):
    """Print dataset statistics"""
    print(f"\n=== {dataset_name.upper()} Dataset Statistics ===")
    print(f"Total instances created: {len(processed_data)}")

    if processed_data:
        # Handle both 'chunks' and 'context_chunks' keys
        chunks_key = 'chunks' if 'chunks' in processed_data[0] else 'context_chunks'
        print(f"Average chunks per instance: {sum(len(inst[chunks_key]) for inst in processed_data) / len(processed_data):.2f}")
        print(f"Average Q&As per instance: {sum(len(inst['questions_and_answers']) for inst in processed_data) / len(processed_data):.2f}")

        # Show example of first instance
        print(f"\n=== Example {dataset_name.upper()} Instance ===")
        example = processed_data[0]
        print(f"Number of chunks: {len(example[chunks_key])}")
        print(f"Number of Q&As: {len(example['questions_and_answers'])}")
        print(f"First chunk preview: {example[chunks_key][0][:200]}...")
        print(f"First Q&A: {example['questions_and_answers'][0]}")

def analyze_dataset_statistics():
    """Analyze and print detailed statistics for each individual dataset"""

    # Define input parquet files
    dataset_files = {
        'squad': "./data/memalpha/processed_squad_data.parquet",
        'hotpotqa': "./data/memalpha/processed_hotpotqa_data.parquet",
        'booksum': "./data/memalpha/processed_booksum_data.parquet",
        'wos46985': "./data/memalpha/processed_wos46985_data.parquet",
        'pubmed-rct': "./data/memalpha/processed_pubmed-rct_data.parquet",
        'arxiv-classification': "./data/memalpha/processed_arxiv-classification_data.parquet",
        'eurlex': "./data/memalpha/processed_eurlex_data.parquet"
    }

    print("\n" + "="*80)
    print("DETAILED DATASET ANALYSIS")
    print("="*80)

    for dataset_name, file_path in dataset_files.items():
        if not os.path.exists(file_path):
            print(f"\nWarning: File not found: {file_path}")
            continue

        print(f"\n{'='*20} {dataset_name.upper()} DATASET {'='*20}")

        # Load the dataset
        df = pd.read_parquet(file_path)

        # Standardize instances to get proper format
        standardized_instances = []
        for idx, row in df.iterrows():
            try:
                # Add source_dataset field for standardization
                row_dict = row.to_dict()
                row_dict['source_dataset'] = dataset_name
                standardized_instance = standardize_instance_format(pd.Series(row_dict))
                standardized_instances.append(standardized_instance)
            except Exception as e:
                print(f"Error processing instance {idx}: {e}")
                continue

        if not standardized_instances:
            print(f"No valid instances found in {dataset_name}")
            continue

        # Calculate statistics
        total_instances = len(standardized_instances)
        total_chunks = sum(len(inst['chunks']) for inst in standardized_instances)
        total_questions = sum(len(inst['questions_and_answers']) for inst in standardized_instances)

        # Calculate chunk counts per instance for min/max statistics
        chunks_per_instance = [len(inst['chunks']) for inst in standardized_instances]
        min_chunks_per_instance = min(chunks_per_instance) if chunks_per_instance else 0
        max_chunks_per_instance = max(chunks_per_instance) if chunks_per_instance else 0

        # Calculate chunk lengths (in tokens)
        all_chunk_lengths = []
        for instance in standardized_instances:
            for chunk in instance['chunks']:
                chunk_length = count_tokens(chunk)
                all_chunk_lengths.append(chunk_length)

        # Calculate averages
        avg_chunks_per_instance = total_chunks / total_instances
        avg_questions_per_instance = total_questions / total_instances
        avg_questions_per_chunk = total_questions / total_chunks if total_chunks > 0 else 0
        avg_chunk_length = sum(all_chunk_lengths) / len(all_chunk_lengths) if all_chunk_lengths else 0

        # Calculate length thresholds for warnings
        max_length = max(all_chunk_lengths) if all_chunk_lengths else 0
        chunks_over_4096 = sum(1 for length in all_chunk_lengths if length > 4096)
        chunks_over_8000 = sum(1 for length in all_chunk_lengths if length > 8000)

        # Print statistics
        print(f"1. Number of data instances: {total_instances:,}")
        print(f"2. Average chunks per instance: {avg_chunks_per_instance:.2f}")
        print(f"3. Average questions per chunk: {avg_questions_per_chunk:.2f}")
        print(f"4. Average length per chunk: {avg_chunk_length:.1f} tokens")

        # Prominent max length display with warnings
        print(f"\n{'='*50}")
        print(f"⚠️  MAXIMUM CHUNK LENGTH: {max_length:.0f} tokens")
        if max_length > 4096:
            print(f"⚠️  WARNING: Maximum chunk exceeds 4096 token target!")
        if max_length > 8000:
            print(f"🚨 CRITICAL: Maximum chunk exceeds 8000 tokens!")
        print(f"{'='*50}")

        # Additional detailed statistics
        print(f"\nDetailed Statistics:")
        print(f"  - Total chunks: {total_chunks:,}")
        print(f"  - Total questions: {total_questions:,}")
        print(f"  - Average questions per instance: {avg_questions_per_instance:.2f}")
        print(f"  - Chunks per instance distribution:")
        print(f"    * Min: {min_chunks_per_instance} chunks")
        print(f"    * Max: {max_chunks_per_instance} chunks")
        print(f"    * Average: {avg_chunks_per_instance:.2f} chunks")
        print(f"  - Chunk length distribution:")
        print(f"    * Min: {min(all_chunk_lengths):.0f} tokens")
        print(f"    * Max: {max_length:.0f} tokens")
        print(f"    * Median: {sorted(all_chunk_lengths)[len(all_chunk_lengths)//2]:.0f} tokens")
        print(f"    * Chunks > 4096 tokens: {chunks_over_4096:,} ({chunks_over_4096/total_chunks*100:.1f}%)")
        if chunks_over_8000 > 0:
            print(f"    * Chunks > 8000 tokens: {chunks_over_8000:,} ({chunks_over_8000/total_chunks*100:.1f}%)")

        # Show sample from dataset
        if standardized_instances:
            sample = standardized_instances[0]
            print(f"\nSample Instance:")
            print(f"  - Prompt: {sample['prompt'][:100]}...")
            print(f"  - Number of chunks: {len(sample['chunks'])}")
            print(f"  - Number of questions: {len(sample['questions_and_answers'])}")
            print(f"  - First chunk preview: {sample['chunks'][0][:150]}...")
            if sample['questions_and_answers']:
                print(f"  - Sample Q&A: {sample['questions_and_answers'][0]}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)

def standardize_instance_format(instance_row):
    """Standardize a single instance to the required format"""
    # Parse JSON strings back to Python objects
    chunks = json.loads(instance_row['chunks'])
    questions_and_answers = json.loads(instance_row['questions_and_answers'])

    # Standardize questions_and_answers format
    standardized_qas = []
    for qa in questions_and_answers:
        standardized_qa = {
            'question': qa['question']
        }

        # Handle different answer formats
        if 'answer' in qa:
            # Already in singular format
            standardized_qa['answer'] = qa['answer']
        elif 'answers' in qa:
            # Convert from plural to singular - take the first answer
            answers = qa['answers']
            if isinstance(answers, list) and len(answers) > 0:
                standardized_qa['answer'] = answers[0]
            else:
                standardized_qa['answer'] = str(answers) if answers else ""
        else:
            # Fallback case
            standardized_qa['answer'] = ""

        standardized_qas.append(standardized_qa)

    # Collect metadata from additional fields (beyond core fields)
    core_fields = {'instance_id', 'prompt', 'chunks', 'questions_and_answers', 'num_chunks', 'num_questions', 'source_dataset'}
    metadata = {}
    for key, value in instance_row.items():
        if key not in core_fields:
            metadata[key] = value

    # Return standardized instance
    # For ttl_train, source_dataset already contains the sub_source (TREC-C or NLU)
    return {
        'prompt': instance_row['prompt'],
        'chunks': chunks,
        'questions_and_answers': standardized_qas,
        'data_source': instance_row['source_dataset'],
        'metadata': metadata
    }




def split_dataset(dataset_name, train_ratio=None, max_train_chunk_num=None, random_seed=42):
    """Split a single dataset into train/test sets with standardized format

    Args:
        dataset_name: Name of the dataset to split (e.g., 'squad', 'hotpotqa', etc.)
        train_ratio: Ratio of data to use for training (default: 0.9)
        max_train_chunk_num: If provided (when train_ratio is None), instances with more chunks go to test
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_path, test_path) or (None, None) if failed
    """

    # If train_ratio is None, we need max_train_chunk_num for chunk-based splitting
    if train_ratio is None:
        assert max_train_chunk_num is not None, "max_train_chunk_num must be provided when train_ratio is None"

    # Define input parquet file
    if dataset_name == 'booksum':
        parquet_file = f"./data/memalpha/processed_{dataset_name}_data_with_keywords.parquet"
    else:
        parquet_file = f"./data/memalpha/processed_{dataset_name}_data.parquet"

    # Check if file exists
    if not os.path.exists(parquet_file):
        print(f"Error: File not found: {parquet_file}")
        return None, None

    print(f"Loading {dataset_name} dataset from {parquet_file}...")

    # Load the dataset
    df = pd.read_parquet(parquet_file)
    # Add source dataset information
    # For ttl_train, use sub_source as the actual source_dataset
    if dataset_name == 'ttl_train' and 'sub_source' in df.columns:
        # Use sub_source as the actual source for ttl_train
        df['source_dataset'] = df['sub_source']
        print(f"Using sub_source field for ttl_train dataset sources")
    else:
        df['source_dataset'] = dataset_name
    print(f"Loaded {parquet_file}: {len(df)} instances")

    # Add dataset statistics
    print(f"\nDataset composition:")
    if dataset_name == 'ttl_train' and 'sub_source' in df.columns:
        # Show breakdown by sub_source
        for source in df['source_dataset'].unique():
            source_df = df[df['source_dataset'] == source]
            print(f"  - {source}: {len(source_df)} instances")
    else:
        print(f"  - {dataset_name}: {len(df)} instances")
    print(f"  - Total questions: {df['num_questions'].sum()}")
    print(f"  - Average questions per instance: {df['num_questions'].mean():.1f}")

    # Standardize format and convert to list of dictionaries
    print("\nStandardizing data format...")
    standardized_instances = []
    for idx, row in df.iterrows():
        try:
            standardized_instance = standardize_instance_format(row)
            standardized_instances.append(standardized_instance)
        except Exception as e:
            print(f"Error standardizing instance {idx}: {e}")
            continue

    print(f"Successfully standardized {len(standardized_instances)} instances")

    if not standardized_instances:
        print("Error: No valid instances remaining after chunk filtering!")
        return None, None

    # Shuffle the data
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)
    random.shuffle(standardized_instances)

    # Split into train/test
    n_total = len(standardized_instances)

    if train_ratio is None:
        # Chunk-based splitting: instances with more than max_train_chunk_num chunks go to test
        train_instances = []
        test_instances = []

        for instance in standardized_instances:
            num_chunks = len(instance['chunks'])
            if num_chunks <= max_train_chunk_num:
                train_instances.append(instance)
            else:
                test_instances.append(instance)

        print(f"\nChunk-based split results (threshold: {max_train_chunk_num} chunks):")
        print(f"Training set: {len(train_instances)} instances ({len(train_instances)/n_total*100:.1f}%) - instances with ≤{max_train_chunk_num} chunks")
        print(f"Test set: {len(test_instances)} instances ({len(test_instances)/n_total*100:.1f}%) - instances with >{max_train_chunk_num} chunks")
    else:
        # Ratio-based splitting
        n_train = int(n_total * train_ratio)

        train_instances = standardized_instances[:n_train]
        test_instances = standardized_instances[n_train:]

        print(f"\nRatio-based split results:")
        print(f"Training set: {len(train_instances)} instances ({len(train_instances)/n_total*100:.1f}%)")
        print(f"Test set: {len(test_instances)} instances ({len(test_instances)/n_total*100:.1f}%)")

    # Create output directory
    output_dir = f"./data/memalpha/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Convert standardized instances back to DataFrame format for parquet saving
    def instances_to_dataframe(instances):
        """Convert standardized instances to DataFrame format suitable for parquet"""
        rows = []
        for i, instance in enumerate(instances):
            # Convert chunks and questions_and_answers back to JSON strings for parquet storage
            chunks_str = json.dumps(instance['chunks'])
            qa_str = json.dumps(instance['questions_and_answers'])
            metadata_str = json.dumps(instance['metadata'])

            for qa_pair in instance['questions_and_answers']:
                if not isinstance(qa_pair['answer'], list):
                    # assert isinstance(qa_pair['answer'], str)
                    if not isinstance(qa_pair['answer'], str):
                        import ipdb; ipdb.set_trace()

            assert len(instance['questions_and_answers']) <= 100, f"Questions and answers length is {len(instance['questions_and_answers'])}"

            row = {
                'instance_id': i,
                'prompt': instance['prompt'],
                'chunks': chunks_str,
                'questions_and_answers': qa_str,
                'data_source': instance['data_source'],
                'metadata': metadata_str,
                'num_chunks': len(instance['chunks']),
                'num_questions': len(instance['questions_and_answers'])
            }
            rows.append(row)
        return pd.DataFrame(rows)

    # Save train and test sets as parquet files with standardized format
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df = instances_to_dataframe(train_instances)
    test_df = instances_to_dataframe(test_instances)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved files:")
    print(f"Training set: {train_path}")
    print(f"Test set: {test_path}")

    # Calculate statistics for each split
    def calculate_split_stats(instances, split_name):
        if not instances:
            return

        source_counts = {}
        total_questions = 0
        total_chunks = 0

        for instance in instances:
            source = instance['data_source']
            source_counts[source] = source_counts.get(source, 0) + 1
            total_questions += len(instance['questions_and_answers'])
            total_chunks += len(instance['chunks'])

        avg_questions = total_questions / len(instances)
        avg_chunks = total_chunks / len(instances)

        print(f"\n{split_name} statistics:")
        print(f"  Source distribution: {source_counts}")
        print(f"  Total questions: {total_questions}")
        print(f"  Average questions per instance: {avg_questions:.1f}")
        print(f"  Average chunks per instance: {avg_chunks:.1f}")

    calculate_split_stats(train_instances, "Training set")
    calculate_split_stats(test_instances, "Test set")

    # Validate format of a few examples
    print(f"\nValidating standardized format...")
    for i, instance in enumerate(train_instances[:3]):
        print(f"Example {i+1}:")
        print(f"  Prompt: {instance['prompt'][:100]}...")
        print(f"  Chunks: {len(instance['chunks'])} chunks")
        print(f"  Q&As: {len(instance['questions_and_answers'])} questions")
        if instance['questions_and_answers']:
            first_qa = instance['questions_and_answers'][0]
            print(f"  Sample Q&A keys: {list(first_qa.keys())}")
            assert 'question' in first_qa and 'answer' in first_qa, f"Invalid Q&A format in instance {i}"

    print("Format validation passed!")

    return train_path, test_path


def combine_and_split_datasets(train_ratio=0.9, random_seed=42):
    """Combine all parquet files, shuffle, and split into train/test sets with standardized format"""

    # Define input parquet files
    parquet_files = [
        "./data/memalpha/processed_squad_data.parquet",
        "./data/memalpha/processed_hotpotqa_data.parquet",
        # "./data/memalpha/processed_booksum_data.parquet",
        # "./data/memalpha/processed_wos46985_data.parquet",
        # "./data/memalpha/processed_pubmed-rct_data.parquet",
        # "./data/memalpha/processed_arxiv-classification_data.parquet",
        # "./data/memalpha/processed_eurlex_data.parquet"
    ]
    source_names = [
        "squad",
        "hotpotqa",
        # "booksum",
        # "wos46985",
        # "pubmed-rct",
        # "arxiv-classification",
        # "eurlex"
    ]

    # Check if files exist
    existing_files = []
    for file_path in parquet_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")

    if not existing_files:
        print("Error: No parquet files found to combine!")
        return None, None

    print(f"Combining {len(existing_files)} parquet files...")

    # Load and combine all datasets
    dataframes = []
    for idx, file_path in enumerate(existing_files):
        df = pd.read_parquet(file_path)
        # Add source dataset information
        dataset_name = os.path.basename(file_path).replace("processed_", "").replace("_data.parquet", "")
        df['source_dataset'] = source_names[idx]
        dataframes.append(df)
        print(f"Loaded {file_path}: {len(df)} instances")

    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataset: {len(combined_df)} instances")

    # Add dataset statistics
    print("\nDataset composition:")
    print(combined_df['source_dataset'].value_counts())
    print(f"Total questions: {combined_df['num_questions'].sum()}")
    print(f"Average questions per instance: {combined_df['num_questions'].mean():.1f}")

    # Standardize format and convert to list of dictionaries
    print("\nStandardizing data format...")
    standardized_instances = []
    for idx, row in combined_df.iterrows():
        try:
            standardized_instance = standardize_instance_format(row)
            standardized_instances.append(standardized_instance)
        except Exception as e:
            print(f"Error standardizing instance {idx}: {e}")
            continue

    print(f"Successfully standardized {len(standardized_instances)} instances")

    # Filter out instances with too many chunks (over 20)
    print("\nFiltering instances with too many chunks...")
    max_chunks_allowed = 20
    filtered_by_chunks = []
    instances_with_too_many_chunks = 0

    for instance in standardized_instances:
        num_chunks = len(instance['chunks'])
        if num_chunks <= max_chunks_allowed:
            filtered_by_chunks.append(instance)
        else:
            instances_with_too_many_chunks += 1
            print(f"    Removing instance from {instance['data_source']} with {num_chunks} chunks (> {max_chunks_allowed})")

    print(f"\n📊 Chunk Filtering Summary:")
    print(f"  • Removed {instances_with_too_many_chunks} instances with > {max_chunks_allowed} chunks")
    print(f"  • Instances: {len(standardized_instances):,} → {len(filtered_by_chunks):,} (kept {len(filtered_by_chunks)/len(standardized_instances)*100:.1f}%)")

    standardized_instances = filtered_by_chunks

    if not standardized_instances:
        print("Error: No valid instances remaining after chunk filtering!")
        return None, None

    print(min([len(instance['questions_and_answers']) for instance in standardized_instances]))

    # Shuffle the data
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)
    random.shuffle(standardized_instances)

    # Split into train/test
    n_total = len(standardized_instances)
    n_train = int(n_total * train_ratio)

    train_instances = standardized_instances[:n_train]
    test_instances = standardized_instances[n_train:]

    print(f"\nSplit results:")
    print(f"Training set: {len(train_instances)} instances ({len(train_instances)/n_total*100:.1f}%)")
    print(f"Test set: {len(test_instances)} instances ({len(test_instances)/n_total*100:.1f}%)")

    # Create output directory
    output_dir = "./data/memalpha"
    os.makedirs(output_dir, exist_ok=True)

    # Convert standardized instances back to DataFrame format for parquet saving
    def instances_to_dataframe(instances):
        """Convert standardized instances to DataFrame format suitable for parquet"""
        rows = []
        for i, instance in enumerate(instances):
            # Convert chunks and questions_and_answers back to JSON strings for parquet storage
            chunks_str = json.dumps(instance['chunks'])
            qa_str = json.dumps(instance['questions_and_answers'])
            metadata_str = json.dumps(instance['metadata'])

            for qa_pair in instance['questions_and_answers']:
                if not isinstance(qa_pair['answer'], list):
                    assert isinstance(qa_pair['answer'], str)

            assert len(instance['questions_and_answers']) <= 100, f"Questions and answers length is {len(instance['questions_and_answers'])}"

            row = {
                'instance_id': i,
                'prompt': instance['prompt'],
                'chunks': chunks_str,
                'questions_and_answers': qa_str,
                'data_source': instance['data_source'],
                'metadata': metadata_str,
                'num_chunks': len(instance['chunks']),
                'num_questions': len(instance['questions_and_answers'])
            }
            rows.append(row)
        return pd.DataFrame(rows)

    # Save train and test sets as parquet files with standardized format
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df = instances_to_dataframe(train_instances)
    test_df = instances_to_dataframe(test_instances)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved files:")
    print(f"Training set: {train_path}")
    print(f"Test set: {test_path}")

    # Calculate statistics for each split
    def calculate_split_stats(instances, split_name):
        if not instances:
            return

        source_counts = {}
        total_questions = 0
        total_chunks = 0

        for instance in instances:
            source = instance['data_source']
            source_counts[source] = source_counts.get(source, 0) + 1
            total_questions += len(instance['questions_and_answers'])
            total_chunks += len(instance['chunks'])

        avg_questions = total_questions / len(instances)
        avg_chunks = total_chunks / len(instances)

        print(f"\n{split_name} statistics:")
        print(f"  Source distribution: {source_counts}")
        print(f"  Total questions: {total_questions}")
        print(f"  Average questions per instance: {avg_questions:.1f}")
        print(f"  Average chunks per instance: {avg_chunks:.1f}")

    calculate_split_stats(train_instances, "Training set")
    calculate_split_stats(test_instances, "Test set")

    # Validate format of a few examples
    print(f"\nValidating standardized format...")
    for i, instance in enumerate(train_instances[:3]):
        print(f"Example {i+1}:")
        print(f"  Prompt: {instance['prompt'][:100]}...")
        print(f"  Chunks: {len(instance['chunks'])} chunks")
        print(f"  Q&As: {len(instance['questions_and_answers'])} questions")
        if instance['questions_and_answers']:
            first_qa = instance['questions_and_answers'][0]
            print(f"  Sample Q&A keys: {list(first_qa.keys())}")
            assert 'question' in first_qa and 'answer' in first_qa, f"Invalid Q&A format in instance {i}"

    print("Format validation passed!")

    return train_path, test_path

def process_instance_batch(batch_data):
    """Worker function to process a batch of instances in parallel"""
    instances_batch, api_key, batch_id, max_questions_to_test = batch_data

    # Create OpenAI client for this worker
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-01-01-preview",
        azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
    )

    results = []
    batch_stats = {
        'total_instances': len(instances_batch),
        'total_questions_processed': 0,
        'total_questions_removed': 0,
        'total_api_calls': 0,
        'instances_processed': 0
    }

    print(f"🚀 Batch {batch_id}: Starting processing of {len(instances_batch)} instances at {time.strftime('%H:%M:%S')}")
    import sys
    sys.stdout.flush()  # Force immediate output

    for instance_idx, instance_data in enumerate(instances_batch):
        instance_id = instance_data['instance_id']
        questions_and_answers = instance_data['questions_and_answers']

        questions_to_keep = 0
        questions_processed = 0
        questions_removed_this_instance = 0

        # Determine how many questions to test
        questions_to_test = questions_and_answers[:max_questions_to_test] if max_questions_to_test else questions_and_answers

        print(f"📝 Batch {batch_id}: Processing instance {instance_id} ({instance_idx+1}/{len(instances_batch)}) - {len(questions_to_test)} questions available")
        print(f"   🎯 Goal: Find 100 questions to keep (stop when reached)")
        sys.stdout.flush()

        for qa_idx, qa in enumerate(questions_to_test):
            # Stop if we already have 100 questions to keep
            if questions_to_keep >= 100:
                print(f"  ✅ Instance {instance_id}: Reached target of 100 questions to keep after processing {questions_processed} questions")
                sys.stdout.flush()
                break

            question = qa['question']
            expected_answer = qa.get('answer', qa.get('answers', [''])[0] if qa.get('answers') else '')

            # Convert expected_answer to string if it's a list
            if isinstance(expected_answer, list):
                expected_answer = expected_answer[0] if expected_answer else ''
            expected_answer = str(expected_answer).strip()

            # Show progress for every question
            print(f"  ❓ Instance {instance_id}: Question {questions_processed + 1}/{len(questions_to_test)} - Currently keeping {questions_to_keep}/100")
            sys.stdout.flush()

            try:
                # Ask GPT-4o-mini the question without context
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer the question as accurately as possible based on your training data."},
                    {"role": "user", "content": question}
                ]

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.0  # Use low temperature for consistent answers
                )

                model_answer = response.choices[0].message.content.strip()
                is_correct = is_answer_correct(model_answer, expected_answer)
                keep_question = not is_correct

                # Store question and prediction for analysis
                result = {
                    'instance_id': instance_id,
                    'question_id': qa_idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'prediction_without_chunk': model_answer,
                    'is_correct': is_correct,
                    'keep_question': keep_question,
                    'batch_id': batch_id
                }
                results.append(result)

                if keep_question:
                    questions_to_keep += 1
                    print(f"    ✅ KEEP: Model couldn't answer correctly (Expected: {expected_answer[:50]}..., Got: {model_answer[:50]}...)")
                else:
                    questions_removed_this_instance += 1
                    print(f"    ❌ REMOVE: Model answered correctly (Answer: {expected_answer[:50]}...)")

                questions_processed += 1
                batch_stats['total_api_calls'] += 1

                # Detailed progress update every 5 questions
                if questions_processed % 5 == 0:
                    progress_pct = (questions_processed / len(questions_to_test)) * 100
                    print(f"  📊 Instance {instance_id}: Progress {questions_processed}/{len(questions_to_test)} ({progress_pct:.1f}%) - Keeping {questions_to_keep}/100, Removed: {questions_removed_this_instance}")
                    sys.stdout.flush()

                # Add small delay to avoid rate limiting
                time.sleep(0.05)  # Reduced delay since we have multiple processes

            except Exception as e:
                print(f"  ⚠️ Instance {instance_id}: ERROR at Q{qa_idx+1}: {e}")
                sys.stdout.flush()

                # Store question with error info for analysis
                result = {
                    'instance_id': instance_id,
                    'question_id': qa_idx,
                    'question': question,
                    'expected_answer': expected_answer,
                    'prediction_without_chunk': None,
                    'is_correct': False,
                    'keep_question': True,  # Keep if there's an error
                    'error': str(e),
                    'batch_id': batch_id
                }
                results.append(result)

                questions_to_keep += 1  # Count error questions as kept
                questions_processed += 1

                time.sleep(0.5)  # Longer delay after error

        # Update batch statistics
        batch_stats['total_questions_processed'] += questions_processed
        batch_stats['total_questions_removed'] += questions_removed_this_instance
        batch_stats['instances_processed'] += 1

        # Instance summary
        if questions_to_keep >= 100:
            print(f"  🎉 Instance {instance_id} COMPLETE: ✅ Target reached - keeping {questions_to_keep} questions after testing {questions_processed}/{len(questions_to_test)}")
        else:
            print(f"  🎉 Instance {instance_id} COMPLETE: ⚠️  Only found {questions_to_keep} questions to keep after testing all {questions_processed} questions")
        print(f"    📈 Stats: Tested={questions_processed}, Keeping={questions_to_keep}, Removed={questions_removed_this_instance}, Keep Rate={questions_to_keep/questions_processed*100:.1f}%")
        sys.stdout.flush()

    # Batch summary
    questions_kept_in_batch = batch_stats['total_questions_processed'] - batch_stats['total_questions_removed']
    print(f"🏁 Batch {batch_id} COMPLETE at {time.strftime('%H:%M:%S')}: {batch_stats['instances_processed']}/{batch_stats['total_instances']} instances")
    print(f"  📊 Questions Tested: {batch_stats['total_questions_processed']}")
    print(f"  ✅ Questions Found to Keep: {questions_kept_in_batch}")
    print(f"  ❌ Questions Removed (model answered correctly): {batch_stats['total_questions_removed']}")
    print(f"  🔗 API Calls Made: {batch_stats['total_api_calls']}")
    print(f"  🧠 Model Accuracy Rate: {batch_stats['total_questions_removed']/batch_stats['total_questions_processed']*100 if batch_stats['total_questions_processed'] > 0 else 0:.1f}% (higher = more questions removed)")
    sys.stdout.flush()

    return results

def filter_dataset(dataset_name, max_questions_to_test=None, num_processes=16):
    """Filter out questions that GPT-4o-mini can answer correctly without context chunks

    For each instance, continues processing questions until it finds 100 questions to KEEP
    (i.e., questions the model cannot answer correctly). Stops early when 100 questions
    are found that should be kept in the dataset.

    Args:
        dataset_name: Either 'squad' or 'hotpotqa'
        max_questions_to_test: Maximum number of questions to test (for debugging/testing)
        num_processes: Number of parallel processes to use (default: 16)
    """
    if dataset_name not in ['squad', 'hotpotqa']:
        print(f"Error: Filtering only supported for 'squad' and 'hotpotqa', got '{dataset_name}'")
        return

    # Limit number of processes to avoid overwhelming the API
    max_processes = min(num_processes, mp.cpu_count(), 16)  # Cap at 16 processes
    if max_processes != num_processes:
        print(f"Limiting processes from {num_processes} to {max_processes} (CPU count: {mp.cpu_count()})")
        num_processes = max_processes

    # Load the parquet file
    parquet_file = f"./data/memalpha/processed_{dataset_name}_data.parquet"
    if not os.path.exists(parquet_file):
        print(f"Error: Parquet file not found: {parquet_file}")
        return

    print(f"Loading {dataset_name} dataset from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df)} instances")

    # Setup API key for workers
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Error: AZURE_OPENAI_API_KEY environment variable not set")
        return

    # Collect all instances to process
    all_instances = []

    print("Collecting instances for processing...")
    total_questions_before = 0

    for idx, row in df.iterrows():
        chunks = json.loads(row['chunks'])
        questions_and_answers = json.loads(row['questions_and_answers'])

        total_questions_before += len(questions_and_answers)

        instance_data = {
            'instance_id': idx,
            'questions_and_answers': questions_and_answers,
            'chunks': chunks
        }
        all_instances.append(instance_data)

    print(f"Collected {len(all_instances)} instances with {total_questions_before} total questions to process with {num_processes} processes")

    # Split instances into batches for multiprocessing
    batch_size = max(1, len(all_instances) // num_processes)
    instance_batches = []

    for i in range(0, len(all_instances), batch_size):
        batch = all_instances[i:i + batch_size]
        batch_id = len(instance_batches)
        instance_batches.append((batch, api_key, batch_id, max_questions_to_test))

    print(f"Split into {len(instance_batches)} batches of ~{batch_size} instances each")

    # Estimate processing time - we'll process questions until we find 100 to keep per instance
    avg_questions_per_instance = total_questions_before / len(all_instances)
    # Assume roughly 70% of questions will be kept (model can't answer), so we need to process ~143 questions to find 100 to keep
    estimated_questions_per_instance = min(avg_questions_per_instance, 143)  # Cap at 143 since we stop at 100 kept
    estimated_api_calls = estimated_questions_per_instance * len(all_instances)
    estimated_minutes = (estimated_api_calls * 0.1) / 60 / num_processes  # ~0.1 seconds per API call, divided by processes

    print(f"🎯 Goal: Find 100 questions to keep per instance (stop when target reached)")
    print(f"⏱️  Estimated processing time: ~{estimated_minutes:.1f} minutes")
    print(f"📊 Expected API calls: ~{estimated_api_calls:,.0f} (early stopping when 100 questions found per instance)")

    # Process batches in parallel
    print("Processing instances in parallel...")
    print(f"Total instances to process: {len(all_instances)}")
    print(f"Estimated questions to process: {total_questions_before} (may be less due to early stopping)")
    print(f"Using {num_processes} parallel processes")
    print("-" * 60)

    with mp.Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_instance_batch, instance_batches),
            total=len(instance_batches),
            desc=f"🔄 Processing {len(instance_batches)} batches",
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} batches [{elapsed}<{remaining}, {rate_fmt}]"
        ))

    # Combine results from all batches
    all_results = []
    for batch_result in batch_results:
        all_results.extend(batch_result)

    questions_and_predictions = all_results
    api_calls_made = len([r for r in all_results if r.get('prediction_without_chunk') is not None])
    questions_removed = len([r for r in all_results if r.get('is_correct', False)])
    questions_with_errors = len([r for r in all_results if r.get('error')])

    print("-" * 60)
    print("🎉 MULTIPROCESSING COMPLETE!")
    print(f"📊 Processing Summary:")
    print(f"  • Total API calls made: {api_calls_made:,}")
    print(f"  • Questions found to keep (model couldn't answer): {api_calls_made - questions_removed:,}")
    print(f"  • Questions removed (model answered correctly): {questions_removed:,}")
    print(f"  • Questions with errors (kept): {questions_with_errors:,}")
    print(f"  • Model accuracy rate: {questions_removed/api_calls_made*100:.1f}%" if api_calls_made > 0 else "  • Model accuracy rate: 0.0%")
    print("-" * 60)
    print(f"\n🔄 Reconstructing filtered dataset...")

    # Reconstruct filtered instances
    filtered_instances = []
    total_questions_after = 0

    # Group results by instance
    results_by_instance = {}
    for result in all_results:
        instance_id = result['instance_id']
        if instance_id not in results_by_instance:
            results_by_instance[instance_id] = []
        results_by_instance[instance_id].append(result)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="🔧 Reconstructing instances", unit="instance"):
        original_questions = json.loads(row['questions_and_answers'])

        # Get results for this instance
        instance_results = results_by_instance.get(idx, [])

        # Build filtered questions list
        filtered_qas = []
        tested_question_ids = {r['question_id'] for r in instance_results}
        questions_removed_from_instance = 0

        # Add questions that should be kept
        for qa_idx, qa in enumerate(original_questions):
            if qa_idx in tested_question_ids:
                # Question was tested - check if we should keep it
                tested_result = next((r for r in instance_results if r['question_id'] == qa_idx), None)
                if tested_result and tested_result.get('keep_question', False):
                    filtered_qas.append(qa)
                elif tested_result and not tested_result.get('keep_question', False):
                    # Question tested and model got it right, remove it
                    questions_removed_from_instance += 1
                    if questions_removed_from_instance <= 3:  # Only show first 3 removals per instance
                        print(f"  Instance {idx}: Removed Q{qa_idx+1} - Model answered correctly")
                        print(f"    Expected: {tested_result['expected_answer']}")
                        print(f"    Model: {tested_result['prediction_without_chunk']}")
                    elif questions_removed_from_instance == 4:
                        print(f"  Instance {idx}: ... (suppressing further removal logs for this instance)")
            else:
                # Question not tested (beyond max_questions_to_test or stopped early due to reaching 100 questions to keep), keep it
                filtered_qas.append(qa)

        total_questions_after += len(filtered_qas)

        # Only keep instances that still have questions
        if filtered_qas:
            # Create new instance with filtered questions
            new_instance = row.copy()
            new_instance['questions_and_answers'] = json.dumps(filtered_qas)
            new_instance['num_questions'] = len(filtered_qas)
            filtered_instances.append(new_instance)
        else:
            print(f"  ⚠️  Removing entire instance {idx} - no questions remaining")

    # Create new DataFrame with filtered data
    if filtered_instances:
        filtered_df = pd.DataFrame(filtered_instances)

        # Save filtered dataset
        output_file = f"./data/memalpha/processed_{dataset_name}_data_filtered.parquet"
        filtered_df.to_parquet(output_file, index=False)

        # Save questions and predictions for analysis
        predictions_file = f"./data/memalpha/{dataset_name}_questions_and_predictions.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(questions_and_predictions, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}")
        print(f"🎯 FINAL FILTERING RESULTS FOR {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"📁 Dataset Statistics:")
        print(f"  • Original instances: {len(df):,}")
        print(f"  • Filtered instances: {len(filtered_df):,}")
        print(f"  • Instances removed: {len(df) - len(filtered_df):,}")
        print(f"")
        print(f"❓ Question Statistics:")
        print(f"  • Questions before filtering: {total_questions_before:,}")
        print(f"  • Questions after filtering: {total_questions_after:,}")
        print(f"  • Questions removed (model answered correctly): {questions_removed:,}")
        print(f"  • Questions kept (model couldn't answer + errors): {api_calls_made - questions_removed + questions_with_errors:,}")
        print(f"  • Questions with errors (kept): {questions_with_errors:,}")
        print(f"  • Dataset reduction: {(total_questions_before - total_questions_after)/total_questions_before*100:.1f}%")
        print(f"")
        print(f"⚡ Processing Statistics:")
        print(f"  • Total API calls made: {api_calls_made:,}")
        print(f"  • Average questions per instance: {total_questions_after/len(filtered_df):.1f}")
        print(f"  • Processing efficiency: {api_calls_made/total_questions_before*100:.1f}% of questions tested")
        print(f"  • Model accuracy rate: {questions_removed/api_calls_made*100:.1f}% (questions correctly answered)")
        print(f"")
        print(f"💾 Output Files:")
        print(f"  • Filtered dataset: {output_file}")
        print(f"  • Analysis data: {predictions_file}")
        print(f"{'='*70}")

        return output_file
    else:
        print("Error: No instances remaining after filtering!")

        # Still save questions and predictions even if no instances remain
        if questions_and_predictions:
            predictions_file = f"./data/memalpha/{dataset_name}_questions_and_predictions.json"
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(questions_and_predictions, f, indent=2, ensure_ascii=False)
            print(f"Questions & predictions saved to: {predictions_file}")

        return None

def is_answer_correct(model_answer, expected_answer):
    """Compare model answer with expected answer to determine if they match

    Args:
        model_answer: Answer from the model
        expected_answer: Expected correct answer

    Returns:
        bool: True if answers are considered equivalent
    """
    if not model_answer or not expected_answer:
        return False

    # Normalize answers
    model_answer = model_answer.lower().strip()
    expected_answer = expected_answer.lower().strip()
    if expected_answer in model_answer:
        return True
    else:
        return False

def split_dataset_by_difficulty():
    """Split training dataset by difficulty based on accuracy bins from analyze_results.py

    Difficulty levels:
    1. Easy (acc > 0.7): High accuracy instances
    2. Medium (0.6 < acc <= 0.7): Medium accuracy instances
    3. Hard (acc <= 0.6): Low accuracy instances
    """

    # Load the bin indices from analyze_results.py output
    indices_file = "figures/all_means_bin_indices.json"
    if not os.path.exists(indices_file):
        print(f"Error: Bin indices file not found: {indices_file}")
        print("Please run analyze_results.py first to generate the bin indices.")
        return

    with open(indices_file, 'r') as f:
        bin_indices = json.load(f)

    print("Loaded bin indices:")
    for bin_range, indices in bin_indices.items():
        print(f"  {bin_range}: {len(indices)} instances")

    # Load the training dataset
    train_file = "data/memalpha/train.parquet"
    if not os.path.exists(train_file):
        print(f"Error: Training file not found: {train_file}")
        return

    print(f"\nLoading training dataset from {train_file}...")
    df = pd.read_parquet(train_file)
    print(f"Loaded {len(df)} training instances")

    # Map bin ranges to difficulty levels
    easy_indices = set()    # acc > 0.7
    medium_indices = set()  # 0.6 < acc <= 0.7
    hard_indices = set()    # acc <= 0.6

    for bin_range, indices in bin_indices.items():
        bin_start = float(bin_range.split('-')[0])

        if bin_start > 0.7:  # 0.7-0.8, 0.8-0.9, 0.9-1.0
            easy_indices.update(indices)
        elif bin_start >= 0.6:  # 0.6-0.7
            medium_indices.update(indices)
        else:  # 0.4-0.5, 0.5-0.6
            hard_indices.update(indices)

    print(f"\nDifficulty distribution:")
    print(f"  Easy (acc > 0.7): {len(easy_indices)} instances")
    print(f"  Medium (0.6 < acc <= 0.7): {len(medium_indices)} instances")
    print(f"  Hard (acc <= 0.6): {len(hard_indices)} instances")
    print(f"  Total: {len(easy_indices) + len(medium_indices) + len(hard_indices)} instances")

    # Check if all indices are covered
    all_mapped_indices = easy_indices | medium_indices | hard_indices
    total_from_bins = sum(len(indices) for indices in bin_indices.values())

    if len(all_mapped_indices) != total_from_bins:
        print(f"Warning: Index count mismatch - mapped {len(all_mapped_indices)} vs expected {total_from_bins}")

    # Filter dataframe rows based on indices
    def filter_by_indices(df, indices, difficulty_name):
        """Filter dataframe to only include rows with instance_ids in the given indices"""
        if not indices:
            print(f"Warning: No indices for {difficulty_name} difficulty")
            return pd.DataFrame()

        # Convert indices to set for faster lookup
        indices_set = set(indices)

        # Filter rows where instance_id is in the indices set
        filtered_df = df[df['instance_id'].isin(indices_set)].copy()

        print(f"  {difficulty_name}: {len(filtered_df)} instances (from {len(indices)} indices)")

        if len(filtered_df) != len(indices):
            missing_count = len(indices) - len(filtered_df)
            print(f"    Warning: {missing_count} indices not found in training data")

        return filtered_df

    # Create difficulty-based datasets
    print(f"\nFiltering datasets by difficulty...")
    easy_df = filter_by_indices(df, easy_indices, "Easy")
    medium_df = filter_by_indices(df, medium_indices, "Medium")
    hard_df = filter_by_indices(df, hard_indices, "Hard")

    # Create output directory
    output_dir = "data/memalpha/difficulty_splits"
    os.makedirs(output_dir, exist_ok=True)

    # Save the split datasets
    output_files = {}

    if not easy_df.empty:
        easy_file = os.path.join(output_dir, "train_easy.parquet")
        easy_df.to_parquet(easy_file, index=False)
        output_files['easy'] = easy_file
        print(f"  Saved easy dataset: {easy_file}")

    if not medium_df.empty:
        medium_file = os.path.join(output_dir, "train_medium.parquet")
        medium_df.to_parquet(medium_file, index=False)
        output_files['medium'] = medium_file
        print(f"  Saved medium dataset: {medium_file}")

    if not hard_df.empty:
        hard_file = os.path.join(output_dir, "train_hard.parquet")
        hard_df.to_parquet(hard_file, index=False)
        output_files['hard'] = hard_file
        print(f"  Saved hard dataset: {hard_file}")

    # Calculate and display statistics
    print(f"\n{'='*60}")
    print(f"DIFFICULTY-BASED DATASET SPLIT SUMMARY")
    print(f"{'='*60}")

    def calculate_dataset_stats(df, difficulty_name):
        """Calculate and display statistics for a dataset split"""
        if df.empty:
            print(f"{difficulty_name.upper()} Dataset: EMPTY")
            return

        total_questions = df['num_questions'].sum()
        avg_questions = df['num_questions'].mean()
        total_chunks = df['num_chunks'].sum()
        avg_chunks = df['num_chunks'].mean()

        # Data source distribution
        source_dist = df['data_source'].value_counts().to_dict()

        print(f"\n{difficulty_name.upper()} Dataset (acc {'> 0.7' if difficulty_name == 'Easy' else '0.6-0.7' if difficulty_name == 'Medium' else '<= 0.6'}):")
        print(f"  • Instances: {len(df):,}")
        print(f"  • Total questions: {total_questions:,}")
        print(f"  • Avg questions per instance: {avg_questions:.1f}")
        print(f"  • Total chunks: {total_chunks:,}")
        print(f"  • Avg chunks per instance: {avg_chunks:.1f}")
        print(f"  • Data sources: {source_dist}")

    calculate_dataset_stats(easy_df, "Easy")
    calculate_dataset_stats(medium_df, "Medium")
    calculate_dataset_stats(hard_df, "Hard")

    # Overall summary
    total_instances = len(easy_df) + len(medium_df) + len(hard_df)
    total_questions = easy_df['num_questions'].sum() + medium_df['num_questions'].sum() + hard_df['num_questions'].sum()

    print(f"\nOVERALL SUMMARY:")
    print(f"  • Original training instances: {len(df):,}")
    print(f"  • Split training instances: {total_instances:,}")
    print(f"  • Coverage: {total_instances/len(df)*100:.1f}%")
    print(f"  • Total questions in splits: {total_questions:,}")

    print(f"\nOUTPUT FILES:")
    for difficulty, filepath in output_files.items():
        print(f"  • {difficulty.capitalize()}: {filepath}")

    print(f"{'='*60}")

    return output_files


def merge_into_memalpha(dataset_names, random_seed=42, status='all', output_name="memalpha", limit_size=None):
    """Combine individual dataset train/test splits into new memalpha train/test files

    Args:
        dataset_names: List of dataset names to merge
        random_seed: Random seed for reproducible shuffling and sampling
        status: Status filter (default 'all')
        output_name: Name for the output directory (default 'memalpha')
        limit_size: Maximum number of examples per training dataset (default None for no limit).
                   Test datasets are not affected by this limit.
    """

    print(f"=== Creating {output_name} from selected datasets ===")
    if limit_size is not None:
        print(f"Training data will be limited to {limit_size} examples per dataset")

    # Define paths for output files
    memalpha_train_path = f"./data/{output_name}/train.parquet"
    memalpha_test_path = f"./data/{output_name}/test.parquet"

    # Create output directory
    os.makedirs(f"./data/{output_name}", exist_ok=True)

    # Load individual dataset splits
    train_dataframes = []
    test_dataframes = []


    for dataset_name in dataset_names:

        if dataset_name in ['accurate_retrieval', 'long_range_understanding', 'conflict_resolution', 'test_time_learning']:
            # there is no train_path
            train_path = None
            test_path = f"./data/memalpha/processed_{dataset_name}_data.parquet"

        elif dataset_name == 'squad_test':
            train_path = None
            test_path = f"./data/memalpha/processed_squad_test_data.parquet"

        else:
            train_path = f"./data/memalpha/{dataset_name}/train.parquet"
            test_path = f"./data/memalpha/{dataset_name}/test.parquet"

        if train_path is not None and os.path.exists(train_path):
            train_df = pd.read_parquet(train_path)
            # For ttl_train, preserve the existing data_source field which contains sub-sources
            if dataset_name == 'ttl_train' and 'data_source' in train_df.columns:
                # Use data_source as source_dataset to preserve sub-sources
                train_df['source_dataset'] = train_df['data_source']
            elif 'source_dataset' not in train_df.columns:
                train_df['source_dataset'] = dataset_name

            # Apply size limit to training data if specified
            original_size = len(train_df)
            if limit_size is not None and len(train_df) > limit_size:
                train_df = train_df.sample(n=limit_size, random_state=random_seed).reset_index(drop=True)
                print(f"Loaded {dataset_name}: train={len(train_df)} instances (limited from {original_size})")
            else:
                print(f"Loaded {dataset_name}: train={len(train_df)} instances")

            train_dataframes.append(train_df)
        else:
            print(f"Warning: Files not found for {dataset_name}: {train_path}")

        if test_path is not None and os.path.exists(test_path):
            test_df = pd.read_parquet(test_path)

            if dataset_name == 'squad_test':
                test_df['source_dataset'] = 'squad_test'
            elif dataset_name == 'ttl_train' and 'data_source' in test_df.columns:
                # For ttl_train, preserve sub-sources from data_source field
                test_df['source_dataset'] = test_df['data_source']
            else:
                if 'source_dataset' not in test_df.columns:
                    if 'sub_source' in test_df.columns:
                        test_df['source_dataset'] = test_df['sub_source']
                    else:
                        test_df['source_dataset'] = dataset_name

            test_dataframes.append(test_df)
            print(f"Loaded {dataset_name}: test={len(test_df)} instances")
        else:
            print(f"Warning: Files not found for {dataset_name}: {test_path}")

    # Combine train and test dataframes separately, handling empty lists
    combined_train_df = pd.concat(train_dataframes, ignore_index=True) if train_dataframes else pd.DataFrame()
    combined_test_df = pd.concat(test_dataframes, ignore_index=True) if test_dataframes else pd.DataFrame()

    if combined_train_df.empty and combined_test_df.empty:
        print("Error: No valid dataset files found!")
        return None, None

    if not combined_train_df.empty:
        print(f"\nCombined train dataset: {len(combined_train_df)} instances")
        print("\nTrain dataset composition:")
        print(combined_train_df['source_dataset'].value_counts())

    if not combined_test_df.empty:
        print(f"\nCombined test dataset: {len(combined_test_df)} instances")
        print("\nTest dataset composition:")
        print(combined_test_df['source_dataset'].value_counts())

    # Helper function to standardize and filter a dataset
    def process_split(df, split_name):
        """Standardize format and filter a train or test split"""
        print(f"\nProcessing {split_name}...")

        if df.empty:
            print(f"Skipping {split_name} as it is empty.")
            return []

        # Standardize format and convert to list of dictionaries
        standardized_instances = []
        for idx, row in df.iterrows():
            try:
                standardized_instance = standardize_instance_format(row)
                standardized_instances.append(standardized_instance)
            except Exception as e:
                print(f"Error standardizing {split_name} instance {idx}: {e}")
                continue

        print(f"Successfully standardized {len(standardized_instances)} {split_name} instances")

        return standardized_instances

    # Process train and test splits
    train_instances = process_split(combined_train_df, "train")
    test_instances = process_split(combined_test_df, "test")

    if not train_instances and not test_instances:
        print("Error: No valid instances remaining after processing!")
        return None, None

    # Shuffle both splits with the same random seed for consistency
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)
    random.shuffle(train_instances)
    # random.shuffle(test_instances)

    # Convert standardized instances back to DataFrame format for parquet saving
    def instances_to_dataframe(instances):
        """Convert standardized instances to DataFrame format suitable for parquet"""
        rows = []
        for i, instance in enumerate(instances):
            # Convert chunks and questions_and_answers back to JSON strings for parquet storage
            chunks_str = json.dumps(instance['chunks'])
            qa_str = json.dumps(instance['questions_and_answers'])
            metadata_str = json.dumps(instance['metadata'])

            for qa_pair in instance['questions_and_answers']:
                if not isinstance(qa_pair['answer'], list):
                    assert isinstance(qa_pair['answer'], str)

            assert len(instance['questions_and_answers']) <= 100, f"Questions and answers length is {len(instance['questions_and_answers'])}"

            row = {
                'instance_id': i,
                'prompt': instance['prompt'],
                'chunks': chunks_str,
                'questions_and_answers': qa_str,
                'data_source': instance['data_source'],
                'metadata': metadata_str,
                'num_chunks': len(instance['chunks']),
                'num_questions': len(instance['questions_and_answers'])
            }
            rows.append(row)
        return pd.DataFrame(rows)

    # Save merged train and test sets as parquet files
    train_df = instances_to_dataframe(train_instances)
    test_df = instances_to_dataframe(test_instances)

    train_df.to_parquet(memalpha_train_path, index=False)
    test_df.to_parquet(memalpha_test_path, index=False)

    print(f"\nSaved new {output_name} files:")
    print(f"Training set: {memalpha_train_path} ({len(train_instances)} instances)")
    print(f"Test set: {memalpha_test_path} ({len(test_instances)} instances)")

    # Calculate statistics for each split
    def calculate_split_stats(instances, split_name):
        if not instances:
            return

        source_counts = {}
        total_questions = 0
        total_chunks = 0

        for instance in instances:
            source = instance['data_source']
            source_counts[source] = source_counts.get(source, 0) + 1
            total_questions += len(instance['questions_and_answers'])
            total_chunks += len(instance['chunks'])

        avg_questions = total_questions / len(instances)
        avg_chunks = total_chunks / len(instances)

        print(f"\n{split_name} statistics:")
        print(f"  Source distribution: {source_counts}")
        print(f"  Total questions: {total_questions}")
        print(f"  Average questions per instance: {avg_questions:.1f}")
        print(f"  Average chunks per instance: {avg_chunks:.1f}")

    calculate_split_stats(train_instances, "Final training set")
    calculate_split_stats(test_instances, "Final test set")

    # Validate format of a few examples from each split
    print(f"\nValidating merged format...")
    for split_name, instances in [("train", train_instances), ("test", test_instances)]:
        print(f"\n{split_name.capitalize()} examples:")
        for i, instance in enumerate(instances[:2]):
            print(f"  Example {i+1}:")
            print(f"    Source: {instance['data_source']}")
            print(f"    Prompt: {instance['prompt'][:80]}...")
            print(f"    Chunks: {len(instance['chunks'])} chunks")
            print(f"    Q&As: {len(instance['questions_and_answers'])} questions")
            if instance['questions_and_answers']:
                assert isinstance(instance['questions_and_answers'], list)
                for qa in instance['questions_and_answers']:
                    if not isinstance(qa['answer'], str):
                        assert isinstance(qa['answer'], list)
                        qa['answer'] = qa['answer'][0]
                first_qa = instance['questions_and_answers'][0]
                print(f"    Sample Q&A keys: {list(first_qa.keys())}")
                assert 'question' in first_qa and 'answer' in first_qa, f"Invalid Q&A format in {split_name} instance {i}"

    print("\nFormat validation passed!")
    print(f"Successfully created {output_name} from selected datasets!")

    return memalpha_train_path, memalpha_test_path

def process_cr_train():
    """Process CR train dataset into the desired format"""
    print("Processing CR train dataset...")

    # Create output directory
    os.makedirs("./data/cr_train", exist_ok=True)

    df = pd.read_parquet("./data/train-00000-of-00001.parquet")
    print(f"Loaded {len(df)} instances from CR train dataset")

    # Process instances
    processed_data = []

    for item_idx, row in df.iterrows():
        # Extract data from row
        context_chunks = row['context']  # This is an array of text chunks
        questions = row['questions']
        answers = row['answers']
        metadata = row['metadata']

        # Format chunks with conversational templates
        base_date = datetime(2024, 1, 1)
        formatted_chunks = []

        for chunk_idx, chunk_content in enumerate(context_chunks):
            # Apply regex formatting to separate numbered items with newlines
            # Using Method 2: lookbehind and lookahead
            pattern = r'(?<=\.) (?=\d+\. )'
            formatted_content = re.sub(pattern, '\n', chunk_content)

            # Format using conversational template with random selection
            user_template = random.choice(USER_MESSAGE_TEMPLATES)
            assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

            # Create a timestamp for each chunk (incrementing by days)
            chunk_date = base_date + timedelta(days=chunk_idx)
            timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

            # Format the chunk with conversational template
            formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template} (There might be conflicts between the following new facts and previous facts saved in the memory, please update the facts according to the following new facts if you see any conflicts)\n{formatted_content}\n<Assistant>{assistant_template}"
            formatted_chunks.append(formatted_chunk)

        # Create Q&A pairs
        qa_pairs = []
        for q, a in zip(questions, answers):
            qa_pairs.append({
                'question': q,
                'answer': a
            })

        # Create the instance
        instance = {
            'prompt': 'I will provide you with sequential information chunks. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
            'chunks': formatted_chunks,
            'questions_and_answers': qa_pairs,
            'data_source': 'cr_train'
        }

        processed_data.append(instance)

        if (item_idx + 1) % 10 == 0:
            print(f"Processed {item_idx + 1}/{len(df)} instances")

    print(f"\nProcessed {len(processed_data)} instances")

    return processed_data


def process_lme_train_dataset():
    """Process longmemeval_combined_train dataset with the same structure as longmemeval_s*"""
    print("Processing lme_train dataset...")

    # Load the dataset
    with open("./data/longmemeval_combined_train_v2.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load the Qwen tokenizer for accurate token counting
    from transformers import AutoTokenizer
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    def count_tokens_qwen(text):
        """Count tokens using Qwen3-32B tokenizer"""
        return len(qwen_tokenizer.encode(text))

    processed_data = []

    # Process each sample
    for sample in tqdm(data['samples'], desc="Processing lme_train"):
        sample_id = sample['sample_id']
        questions = sample['questions']
        answers = sample['answers']
        question_types = sample['question_types']
        haystack_dates = sample['haystack_dates']
        haystack_sessions = sample['haystack_sessions']

        # Build the context in the same format as longmemeval_s*
        # Alternating list: [timestamp:str, session:list, timestamp:str, session:list, ...]
        all_context = []
        for date, session in zip(haystack_dates, haystack_sessions):
            all_context.append(date)
            all_context.append(session)

        # Process each session to create chunks (following longmemeval_s* logic)
        def render_session(timestamp_str, turns, continuation=False):
            header = f"[Dialogue at timestamp {timestamp_str}]"
            turns_text = "\n".join(
                f"<{turn['role'].capitalize()}>{turn['content']}" for turn in turns
            )
            return f"{header}\n{turns_text}"

        def split_large_turn(turn, max_tokens=2048):
            """Split a large turn into smaller chunks while preserving structure"""
            content = turn['content']
            role = turn['role']

            # If the turn is small enough, return as is
            test_turn = {'role': role, 'content': content}
            if count_tokens_qwen(f"<{role.capitalize()}>{content}") <= max_tokens - 150:  # Leave some buffer for headers
                return [test_turn]

            # Split content into sentences or smaller chunks
            import re
            # Try to split by sentences first
            sentences = re.split(r'(?<=[.!?])\s+', content)

            split_turns = []
            current_content = ""

            for sentence in sentences:
                test_content = current_content + (" " if current_content else "") + sentence
                test_text = f"<{role.capitalize()}>{test_content}"

                if count_tokens_qwen(test_text) <= max_tokens - 150:
                    current_content = test_content
                else:
                    if current_content:
                        # Save current chunk
                        split_turns.append({'role': role, 'content': current_content.strip()})
                        current_content = sentence
                    else:
                        # Single sentence is too long, need to split it further
                        words = sentence.split()
                        chunk = ""
                        for word in words:
                            test_chunk = chunk + (" " if chunk else "") + word
                            if count_tokens_qwen(f"<{role.capitalize()}>{test_chunk}") <= max_tokens - 150:
                                chunk = test_chunk
                            else:
                                if chunk:
                                    split_turns.append({'role': role, 'content': chunk.strip()})
                                chunk = word
                        if chunk:
                            current_content = chunk

            # Add any remaining content
            if current_content:
                split_turns.append({'role': role, 'content': current_content.strip()})

            return split_turns

        def split_session_into_segments(timestamp_str, turns):
            # Split session into segments, splitting large turns if needed
            # Each segment should be <= 2048 tokens
            segments = []
            current_turns = []

            for turn in turns:
                # Check if this single turn is too large
                single_turn_text = f"<{turn['role'].capitalize()}>{turn['content']}"
                if count_tokens_qwen(single_turn_text) > 2048 - 150:  # Leave buffer for header
                    # Need to split this turn
                    split_turns = split_large_turn(turn, max_tokens=2048)

                    for split_turn in split_turns:
                        # Try to add each split part
                        test_turns = current_turns + [split_turn]
                        is_cont = len(segments) > 0
                        candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)

                        if count_tokens_qwen(candidate_text) <= 2048:
                            current_turns = test_turns
                        else:
                            # Flush current segment and start new one
                            if current_turns:
                                # Check if last turn is User - if so, move it to next segment
                                if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                    last_turn = current_turns.pop()
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [last_turn, split_turn]
                                else:
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [split_turn]
                            else:
                                current_turns = [split_turn]
                else:
                    # Normal processing for turns that fit
                    test_turns = current_turns + [turn]
                    is_cont = len(segments) > 0
                    candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)

                    if count_tokens_qwen(candidate_text) <= 2048:
                        # Turn fits, add it to current segment
                        current_turns = test_turns
                    else:
                        # Adding this turn would exceed limit
                        if current_turns:
                            # Check if last turn is User - if so, move it to next segment
                            if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                last_turn = current_turns.pop()
                                segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                current_turns = [last_turn, turn]
                            else:
                                # Flush current segment and start new one with this turn
                                segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                current_turns = [turn]
                        else:
                            # This shouldn't happen now since we split large turns
                            segments.append(render_session(timestamp_str, [turn], continuation=is_cont))
                            current_turns = []

            # Add any remaining turns as the final segment
            if current_turns:
                # Check if last turn is User - if so, we might want to keep it with previous content
                # For the final segment, we'll allow User turns since there's no next segment
                segments.append(
                    render_session(timestamp_str, current_turns, continuation=len(segments) > 0)
                )

            return segments

        # Build segments per session (each <= 2048 tokens), then aggregate into chunks
        # Aim for chunks around 2048 tokens but enforce a maximum of 2048 tokens
        chunks = []
        current_chunk = ""
        for idx in range(0, len(all_context), 2):
            ts = all_context[idx]
            session_turns = all_context[idx + 1]
            session_segments = split_session_into_segments(ts, session_turns)
            for segment_text in session_segments:
                if current_chunk:
                    candidate = current_chunk + "\n\n" + segment_text
                else:
                    candidate = segment_text

                candidate_tokens = count_tokens_qwen(candidate)

                if candidate_tokens > 2048:
                    # Adding this segment would exceed the maximum
                    # Save current chunk if it has content
                    if current_chunk:
                        chunks.append(current_chunk)
                    # Start new chunk with this segment
                    current_chunk = segment_text
                elif candidate_tokens >= 1800:  # Close enough to target, make it a chunk
                    chunks.append(candidate)
                    current_chunk = ""
                else:
                    # Still room to add more
                    current_chunk = candidate

        if current_chunk:
            # Check if the last chunk ends with a User turn
            # We'll inspect the chunk to see if it ends with "<User>"
            lines = current_chunk.strip().split('\n')
            if lines and lines[-1].startswith('<User>'):
                # If possible, we should have moved this to the next chunk
                # But since this is the last chunk, we'll keep it
                pass
            chunks.append(current_chunk)

        # Verify all chunks are within token limit
        for chunk in chunks:
            token_count = count_tokens_qwen(chunk)
            if token_count > 2048:
                print(f"Warning: Chunk in {sample_id} has {token_count} tokens (exceeds 2048)")

        # Format questions and answers
        questions_and_answers = []
        for q, a in zip(questions, answers):
            questions_and_answers.append({'question': q, 'answer': a})

        # Create instance
        instance = {
            'prompt': "I will provide you with the conversation history between the user and the assistant and I need you to remember the details of the conversation for future reference.",
            'instance_id': f"lme_train_{sample_id}",
            'context_chunks': chunks,
            'questions_and_answers': questions_and_answers,
            'metadata': {
                'source': 'lme_train',
                'sample_id': sample_id,
                'question_types': question_types,
                'haystack_dates': haystack_dates
            }
        }

        processed_data.append(instance)

    print(f"\nProcessed {len(processed_data)} instances from lme_train dataset")
    return processed_data


def process_memory_agent_bench(split='Accurate_Retrieval'):

    ar_ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)

    # Load the Qwen tokenizer for accurate token counting for longmemeval_s*
    from transformers import AutoTokenizer
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    def count_tokens_qwen(text):
        """Count tokens using Qwen3-32B tokenizer"""
        return len(qwen_tokenizer.encode(text))

    processed_data = []

    if split == 'Long_Range_Understanding':
        with open("./data/memoryagentbench_answers_to_keywords_mapping.json", 'r') as file:
            answers_to_keywords_mapping = json.load(file)

    for item_idx, item in tqdm(enumerate(ar_ds), desc=f"Processing {split}", total=len(ar_ds)):
        # Filter out sources containing 'ruler_qa'
        if 'niah' in item['metadata'].get('source', ''):
            continue

        if item['metadata']['source'] in ['longmemeval_s_-1_500', 'eventqa_65536', 'eventqa_131072', 'eventqa_full', 'infbench_qa_eng_shots2']:
            continue

        if item['metadata']['source'] in ['recsys_redial_full']:
            continue

        # if item['metadata']['source'] in ['icl_trec_coarse_6600shot_balance', 'icl_nlu_8296shot_balance']:
        #     continue

        context = item['context']
        questions = item['questions']
        answers = item['answers']

        # Create chunks from the context using sentence tokenization

        if item['metadata']['source'] == 'longmemeval_s*':
            # Context is an alternating list: [timestamp:str, session:list, timestamp:str, session:list, ...]
            # Each session is a list of turns with keys 'role' and 'content'.
            # We will render each pair into text and then group into chunks with at least 2048 tokens.
            try:
                all_context = eval(context) if isinstance(context, str) else context
            except Exception:
                all_context = json.loads(context)

            assert isinstance(all_context, list), "Expected all_context to be a list"
            assert len(all_context) % 2 == 0, "Expected alternating [timestamp, session] pairs"
            for idx in range(0, len(all_context), 2):
                ts = all_context[idx]
                session = all_context[idx + 1]
                assert isinstance(ts, str), f"Expected timestamp at index {idx} to be str"
                assert isinstance(session, list), f"Expected session at index {idx+1} to be list"
                for turn in session:
                    assert isinstance(turn, dict) and 'role' in turn and 'content' in turn, "Invalid turn format"

            def render_session(timestamp_str, turns, continuation=False):
                header = f"[Dialogue at timestamp {timestamp_str}]"
                turns_text = "\n".join(
                    f"<{turn['role'].capitalize()}>{turn['content']}" for turn in turns
                )
                return f"{header}\n{turns_text}"

            def split_large_turn(turn, max_tokens=2048):
                """Split a large turn into smaller chunks while preserving structure"""
                content = turn['content']
                role = turn['role']

                # If the turn is small enough, return as is
                test_turn = {'role': role, 'content': content}
                if count_tokens_qwen(f"<{role.capitalize()}>{content}") <= max_tokens - 150:  # Leave some buffer for headers
                    return [test_turn]

                # Split content into sentences or smaller chunks
                import re
                # Try to split by sentences first
                sentences = re.split(r'(?<=[.!?])\s+', content)

                split_turns = []
                current_content = ""

                for sentence in sentences:
                    test_content = current_content + (" " if current_content else "") + sentence
                    test_text = f"<{role.capitalize()}>{test_content}"

                    if count_tokens_qwen(test_text) <= max_tokens - 150:
                        current_content = test_content
                    else:
                        if current_content:
                            # Save current chunk
                            split_turns.append({'role': role, 'content': current_content.strip()})
                            current_content = sentence
                        else:
                            # Single sentence is too long, need to split it further
                            words = sentence.split()
                            chunk = ""
                            for word in words:
                                test_chunk = chunk + (" " if chunk else "") + word
                                if count_tokens_qwen(f"<{role.capitalize()}>{test_chunk}") <= max_tokens - 150:
                                    chunk = test_chunk
                                else:
                                    if chunk:
                                        split_turns.append({'role': role, 'content': chunk.strip()})
                                    chunk = word
                            if chunk:
                                current_content = chunk

                # Add any remaining content
                if current_content:
                    split_turns.append({'role': role, 'content': current_content.strip()})

                return split_turns

            def split_session_into_segments(timestamp_str, turns):
                # Split session into segments, splitting large turns if needed
                # Each segment should be <= 2048 tokens
                segments = []
                current_turns = []

                for turn in turns:
                    # Check if this single turn is too large
                    single_turn_text = f"<{turn['role'].capitalize()}>{turn['content']}"
                    if count_tokens_qwen(single_turn_text) > 2048 - 150:  # Leave buffer for header
                        # Need to split this turn
                        split_turns = split_large_turn(turn, max_tokens=2048)

                        for split_turn in split_turns:
                            # Try to add each split part
                            test_turns = current_turns + [split_turn]
                            is_cont = len(segments) > 0
                            candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)

                            if count_tokens_qwen(candidate_text) <= 2048:
                                current_turns = test_turns
                            else:
                                # Flush current segment and start new one
                                if current_turns:
                                    # Check if last turn is User - if so, move it to next segment
                                    if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                        last_turn = current_turns.pop()
                                        segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                        current_turns = [last_turn, split_turn]
                                    else:
                                        segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                        current_turns = [split_turn]
                                else:
                                    current_turns = [split_turn]
                    else:
                        # Normal processing for turns that fit
                        test_turns = current_turns + [turn]
                        is_cont = len(segments) > 0
                        candidate_text = render_session(timestamp_str, test_turns, continuation=is_cont)

                        if count_tokens_qwen(candidate_text) <= 2048:
                            # Turn fits, add it to current segment
                            current_turns = test_turns
                        else:
                            # Adding this turn would exceed limit
                            if current_turns:
                                # Check if last turn is User - if so, move it to next segment
                                if current_turns[-1]['role'].lower() == 'user' and len(current_turns) > 1:
                                    last_turn = current_turns.pop()
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [last_turn, turn]
                                else:
                                    # Flush current segment and start new one with this turn
                                    segments.append(render_session(timestamp_str, current_turns, continuation=is_cont))
                                    current_turns = [turn]
                            else:
                                # This shouldn't happen now since we split large turns
                                segments.append(render_session(timestamp_str, [turn], continuation=is_cont))
                                current_turns = []

                # Add any remaining turns as the final segment
                if current_turns:
                    # Check if last turn is User - if so, we might want to keep it with previous content
                    # For the final segment, we'll allow User turns since there's no next segment
                    segments.append(
                        render_session(timestamp_str, current_turns, continuation=len(segments) > 0)
                    )

                return segments

            # Build segments per session (each <= 2048 tokens), then aggregate into chunks
            # Aim for chunks around 2048 tokens but enforce a maximum of 2048 tokens
            chunks = []
            current_chunk = ""
            for idx in range(0, len(all_context), 2):
                ts = all_context[idx]
                session_turns = all_context[idx + 1]
                session_segments = split_session_into_segments(ts, session_turns)
                for segment_text in session_segments:
                    if current_chunk:
                        candidate = current_chunk + "\n\n" + segment_text
                    else:
                        candidate = segment_text

                    candidate_tokens = count_tokens_qwen(candidate)

                    if candidate_tokens > 2048:
                        # Adding this segment would exceed the maximum
                        # Save current chunk if it has content
                        if current_chunk:
                            chunks.append(current_chunk)
                        # Start new chunk with this segment
                        current_chunk = segment_text
                    elif candidate_tokens >= 1800:  # Close enough to target, make it a chunk
                        chunks.append(candidate)
                        current_chunk = ""
                    else:
                        # Still room to add more
                        current_chunk = candidate

            if current_chunk:
                # Check if the last chunk ends with a User turn
                # We'll inspect the chunk to see if it ends with "<User>"
                lines = current_chunk.strip().split('\n')
                if lines and lines[-1].startswith('<User>'):
                    # If possible, we should have moved this to the next chunk
                    # But since this is the last chunk, we'll keep it
                    pass
                chunks.append(current_chunk)

            for chunk in chunks:
                assert count_tokens(chunk) <= 2048

        elif split == 'Test_Time_Learning':
            chunks = create_chunks_use_sent_tokenizer(context, max_tokens=1024)

        else:
            chunks = create_chunks_use_sent_tokenizer(context, max_tokens=2048)

        # Format questions and answers
        questions_and_answers = []
        for q, a in zip(questions, answers):
            questions_and_answers.append({'question': q, 'answer': a})

        # Format chunks with different templates based on split
        if split == 'Long_Range_Understanding':
            # BookSum-style date-based template for long_range_understanding
            base_date = datetime(2024, 1, 1)
            formatted_chunks = []
            chunk_dates = []
            current_date = base_date

            for chunk_idx, chunk_content in enumerate(chunks):
                # Create progressive dates (incrementing by random 1-3 days)
                if chunk_idx > 0:  # Don't add days for the first chunk
                    days_to_add = random.randint(1, 3)
                    current_date = current_date + timedelta(days=days_to_add)
                date_str = current_date.strftime("%Y-%m-%d")
                chunk_dates.append(date_str)

                # Format with the BookSum template
                formatted_chunk = f"[Event happened on {date_str} The user is reading a book]\n<User> {chunk_content}\n\n<System> Please remember what the user reads on {date_str}, save the details within the book, and retain a summary of the book the user has read so far."
                formatted_chunks.append(formatted_chunk)

            # Update questions to use date ranges for long_range_understanding
            if chunk_dates:
                first_date = chunk_dates[0]
                last_date = chunk_dates[-1]
                updated_questions_and_answers = []
                for qa in questions_and_answers:
                    updated_qa = qa.copy()
                    # updated_qa['question'] = f"Based on the content I read from {first_date} to {last_date}, {qa['question']}"
                    updated_qa['question'] = f"Summarize the content of the book I read from {first_date} to {last_date}."
                    updated_qa['answer'] = answers_to_keywords_mapping[str(item_idx)]['keywords']
                    updated_questions_and_answers.append(updated_qa)
                questions_and_answers = updated_questions_and_answers

            chunks = formatted_chunks

        elif split == 'Accurate_Retrieval' and item['metadata']['source'] == 'infbench_qa_eng_shots2':
            # BookSum-style date-based template for infbench_qa_eng_shots2
            base_date = datetime(2024, 1, 1)
            formatted_chunks = []
            chunk_dates = []
            current_date = base_date

            for chunk_idx, chunk_content in enumerate(chunks):
                # Create progressive dates (incrementing by random 1-3 days)
                if chunk_idx > 0:  # Don't add days for the first chunk
                    days_to_add = random.randint(1, 3)
                    current_date = current_date + timedelta(days=days_to_add)
                date_str = current_date.strftime("%Y-%m-%d")
                chunk_dates.append(date_str)

                # Format with the BookSum template
                formatted_chunk = f"[Event happened on {date_str} The user is reading a book]\n<User> {chunk_content}\n\n<System> Please remember what the user reads on {date_str}, save the details within the book, and retain a summary of the book the user has read so far."
                formatted_chunks.append(formatted_chunk)

            # Update questions to use date ranges for infbench_qa_eng_shots2
            if chunk_dates:
                first_date = chunk_dates[0]
                last_date = chunk_dates[-1]
                updated_questions_and_answers = []
                for qa in questions_and_answers:
                    updated_qa = qa.copy()
                    updated_qa['question'] = f"Based on the content I read from {first_date} to {last_date}, {qa['question']}"
                    updated_questions_and_answers.append(updated_qa)
                questions_and_answers = updated_questions_and_answers

            chunks = formatted_chunks

        elif split == 'Accurate_Retrieval' and item['metadata']['source'] != 'longmemeval_s*' and item['metadata']['source'] != 'infbench_qa_eng_shots2':
            # Conversational templates (like SQuAD/HotpotQA) for accurate_retrieval (excluding longmemeval_s* and infbench_qa_eng_shots2)
            base_date = datetime(2024, 1, 1)
            formatted_chunks = []

            for chunk_idx, chunk_content in enumerate(chunks):
                # Format using conversational template with random selection
                user_template = random.choice(USER_MESSAGE_TEMPLATES)
                assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

                # Create a timestamp for each chunk (incrementing by days)
                chunk_date = base_date + timedelta(days=chunk_idx)
                timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

                # Format the chunk with conversational template
                formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n{chunk_content}\n<Assistant>{assistant_template}"
                formatted_chunks.append(formatted_chunk)

            chunks = formatted_chunks

        elif split == 'Test_Time_Learning':
            # Classification templates for test_time_learning
            base_date = datetime(2024, 1, 1)
            formatted_chunks = []

            for chunk_idx, chunk_content in enumerate(chunks):
                # Format using classification-specific templates
                user_template = random.choice(CLASSIFICATION_USER_TEMPLATES)
                assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

                # Create a timestamp for each chunk (incrementing by days)
                chunk_date = base_date + timedelta(days=chunk_idx)
                timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

                new_chunk_content = ''
                for x in chunk_content.split("\n\n"):
                    try:
                        sentence, label = x.split("label:")
                    except:
                        continue
                    sentence = sentence.strip()
                    if len(sentence) == 0:
                        continue
                    label = label.strip()
                    new_chunk_content += f"Sentence: {sentence}\nLabel: {label}\n\n"
                chunk_content = new_chunk_content.strip()

                # Format the chunk with classification template
                formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n{chunk_content}\n<Assistant>{assistant_template}"
                formatted_chunks.append(formatted_chunk)

            chunks = formatted_chunks

        # Create the data instance
        data_instance = {
            'prompt': 'I will provide you with sequential information chunks. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
            'chunks': chunks,
            'questions_and_answers': questions_and_answers,
            'data_source': 'accurate_retrieval' if split != 'Long_Range_Understanding' else 'long_range_understanding',
            'sub_source': item['metadata']['source']
        }

        # Add reading dates for long_range_understanding
        if split == 'Long_Range_Understanding' and chunk_dates:
            data_instance['reading_dates'] = {'start': chunk_dates[0], 'end': chunk_dates[-1]}

        processed_data.append(data_instance)

    return processed_data

def process_detectiveqa_dataset():
    """
    Process the DetectiveQA dataset - assumes detectiveqa_qa.json exists
    """
    print("Loading DetectiveQA dataset...")

    with open('data/detectiveqa_qa.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

        # each item has the following keys:
        # dict_keys(['question', 'answer', 'context', 'novel_id', 'question_id', 'answer_key', 'options', 'clue_position', 'answer_position', 'source'])

    with open('data/detectiveqa_novels.json', 'r', encoding='utf-8') as f:
        novels = json.load(f)
        # it has the following keys:
        # dict_keys(['100', '103', '104', '105', '106', '107', '108', '109', '110', '114', '116', '117', '118', '120', '121', '124', '126', '127', '128', '130', '132', '133', '134', '136', '137', '138', '140', '142', '144', '145', '149', '15', '150', '151', '16', '198', '203', '209', '219', '241', '25', '252', '26', '27', '28', '29', '30', '31', '33', '40', '53', '56', '79', '81', '82', '83', '84', '87', '90', '93', '97', '99'])

    print(f"Loaded {len(data)} DetectiveQA examples from {len(novels)} novels")

    # Group questions and answers by novel_id
    questions_by_novel = {}
    for qa_item in data:
        novel_id = str(qa_item['novel_id'])
        if novel_id not in questions_by_novel:
            questions_by_novel[novel_id] = []

        options = json.loads(qa_item['options'])
        question = qa_item['question']

        question = f"Question: {question}\nOptions: {options}"

        answer_choice = None
        for key, option in options.items():
            if option == qa_item['answer']:
                answer_choice = key
                break
        assert answer_choice is not None, f"No answer choice found for {question}"

        questions_by_novel[novel_id].append({
            'question': question,
            'answer': answer_choice
        })

    processed_data = []
    processed_count = 0
    skipped_count = 0

    print("Processing novels...")
    for novel_id, novel in novels.items():
        if novel_id not in questions_by_novel:
            print(f"Skipping novel {novel_id} - no questions found")
            skipped_count += 1
            continue

        novel_text = novel['text']
        # Count tokens in the novel
        num_tokens = count_tokens(novel_text)
        print(f"Processing novel {novel_id}: {num_tokens} tokens, {len(questions_by_novel[novel_id])} questions")

        # Create chunks for this novel
        novel_chunks_raw = []
        if num_tokens > 4096:
            chunks = create_chunks_use_sent_tokenizer(novel_text, max_tokens=4096)
            print(f"  Created {len(chunks)} chunks with lengths: {[count_tokens(chunk) for chunk in chunks]}")
            novel_chunks_raw = chunks
        else:
            novel_chunks_raw = [novel_text]

        # Filter out novels with fewer than 5 chunks
        if len(novel_chunks_raw) < 5:
            print(f"  Skipping novel {novel_id} - only {len(novel_chunks_raw)} chunks (minimum 5 required)")
            skipped_count += 1
            continue

        # Format chunks with BookSum-style date-based template
        base_date = datetime(2024, 1, 1)
        novel_chunks = []
        novel_dates = []  # Store dates for potential use
        current_date = base_date

        for chunk_idx, chunk_content in enumerate(novel_chunks_raw):
            # Create progressive dates (incrementing by random 1-3 days)
            if chunk_idx > 0:  # Don't add days for the first chunk
                days_to_add = random.randint(1, 3)
                current_date = current_date + timedelta(days=days_to_add)
            date_str = current_date.strftime("%Y-%m-%d")
            novel_dates.append(date_str)

            # Format with the BookSum template
            formatted_chunk = f"[Event happened on {date_str} The user is reading a book]\n<User> {chunk_content}\n\n<System> Please remember what the user reads on {date_str}, save the details within the book, and retain a summary of the book the user has read so far."
            novel_chunks.append(formatted_chunk)

        # Create a data instance for this novel
        data_instance = {
            'prompt': 'I will provide you with sequential chunks from a detective novel. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
            'chunks': novel_chunks,
            'questions_and_answers': questions_by_novel[novel_id],
            'data_source': 'detectiveqa',
            'novel_id': novel_id,
            'reading_dates': {'start': novel_dates[0] if novel_dates else "2024-01-01", 'end': novel_dates[-1] if novel_dates else "2024-01-01"}
        }

        processed_data.append(data_instance)
        processed_count += 1

    print(f"Processed {processed_count} novels, skipped {skipped_count} novels")

    # Filter out instances with too many chunks (similar to booksum)
    max_chunks_allowed = 100
    filtered_instances = []
    instances_with_too_many_chunks = 0

    for instance in processed_data:
        num_chunks = len(instance['chunks'])
        if num_chunks <= max_chunks_allowed:
            filtered_instances.append(instance)
        else:
            instances_with_too_many_chunks += 1
            print(f"Removing DetectiveQA instance (novel {instance['novel_id']}) with {num_chunks} chunks (> {max_chunks_allowed})")

    print(f"\n📊 DetectiveQA Chunk Filtering Summary:")
    print(f"  • Removed {instances_with_too_many_chunks} instances with > {max_chunks_allowed} chunks")
    if processed_data:
        print(f"  • Instances: {len(processed_data):,} → {len(filtered_instances):,} (kept {len(filtered_instances)/len(processed_data)*100:.1f}%)")
    else:
        print(f"  • Instances: 0 → 0")

    print(f"Final processed dataset: {len(filtered_instances)} novels")
    return filtered_instances

class PerLTQAProcessor:
    def __init__(self, perltmem_path: str, perltqa_path: str):
        self.perltmem_path = perltmem_path
        self.perltqa_path = perltqa_path
        self.target_chunk_size = 2048  # Target tokens per chunk
        self.min_chunk_tokens = 512  # Minimum tokens per chunk
        self.max_num_chunks = 10

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load both perltmem and perltqa datasets."""
        with open(self.perltmem_path, 'r', encoding='utf-8') as f:
            perltmem_data = json.load(f)

        with open(self.perltqa_path, 'r', encoding='utf-8') as f:
            perltqa_data = json.load(f)

        return perltmem_data, perltqa_data

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        return len(text) // 4

    def extract_conversations_from_dialogues(self, dialogues: Dict, events: Dict, character_name: str) -> List[Dict]:
        """Extract individual conversations from dialogues section with event content."""

        conversations = []

        for event_id, event_data in events.items():
            corresponding_dialogue_ids = []
            for dialogue_id, dialogue_data in dialogues.items():
                if dialogue_id.startswith(event_id):
                    corresponding_dialogue_ids.append(dialogue_id)

            dialogues_corresponding_to_event = []
            for dialogue_id in corresponding_dialogue_ids:
                dialogue_data = dialogues[dialogue_id]
                contents = dialogue_data.get('contents', {})

                for timestamp, messages in contents.items():
                    # Handle both list and string message formats
                    if isinstance(messages, list):
                        text = ' '.join(str(msg) for msg in messages)
                        token_count = sum(self.estimate_tokens(str(msg)) for msg in messages)
                    else:
                        text = str(messages)
                        token_count = self.estimate_tokens(text)

                    if isinstance(messages[0], list):
                        messages = messages[0]

                    print(type(messages[0]))

                    conversation = {
                        'type': 'dialogue',
                        'dialogue_id': dialogue_id,
                        'event_id': event_id,
                        'timestamp': timestamp,
                        'messages': messages,
                        'text': text,
                        'token_count': token_count,

                    }
                    dialogues_corresponding_to_event.append(conversation)

            conversations.append({
                'type': 'event',
                'event_id': event_id,
                'event_content': event_data.get('content', ''),
                'event_summary': event_data.get('summary', ''),
                'creation_time': event_data.get('Creation Time', ''),
                'dialogue_ids': corresponding_dialogue_ids,
                'dialogues': dialogues_corresponding_to_event,
                'theme': event_data.get('Theme', ''),
                'user_name': character_name
            })

        return conversations

    def create_chunks(self, conversation: Dict):
        """Create chunks of conversations with target size ~2k tokens and minimum token requirement."""
        num_conversation_used = 0

        event_data = conversation['event_content']
        user_name = conversation['user_name']

        chunk_text = f'The following is the event happened about the user {user_name} on {conversation["creation_time"]}:\n' + "Summary: " + conversation['event_summary'] + "\n" + "Content: " + conversation['event_content']
        chunk_text += '\n\n'
        chunk_text += 'The following are the dialogues.'
        chunk_text += '\n\n'
        for dialogue in conversation['dialogues']:
            chunk_text += f'Dialogue happened at {dialogue["timestamp"]}\n'
            # chunk_text += "\n".join(dialogue['messages'])
            for message in dialogue['messages']:
                if isinstance(message, list):
                    import ipdb; ipdb.set_trace()
                if len(message.split(": ")) != 2:
                    continue
                speaker = message.split(": ")[0]
                content = message.split(": ")[1]
                chunk_text += f"<{speaker}> {content}" + "\n"
            chunk_text += '\n\n'
        chunk_text = chunk_text.strip()
        return chunk_text

            # conv_tokens = count_tokens(conversation['event_content'])
            # for dialogue in conversation['dialogues']:
            #     for message in dialogue['messages']:
            #         conv_tokens += count_tokens(message)

            # # Add conversation to current chunk
            # current_chunk.append(conversation)
            # current_tokens += conv_tokens

            # # Check if we should finalize this chunk
            # meets_min_tokens = current_tokens >= self.min_chunk_tokens

            # # Check if adding the next conversation would exceed target tokens
            # next_conv_tokens = conversations[i + 1]['token_count'] if i + 1 < len(conversations) else 0
            # would_exceed_tokens = current_tokens + next_conv_tokens > self.target_chunk_size

            # num_conversation_used += 1

            # Finalize chunk only if:
            # 1. We meet BOTH minimum requirements (conversations AND tokens)
            # 2. AND (adding next conversation would exceed target OR we've hit max conversations)
        #     if (meets_min_tokens and (would_exceed_tokens)):

        #         chunks.append(current_chunk)
        #         current_chunk = []
        #         current_tokens = 0

        #         if len(chunks) >= self.max_num_chunks:
        #             break

        # # Add the last chunk if it has any conversations
        # if current_chunk:
        #     # Check if the last chunk exceeds target size and split if necessary
        #     last_chunk_tokens = sum(conv['token_count'] for conv in current_chunk)
        #     if last_chunk_tokens > self.target_chunk_size and len(current_chunk) > 1:
        #         # Split the last chunk into two parts
        #         split_chunks = self._split_chunk(current_chunk)
        #         chunks.extend(split_chunks)
        #     else:
        #         chunks.append(current_chunk)

        # return chunks, num_conversation_used

    def _split_chunk(self, chunk: List[Dict]) -> List[List[Dict]]:
        """Split a chunk that exceeds target size into two smaller chunks."""
        if len(chunk) <= 1:
            return [chunk]

        # Find the best split point - aim for roughly half the tokens in each part
        total_tokens = sum(conv['token_count'] for conv in chunk)
        target_first_half = total_tokens // 2

        current_tokens = 0
        split_point = len(chunk) // 2  # fallback split point

        # Find the split point closest to half the tokens
        for i, conv in enumerate(chunk):
            current_tokens += conv['token_count']
            if current_tokens >= target_first_half:
                split_point = i + 1
                break

        # Ensure we don't create empty chunks
        split_point = max(1, min(split_point, len(chunk) - 1))

        first_chunk = chunk[:split_point]
        second_chunk = chunk[split_point:]

        # Verify both chunks meet minimum requirements if possible
        first_tokens = sum(conv['token_count'] for conv in first_chunk)
        second_tokens = sum(conv['token_count'] for conv in second_chunk)

        # If either chunk is too small, adjust split point
        if first_tokens < self.min_chunk_tokens and len(chunk) > 2:
            # Move one more conversation to first chunk
            if split_point < len(chunk) - 1:
                split_point += 1
                first_chunk = chunk[:split_point]
                second_chunk = chunk[split_point:]
        elif second_tokens < self.min_chunk_tokens and len(chunk) > 2:
            # Move one conversation back to first chunk
            if split_point > 1:
                split_point -= 1
                first_chunk = chunk[:split_point]
                second_chunk = chunk[split_point:]

        return [first_chunk, second_chunk]

    def create_multiple_chunk_lists(self, conversations: List[Dict], num_lists: int = 10) -> List[List[List[Dict]]]:
        """Create multiple chunk lists starting from different timesteps."""

        # chunk_lists = []
        # num_conversation_used = 0
        # while num_conversation_used < len(conversations):
        #     chunks, num_conversation_used = self.create_chunks(conversations[num_conversation_used:])
        #     chunk_lists.append(chunks)
        #     conversations = conversations[num_conversation_used:]

        chunk_lists = []
        for conversation in conversations:
            chunks = self.create_chunks(conversation)
            chunk_lists.append(chunks)

        return chunk_lists

    def extract_qa_for_character(self, character_name: str, perltqa_data: List[Dict]) -> List[Dict]:
        """Extract Q&A pairs for a specific character."""
        qa_pairs = []

        for person_data in perltqa_data:

            if character_name in person_data:

                character_qa = person_data[character_name]

                # Process each memory type (profile, social_relationship, events, dialogues)
                for memory_type, qa_list in character_qa.items():
                    if memory_type == 'profile':
                        continue
                    if isinstance(qa_list, list):
                        for qa_item in qa_list:
                            # Handle nested structure like "4_0_0#0": [qa_items...]
                            if isinstance(qa_item, dict):
                                for key, value in qa_item.items():
                                    if isinstance(value, list):
                                        # This is a nested structure like "4_0_0#0": [qa_items...]
                                        for nested_qa in value:
                                            if isinstance(nested_qa, dict) and 'Question' in nested_qa and 'Answer' in nested_qa:
                                                qa_pairs.append({
                                                    'question': nested_qa['Question'],
                                                    'answer': nested_qa['Answer'],
                                                    'reference_memory': nested_qa.get('Reference Memory', ''),
                                                    'memory_type': memory_type
                                                })
                                    elif 'Question' in qa_item and 'Answer' in qa_item:
                                        # Direct Q&A item
                                        qa_pairs.append({
                                            'question': qa_item['Question'],
                                            'answer': qa_item['Answer'],
                                            'reference_memory': qa_item.get('Reference Memory', ''),
                                            'memory_type': memory_type
                                        })
                                        break  # Don't process this item again

        return qa_pairs

    def get_character_name_from_memory(self, perltmem_item: Dict) -> str:
        """Extract character name from perltmem data."""
        profile = perltmem_item.get('profile', {})
        return profile.get('Protagonist', 'Unknown')

    def get_reference_memory_ids(self, reference_memory: str) -> List[str]:
        """Extract memory IDs from reference memory string."""
        try:
            # Handle both string and list formats
            if reference_memory.startswith('[') and reference_memory.endswith(']'):
                # Parse as list string, e.g., "['4_0_0']"
                import ast
                memory_list = ast.literal_eval(reference_memory)
                return memory_list if isinstance(memory_list, list) else [memory_list]
            else:
                # Single memory reference
                return [reference_memory]
        except:
            return []

    def filter_qa_pairs_for_chunk(self, qa_pairs: List[Dict], chunk_conversations: List[Dict]) -> List[Dict]:
        """Filter Q&A pairs to only include those covered by conversations in the chunk."""
        # Get all event and dialogue IDs covered by this chunk
        covered_event_ids = set()
        covered_dialogue_ids = set()

        for chunk in chunk_conversations:
            for conv in chunk:
                # Add event ID from the conversation
                event_id = conv['event_id']
                if event_id:
                    # Handle both string and list event IDs
                    if isinstance(event_id, list):
                        covered_event_ids.update(event_id)
                    else:
                        covered_event_ids.add(event_id)

                # Add dialogue ID (both full and base)
                dialogue_id = conv['dialogue_id']
                if dialogue_id:
                    covered_dialogue_ids.add(dialogue_id)
                    # Also add base dialogue ID (without #X suffix)
                    if '#' in dialogue_id:
                        base_dialogue_id = dialogue_id.split('#')[0]
                        covered_dialogue_ids.add(base_dialogue_id)

        # Combine all covered memory IDs
        all_covered_ids = covered_event_ids.union(covered_dialogue_ids)

        # Filter Q&A pairs based on reference memory coverage
        filtered_qa_pairs = []
        for qa in qa_pairs:
            reference_memory = qa.get('reference_memory', '')
            memory_type = qa.get('memory_type', '')
            memory_ids = self.get_reference_memory_ids(reference_memory)

            if memory_type in ['events', 'dialogues']:
                if any(mem_id in all_covered_ids for mem_id in memory_ids):
                    filtered_qa_pairs.append(qa)

        return filtered_qa_pairs

    def format_dialogue_text(self, text: str, character_name: str) -> str:
        """Format dialogue text with proper user and assistant tags."""
        import re

        # Common patterns for AI Assistant names
        assistant_patterns = ['AI Assistant', 'AI assistant', 'Assistant']

        # Create regex pattern to split on speaker names
        speaker_pattern = r'(?:^|\s)(' + '|'.join(assistant_patterns + [character_name]) + r'):\s*'

        # Split the text while keeping the delimiters
        parts = re.split(speaker_pattern, text)

        formatted_lines = []
        i = 0

        while i < len(parts):
            part = parts[i].strip()

            # Check if this is a speaker name
            if part in assistant_patterns:
                # Get the text following the assistant name
                if i + 1 < len(parts):
                    assistant_text = parts[i + 1].strip()
                    formatted_lines.append(f"<Assistant> {assistant_text}")
                    i += 2
                else:
                    i += 1
            elif part == character_name:
                # Get the text following the user name
                if i + 1 < len(parts):
                    user_text = parts[i + 1].strip()
                    formatted_lines.append(f"<User {character_name}> {user_text}")
                    i += 2
                else:
                    i += 1
            else:
                # This is text without a clear speaker (shouldn't happen often)
                if part:
                    formatted_lines.append(part)
                i += 1

        # Join with newlines for proper formatting
        return '\n'.join(formatted_lines)

    def format_conversational_chunk(self, chunk_content: str, chunk_index: int = 0) -> str:
        """Format chunk content with conversational templates and random selection."""
        # Generate timestamp for dialogue header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Select templates randomly
        user_template = random.choice(USER_MESSAGE_TEMPLATES)
        assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

        # Format the conversational dialogue
        formatted_text = f"[Dialogue between User and Assistant on {timestamp}]\n"
        formatted_text += f"<User>{user_template}\n{chunk_content}\n"
        formatted_text += f"<Assistant>{assistant_template}"

        return formatted_text

    def process_single_character(self, perltmem_item: Dict, perltqa_data: List[Dict]) -> List[Dict]:
        """Process a single character's data to generate multiple chunk groups from different starting points."""
        character_name = self.get_character_name_from_memory(perltmem_item)

        # Extract conversations from dialogues with events context
        dialogues = perltmem_item.get('dialogues', {})
        events = perltmem_item.get('events', {})

        conversations = self.extract_conversations_from_dialogues(dialogues, events, character_name)

        if not conversations:
            return []

        # Create multiple chunk lists starting from different timesteps
        chunks = self.create_multiple_chunk_lists(conversations)

        # Extract all Q&A pairs for this character
        all_qa_pairs = self.extract_qa_for_character(character_name, perltqa_data)

        return chunks, all_qa_pairs


    def process_all_data(self) -> List[Dict]:
        """Process all characters' data."""
        perltmem_data, perltqa_data = self.load_data()

        all_instances = []

        for perltmem_item in perltmem_data:
            chunks, qa_pairs = self.process_single_character(perltmem_item, perltqa_data)

            all_instances.append({
                'chunks': chunks,
                'qa_pairs': qa_pairs
            })

        return all_instances

def parse_answers_cache(filepath="answers.txt"):
    """Parse the answers.txt file to create a mapping of question endings to answerability."""
    question_cache = {}

    if not os.path.exists(filepath):
        return question_cache

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Check if it's answerable (✓) or not (✗)
            is_answerable = line.strip().startswith('✓')

            # Extract question ending after "..."
            if '...' in line:
                question_ending = line.split('...', 1)[1].strip()
                # Handle HTML entities
                question_ending = question_ending.replace('&#39;', "'")

                if question_ending in question_cache:
                    if is_answerable != question_cache[question_ending]:
                        del question_cache[question_ending]

                else:
                    question_cache[question_ending] = is_answerable

    return question_cache

def process_perltqa_batch(batch_data):
    """Worker function to process a batch of PerLTQA instances with GPT-4.1"""
    instances_batch, api_key, batch_id, question_cache = batch_data

    # Create Azure OpenAI client for this worker
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2025-01-01-preview",
        azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
    )

    results = []

    print(f"🚀 Batch {batch_id}: Starting processing of {len(instances_batch)} instances")
    sys.stdout.flush()

    for instance_idx, instance in enumerate(instances_batch):
        character_name = instance.get('metadata', {}).get('character', 'Unknown')
        print(f"📝 Batch {batch_id}: Processing instance {instance_idx+1}/{len(instances_batch)}: Character {character_name}")
        sys.stdout.flush()

        # Combine all chunks into context
        context = "\n\n".join(instance['chunks'])

        # Track answerable questions for this instance
        answerable_questions = []

        for qa in instance['questions_and_answers']:
            question = qa['question']
            expected_answer = qa['answer']

            # Check if we have a cached result
            cached_result = None
            for ending, is_answerable in question_cache.items():
                if question.endswith(ending) or ending in question[-100:]:
                    cached_result = is_answerable
                    break

            if cached_result is not None:
                # Use cached result
                if cached_result:
                    answerable_questions.append(qa)
                    print(f"  ✓ [CACHED] Question answerable: ...{question[-50:]}")
                else:
                    print(f"  ✗ [CACHED] Question not answerable: ...{question[-50:]}")
            else:
                # Need to query GPT-4.1
                check_prompt = f"""Given the following context, please answer this question. Only output the answer without saying anything else.

Context:
{context}

Question: {question}

Answer:"""

                try:
                    response = client.chat.completions.create(
                        model="gpt-4.1",
                        messages=[
                            {
                                "role": "user",
                                "content": check_prompt,
                            }
                        ],
                        temperature=0.0,
                        max_tokens=100
                    )

                    gpt_answer = response.choices[0].message.content.strip()

                    # Check if GPT-4.1 can answer correctly
                    hit = False
                    if ";" not in expected_answer:
                        if expected_answer.lower() in gpt_answer.lower():
                            hit = True
                    else:
                        new_answers = []
                        for answer in expected_answer.split(";"):
                            answer = answer.strip()
                            if answer.lower() in gpt_answer.lower():
                                new_answers.append(answer)

                        if len(new_answers) > 0:
                            hit = True
                            qa['answer'] = ";".join(new_answers)

                    if hit:
                        answerable_questions.append(qa)
                        print(f"  ✓ Question answerable: ...{question[-50:]}")
                    else:
                        print(f"  ✗ Question answered wrong: ...{question[-50:]}")

                except Exception as e:
                    print(f"  Error checking question: {e}")
                    continue

        # Store result for this instance
        result = {
            'instance': instance,
            'answerable_questions': answerable_questions,
            'num_answerable': len(answerable_questions)
        }
        results.append(result)

        print(f"  Instance summary: {len(answerable_questions)} answerable questions")
        sys.stdout.flush()

    return results

def process_perltqa_dataset():
    """Process PerLTQA dataset into the desired format"""

    if not os.path.exists("./data/perltqa/raw_instances.json"):
        perltmem_path = "./data/perltmem_en.json"
        perltqa_path = "./data/perltqa_en_shortened.json"  # Use the shortened version!

        # Check if input files exist
        if not os.path.exists(perltmem_path):
            print(f"Error: PerLTMem file not found: {perltmem_path}")
            return []

        if not os.path.exists(perltqa_path):
            print(f"Error: PerLTQA file not found: {perltqa_path}")
            return []

        processor = PerLTQAProcessor(perltmem_path, perltqa_path)

        print("Processing PerLTQA dataset...")
        all_instances = processor.process_all_data()

        # Convert to the standard format
        processed_data = []

        for instance in all_instances:
            # Extract metadata from the group

            formatted_group = {
                'prompt': 'I will provide you with sequential information chunks. Please analyze each chunk and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.',
                'chunks': instance['chunks'],
                'questions_and_answers': instance['qa_pairs'],
                'data_source': 'perltqa',
            }
            processed_data.append(formatted_group)

        # Print statistics
        total_chunk_groups = len(processed_data)
        total_qa_pairs = sum(len(group['questions_and_answers']) for group in processed_data)

        # Calculate token statistics
        token_counts = [count_tokens("\n".join(item['chunks'])) for item in processed_data]
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        avg_chunks_per_list = sum(len(group['chunks']) for group in processed_data) / total_chunk_groups if total_chunk_groups else 0

        print(f"Processing complete!")
        print(f"Total chunk groups: {total_chunk_groups}")
        print(f"Average chunks per list: {avg_chunks_per_list:.1f}")
        print(f"Total Q&A pairs: {total_qa_pairs}")
        print(f"Token statistics - Min: {min_tokens}, Max: {max_tokens}, Avg: {avg_tokens:.1f}")

        # Save raw instances
        os.makedirs("./data/perltqa", exist_ok=True)
        with open("./data/perltqa/raw_instances.json", "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    else:
        with open("./data/perltqa/raw_instances.json", "r", encoding="utf-8") as f:
            processed_data = json.load(f)

    # Filter out instances with too many chunks (over 20)
    max_chunks_allowed = 30
    filtered_instances = []
    instances_with_too_many_chunks = 0

    for instance in processed_data:
        num_chunks = len(instance['chunks'])
        if num_chunks <= max_chunks_allowed:
            filtered_instances.append(instance)
        else:
            instances_with_too_many_chunks += 1
            print(f"Removing PerLTQA instance with {num_chunks} chunks (> {max_chunks_allowed})")

    print(f"\n📊 PerLTQA Chunk Filtering Summary:")
    print(f"  • Removed {instances_with_too_many_chunks} instances with > {max_chunks_allowed} chunks")
    if processed_data:
        print(f"  • Instances: {len(processed_data):,} → {len(filtered_instances):,} (kept {len(filtered_instances)/len(processed_data)*100:.1f}%)")
    else:
        print(f"  • Instances: 0 → 0")

    # Filter out instances which GPT-4.1 fails to answer:
    print("Filtering instances based on GPT-4.1 answerability...")

    # Load cached results from answers.txt
    question_cache = parse_answers_cache("answers.txt")
    print(f"Loaded {len(question_cache)} cached question results from answers.txt")

    # Prepare for multi-processing
    num_processes = min(16, mp.cpu_count())
    print(f"Using {num_processes} parallel processes for GPT-4.1 queries")

    # Get API key
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("Warning: AZURE_OPENAI_API_KEY not found, falling back to single-threaded processing")
        num_processes = 1

    # Split instances into batches
    batch_size = max(1, len(filtered_instances) // num_processes)
    instance_batches = []

    for i in range(0, len(filtered_instances), batch_size):
        batch = filtered_instances[i:i + batch_size]
        batch_id = len(instance_batches)
        instance_batches.append((batch, api_key, batch_id, question_cache))

    print(f"Split {len(filtered_instances)} instances into {len(instance_batches)} batches")
    print("-" * 60)

    # Process batches in parallel
    with mp.Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_perltqa_batch, instance_batches),
            total=len(instance_batches),
            desc="🔄 Processing batches",
            unit="batch"
        ))

    # Combine results from all batches
    answerable_instances = []
    total_kept = 0
    total_skipped = 0

    for batch_result in batch_results:
        for result in batch_result:
            instance = result['instance']
            answerable_questions = result['answerable_questions']
            num_answerable = result['num_answerable']

            # Only keep instances where at least 20 questions are answerable
            if num_answerable >= 20:
                instance['questions_and_answers'] = answerable_questions
                answerable_instances.append(instance)
                total_kept += 1
            else:
                total_skipped += 1

    print("-" * 60)
    print("🎉 MULTIPROCESSING COMPLETE!")
    print(f"📊 Filtering Summary:")
    print(f"  • Instances kept: {total_kept} (with ≥20 answerable questions)")
    print(f"  • Instances skipped: {total_skipped}")
    print(f"  • Total instances after filtering: {len(answerable_instances)}")

    filtered_instances = answerable_instances

    print(f"Final processed dataset: {len(filtered_instances)} chunk groups")
    return filtered_instances

def process_ttl_train_dataset():
    """Process TTL train dataset (TREC-C and NLU from MemoryAgentBench)"""
    print("Processing TTL train dataset from MemoryAgentBench...")

    # Load the MemoryAgentBench dataset
    dataset = load_dataset("ai-hyz/MemoryAgentBench")

    processed_data = []
    target_total_tokens = 2000  # Total tokens across all chunks
    chunks_per_instance = 10
    instances_per_dataset = 200

    # Set random seed for reproducibility
    random.seed(42)

    def parse_context_examples(context):
        """Parse context into training examples with questions and labels."""
        examples = []
        lines = context.strip().split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line and not line.startswith('label:'):
                # This should be a question
                question = line
                # Next line should be the label
                if i + 1 < len(lines) and lines[i + 1].strip().startswith('label:'):
                    label = lines[i + 1].strip().replace('label:', '').strip()
                    examples.append({
                        'question': question,
                        'label': label,
                        'tokens': count_tokens(question)
                    })
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        return examples

    def get_all_labels(examples):
        """Get all unique labels from examples."""
        labels = set()
        for example in examples:
            labels.add(example['label'])
        return sorted(list(labels))

    def create_chunks_from_examples(examples, num_chunks):
        """Create chunks from examples with total ~2k tokens across all chunks."""
        if not examples:
            return []

        # Calculate how many examples we need total to reach ~2k tokens
        avg_tokens_per_example = sum(ex['tokens'] for ex in examples) / len(examples)
        total_examples_needed = max(num_chunks, int(target_total_tokens / avg_tokens_per_example))

        # Select examples (cycling if needed)
        selected_examples = []
        for i in range(total_examples_needed):
            selected_examples.append(examples[i % len(examples)])

        # Shuffle selected examples for randomness
        random.shuffle(selected_examples)

        # Distribute selected examples evenly across chunks
        examples_per_chunk = max(1, total_examples_needed // num_chunks)
        remainder = total_examples_needed % num_chunks

        chunks = []
        start_idx = 0

        for i in range(num_chunks):
            # Some chunks get one extra example if there's a remainder
            chunk_size = examples_per_chunk + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size

            chunk_examples = selected_examples[start_idx:end_idx]

            # Convert to text format - format each example like pubmed-rct
            chunk_lines = []
            for ex in chunk_examples:
                # Format each example like pubmed-rct with the question wrapped
                formatted_example = f"Sentence: {ex['question']}\nLabel: {ex['label']}"
                chunk_lines.append(formatted_example)

            chunks.append('\n\n'.join(chunk_lines))  # Use double newline between examples
            start_idx = end_idx

        return chunks

    def ensure_all_labels_in_chunks(chunks, all_labels, examples):
        """Ensure all labels appear in the chunks by adding examples if needed."""
        # Parse existing chunks to see which labels are present
        present_labels = set()
        for chunk in chunks:
            chunk_examples = parse_context_examples(chunk)
            for ex in chunk_examples:
                present_labels.add(ex['label'])

        missing_labels = set(all_labels) - present_labels

        if not missing_labels:
            return chunks

        # Find examples for missing labels
        label_to_examples = {}
        for example in examples:
            label = example['label']
            if label not in label_to_examples:
                label_to_examples[label] = []
            label_to_examples[label].append(example)

        # Add missing examples to the first chunk
        if chunks:
            first_chunk = chunks[0]
            missing_examples = []

            for label in missing_labels:
                if label in label_to_examples:
                    # Randomly select one example for each missing label
                    available_examples = label_to_examples[label]
                    selected_example = random.choice(available_examples)
                    missing_examples.append(selected_example)

            if missing_examples:
                additional_lines = []
                for ex in missing_examples:
                    # Format like pubmed-rct
                    formatted_example = f"Sentence: {ex['question']}\nLabel: {ex['label']}"
                    additional_lines.append(formatted_example)

                chunks[0] = first_chunk + '\n\n' + '\n\n'.join(additional_lines)

        return chunks

    def process_single_dataset(dataset_name, original_source, context, test_questions, test_answers):
        """Process a single dataset (TREC-C or NLU)."""
        print(f"  Processing {dataset_name} dataset (source: {original_source})...")

        # Parse context into training examples
        examples = parse_context_examples(context)
        all_labels = get_all_labels(examples)

        print(f"    Found {len(examples)} training examples")
        print(f"    Labels: {all_labels} ({len(all_labels)} total)")
        print(f"    Test questions: {len(test_questions)}")

        # Create instances
        instances = []
        base_date = datetime(2024, 1, 1)

        for i in range(instances_per_dataset):
            # Shuffle examples for variety
            shuffled_examples = examples.copy()
            random.shuffle(shuffled_examples)

            # Create chunks
            raw_chunks = create_chunks_from_examples(shuffled_examples, chunks_per_instance)

            # Ensure all labels are present
            raw_chunks = ensure_all_labels_in_chunks(raw_chunks, all_labels, examples)

            # Format chunks with conversational templates
            formatted_chunks = []
            for chunk_idx, chunk_content in enumerate(raw_chunks):
                # Use classification-specific templates
                user_template = random.choice(CLASSIFICATION_USER_TEMPLATES)
                assistant_template = random.choice(ASSISTANT_RESPONSE_TEMPLATES)

                # Create a timestamp for each chunk (incrementing by days)
                chunk_date = base_date + timedelta(days=chunk_idx)
                timestamp = chunk_date.strftime("%Y-%m-%d %H:%M")

                # Format the chunk with conversational template
                formatted_chunk = f"[Dialogue between User and Assistant on {timestamp}]\n<User>{user_template}\n{chunk_content}\n<Assistant>{assistant_template}"
                formatted_chunks.append(formatted_chunk)

            # Create Q&A pairs from test data with pubmed-rct style formatting
            qa_pairs = []
            for q, a in zip(test_questions, test_answers):
                # Format question like pubmed-rct
                formatted_question = f"Sentence: {q}\nWhat are the labels for the above sentence? Put your final answer as \\boxed{{label}}."
                qa_pairs.append({
                    'question': formatted_question,
                    'answer': a,
                    'evidence_idx': 0  # For classification tasks, evidence is across all chunks
                })

            # Create the data instance
            data_instance = {
                'prompt': f'I will provide you with classification training examples for {dataset_name}. Please analyze each chunk and store the classification patterns and labels to answer test questions correctly.',
                'chunks': formatted_chunks,
                'questions_and_answers': qa_pairs,
                'data_source': original_source,
            }

            instances.append(data_instance)

            if (i + 1) % 50 == 0:
                print(f"    Created {i + 1}/{instances_per_dataset} instances")

        return instances

    # Process both TREC-C and NLU datasets
    all_processed_data = []

    # Find and process each dataset
    for split_name, split_data in dataset.items():
        for example in tqdm(split_data, desc=f"Scanning {split_name}"):
            source = example['metadata']['source']

            if 'trec_coarse' in source:
                trec_instances = process_single_dataset(
                    'TREC-C',
                    source,  # Pass original source
                    example['context'],
                    example['questions'],
                    [ans[0] for ans in example['answers']]
                )

                all_processed_data.extend(trec_instances)
                break  # Only process the first TREC-C example

    # Process NLU dataset
    for split_name, split_data in dataset.items():
        for example in tqdm(split_data, desc=f"Scanning {split_name} for NLU"):
            source = example['metadata']['source']

            if 'nlu' in source:
                nlu_instances = process_single_dataset(
                    'NLU',
                    source,  # Pass original source
                    example['context'],
                    example['questions'],
                    [ans[0] for ans in example['answers']]
                )
                all_processed_data.extend(nlu_instances)
                break  # Only process the first NLU example

    print(f"\nTotal instances created: {len(all_processed_data)}")
    print(f"  TREC-C instances: {sum(1 for inst in all_processed_data if 'trec' in inst['data_source'])}")
    print(f"  NLU instances: {sum(1 for inst in all_processed_data if 'nlu' in inst['data_source'])}")

    return all_processed_data

def main():
    parser = argparse.ArgumentParser(description='Process datasets for memory training')
    parser.add_argument('--dataset', type=str, choices=['squad', 'squad_test', 'hotpotqa', 'booksum', 'friends', 'wos46985', 'pubmed-rct', 'arxiv-classification', 'eurlex', 'accurate_retrieval', 'long_range_understanding', 'conflict_resolution', 'test_time_learning', 'detectiveqa', 'lamp4', 'perltqa', 'narrativeqa', 'ttl_train', 'cr_train', 'lme_train'],
                       default='squad', help='Dataset to process (default: squad)')
    parser.add_argument('--convert-to-parquet', action='store_true',
                       help='Convert existing JSON files to parquet format')
    parser.add_argument('--split-train-test', action='store_true',
                       help='Combine all parquet files, shuffle, and split into train/test sets (80/20)')
    parser.add_argument('--split-single-dataset', action='store_true',
                       help='Split a single dataset into train/test sets (works with --dataset)')
    parser.add_argument('--filter-dataset', action='store_true',
                       help='Filter out questions that GPT-4o-mini can answer without context (squad/hotpotqa only)')
    parser.add_argument('--max-questions', type=int, default=None,
                       help='Maximum number of questions to test for filtering (for debugging)')
    parser.add_argument('--num-processes', type=int, default=16,
                       help='Number of parallel processes for filtering (default: 16)')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio of data to use for training when splitting (default: 0.9)')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild the dataset even if it already exists')
    parser.add_argument('--split-by-difficulty', action='store_true',
                       help='Split training dataset by difficulty based on accuracy bins from analyze_results.py')
    parser.add_argument('--merge-datasets', type=str, nargs='+',
                       help='Create memalpha train/test files from specified datasets. Accepts multiple dataset names (e.g., booksum pubmed-rct)')
    parser.add_argument('--status', type=str, default='all',
                       help='Status of the dataset to process (default: all)')
    parser.add_argument('--output-name', type=str, default='memalpha',
                       help='Name of the output dataset (default: memalpha)')
    parser.add_argument('--limit-size', type=int, default=None,
                       help='Maximum number of examples per training dataset (default: None)')

    args = parser.parse_args()


    # If splitting by difficulty
    if args.split_by_difficulty:
        print("Splitting training dataset by difficulty based on accuracy bins...")
        output_files = split_dataset_by_difficulty()
        if output_files:
            print(f"Successfully split dataset into {len(output_files)} difficulty levels!")
        else:
            print("Failed to split dataset by difficulty.")
        return

    # If creating memalpha from specified datasets
    if args.merge_datasets:
        print(f"Creating memalpha from datasets {args.merge_datasets}...")
        train_path, test_path = merge_into_memalpha(args.merge_datasets, random_seed=42, status=args.status, output_name=args.output_name, limit_size=args.limit_size)
        if train_path and test_path:
            print(f"\nSuccessfully created {args.output_name} from specified datasets!")
        else:
            print("Failed to create memalpha from specified datasets.")
        return

    # If filtering dataset
    if args.filter_dataset:
        if args.dataset not in ['squad', 'hotpotqa']:
            print(f"Error: Filtering only supported for 'squad' and 'hotpotqa', got '{args.dataset}'")
            return

        print(f"Filtering {args.dataset} dataset with {args.num_processes} processes...")
        output_file = filter_dataset(args.dataset, max_questions_to_test=args.max_questions, num_processes=args.num_processes)
        if output_file:
            print(f"Successfully filtered {args.dataset} dataset!")
        else:
            print(f"Failed to filter {args.dataset} dataset.")
        return

    # If combining and splitting datasets
    if args.split_train_test:
        print("Combining and splitting datasets into train/test sets...")
        train_path, test_path = combine_and_split_datasets(train_ratio=args.train_ratio)
        if train_path and test_path:
            print(f"\nSuccessfully created train/test split!")
        else:
            print("Failed to create train/test split.")
        return

    # If splitting a single dataset
    if args.split_single_dataset:
        print(f"Splitting {args.dataset} dataset into train/test sets...")
        if args.dataset == 'perltqa':
            train_path, test_path = split_dataset(args.dataset, train_ratio=0.9)
        elif args.dataset == 'lme_train':
            train_path, test_path = split_dataset(args.dataset, train_ratio=1.0)
        else:
            train_path, test_path = split_dataset(args.dataset, train_ratio=args.train_ratio)
        if train_path and test_path:
            print(f"\nSuccessfully created train/test split for {args.dataset}!")
        else:
            print(f"Failed to create train/test split for {args.dataset}.")
        return

    # Process the dataset based on the argument
    if args.dataset == 'squad':
        filename = './data/processed_squad_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_squad_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'hotpotqa':
        filename = './data/processed_hotpotqa_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_hotpotqa_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'booksum':
        filename = './data/processed_booksum_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_booksum_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'pubmed-rct':
        filename = './data/processed_pubmed_rct_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_pubmed_rct_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'arxiv-classification':
        filename = './data/processed_arxiv_classification_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_arxiv_classification_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'accurate_retrieval':
        filename = './data/processed_accurate_retrieval_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_memory_agent_bench("Accurate_Retrieval")
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'long_range_understanding':
        filename = './data/processed_long_range_understanding_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_memory_agent_bench("Long_Range_Understanding")
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'test_time_learning':
        filename = './data/processed_test_time_learning_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_memory_agent_bench("Test_Time_Learning")
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'perltqa':
        filename = './data/processed_perltqa_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_perltqa_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'ttl_train':
        filename = './data/processed_ttl_train_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_ttl_train_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    elif args.dataset == 'lme_train':
        filename = './data/processed_lme_train_data.json'
        if not os.path.exists(filename) or args.force:
            processed_data = process_lme_train_dataset()
            save_processed_data(processed_data, filename=filename)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)

    else:
        print(f"Unknown dataset: {args.dataset}")
        return

    # Save to file (only if we have data)
    if processed_data:

        # Print statistics
        print_statistics(processed_data, args.dataset)

        # Automatically convert to parquet
        print("\nConverting to parquet format...")
        try:
            parquet_file = convert_json_to_parquet(filename, args.dataset)
            print(f"Successfully created parquet file: {parquet_file}")
        except Exception as e:
            print(f"Error converting to parquet: {e}")
    else:
        print(f"No data processed for {args.dataset} dataset.")

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Start method already set
    main()
