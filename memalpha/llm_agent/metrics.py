"""
Evaluation utilities for memory agent benchmarks.
Adopted from https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/utils/eval_utils.py
"""

import os 
import string
import re
import json
from collections import Counter
import numpy as np
import nltk
import tiktoken
from rouge_score import rouge_scorer
from editdistance import eval as edit_distance

import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ============================================================================
# TEXT NORMALIZATION AND SCORING UTILITIES
# ============================================================================

def normalize_answer(answer_text):
    """
    Normalize text for evaluation by removing articles, punctuation, and extra whitespace.
    
    Args:
        answer_text: The text to normalize
        
    Returns:
        Normalized text string
    """
    # Apply all normalization steps in sequence
    text = answer_text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    
    return text


def f1_score(prediction, ground_truth):
    """
    Calculate F1 score between prediction and ground truth.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Tuple of (f1_score, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # Handle special cases for yes/no/noanswer responses
    special_answers = {'yes', 'no', 'noanswer'}
    if ((normalized_prediction in special_answers or normalized_ground_truth in special_answers) and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    # Tokenize both texts and calculate token overlap
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_common_tokens = sum(common_tokens.values())
    
    if num_common_tokens == 0:
        return ZERO_METRIC
    
    # Calculate precision, recall, and F1
    precision = num_common_tokens / len(prediction_tokens)
    recall = num_common_tokens / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall


def drqa_exact_match_score(prediction, ground_truth):
    """
    Check if prediction is an exact match with ground truth after normalization.
    
    Args:
        prediction: The predicted text
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating exact match
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction, ground_truth):
    """
    Check if ground truth is a substring of the prediction after normalization.
    
    Args:
        prediction: The predicted text  
        ground_truth: The ground truth text
        
    Returns:
        Boolean indicating substring match
    """
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def drqa_metric_max_over_ground_truths(metric_function, prediction, ground_truths):
    """
    Calculate the maximum score over multiple ground truth answers.
    
    Args:
        metric_function: Function to calculate score between prediction and single ground truth
        prediction: The predicted text
        ground_truths: List of ground truth answers (can be string, list, or nested list)
        
    Returns:
        Maximum score across all ground truths
    """
    # Normalize ground_truths to a flat list of strings
    if isinstance(ground_truths, str):
        ground_truth_list = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        # Flatten nested lists
        ground_truth_list = [gt for gt_sublist in ground_truths for gt in gt_sublist]
    else:
        ground_truth_list = ground_truths

    # Calculate score for each ground truth and return maximum
    return max(metric_function(prediction, gt) for gt in ground_truth_list)


def parse_output(output_text, answer_prefix="Answer:"):
    """
    Parse model output to extract the answer portion.
    
    Args:
        output_text: The complete model output
        answer_prefix: The prefix that indicates where the answer starts
        
    Returns:
        Extracted answer text or None if not found
    """
    # Try multiple patterns to extract the answer
    extraction_patterns = [
        re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", flags=re.IGNORECASE), 
        re.compile(r"(?:^)(.*)(?:\n|$)")
    ]
    
    for pattern in extraction_patterns:
        match = pattern.search(output_text)
        if match:
            extracted_text = match[1].strip()
            # Remove prefix again in case it was repeated
            clean_answer = re.sub(f'^{re.escape(answer_prefix)}', '', extracted_text, flags=re.IGNORECASE).strip()
            return clean_answer
    
    # Should rarely reach here, but return None if no pattern matches
    return None


# ============================================================================
# TEXT CHUNKING UTILITIES
# ============================================================================

def chunk_text_into_sentences(text, model_name="gpt-4o-mini", chunk_size=4096):
    """
    Split text into chunks of specified token size, preserving sentence boundaries.
    
    Args:
        text: The long text document to be split
        model_name: The tokenizer model name (default: gpt-4o-mini)
        chunk_size: Maximum number of tokens allowed per chunk
        
    Returns:
        List of text chunks, each within the specified token limit
    """
    # Ensure NLTK sentence tokenizer is available
    nltk.download('punkt', quiet=True)
    
    # Initialize tokenizer with fallback
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Use fallback model if specified model is not recognized
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)
    
    text_chunks = []
    current_chunk_sentences = []
    current_chunk_token_count = 0

    for sentence in sentences:
        # Count tokens in current sentence
        sentence_tokens = encoding.encode(sentence, allowed_special={'<|endoftext|>'})
        sentence_token_count = len(sentence_tokens)
        
        # Check if adding this sentence would exceed chunk size
        if current_chunk_token_count + sentence_token_count > chunk_size:
            # Finalize current chunk and start new one
            text_chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_chunk_token_count += sentence_token_count
    
    # Add final chunk if it contains any sentences
    if current_chunk_sentences:
        text_chunks.append(" ".join(current_chunk_sentences))
    
    return text_chunks


def count_tokens(text, model_name="gpt-3.5-turbo"):
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for tokenizer
        
    Returns:
        Number of tokens in the text
    """
    encoding = tiktoken.encoding_for_model(model_name)
    
    # Convert input to string if it's not already a string
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        else:
            text = str(text)
    
    return len(encoding.encode(text))


def create_chunks_use_sent_tokenizer(text, max_tokens=10000):
    """
    Create text chunks using sentence tokenization with token limits.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    # Ensure NLTK punkt tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk_text = ""
    current_token_count = 0
    
    for sentence in sentences:
        sentence_token_count = count_tokens(sentence)
        
        # Start new chunk if adding sentence would exceed limit
        if current_token_count + sentence_token_count > max_tokens and current_chunk_text:
            chunks.append(current_chunk_text.strip())
            current_chunk_text = sentence
            current_token_count = sentence_token_count
        else:
            # Add sentence to current chunk
            if current_chunk_text:
                current_chunk_text += " " + sentence
                current_token_count += sentence_token_count + count_tokens(" ")
            else:
                current_chunk_text = sentence
                current_token_count = sentence_token_count
    
    # Add final chunk
    if current_chunk_text:
        chunks.append(current_chunk_text.strip())
    
    return chunks


# ============================================================================
# RECOMMENDATION SYSTEM UTILITIES
# ============================================================================

def clean_text_elements(text, remove_parentheses=True, normalize_ws=True, remove_nums=True):
    """Clean text by removing various elements."""
    if remove_parentheses:
        text = re.sub(r"\([^()]*\)", "", text)
    if remove_nums:
        text = re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)
    if normalize_ws:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_parentheses(text):
    """Remove content within parentheses from text."""
    return re.sub(r"\([^()]*\)", "", text)


def normalize_whitespace(text):
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def remove_numbering(text):
    """Remove numbering from the beginning of text."""
    return re.sub(r"^(?:\d+[\.\)、]?\s*[\-\—\–]?\s*)?", "", text)


def extract_movie_name(text):
    """
    Extract and clean movie name from file path or text.
    
    Args:
        text: Raw text containing movie name
        
    Returns:
        Cleaned movie name
    """
    # Extract filename if it's a path
    filename = text.split('/')[-1]
    # Replace common separators with spaces
    cleaned_name = filename.replace('_', ' ').replace('-', ' ').replace('>', ' ')
    # Apply cleaning functions
    return normalize_whitespace(clean_parentheses(cleaned_name))


def find_nearest_movie(target_name, candidate_movies):
    """
    Find the nearest movie name using edit distance.
    
    Args:
        target_name: The movie name to match
        candidate_movies: List of candidate movie names
        
    Returns:
        Dictionary with matching information
    """
    # Remove duplicates from candidates
    unique_candidates = list(set(candidate_movies))
    
    # Calculate edit distances
    distances = [edit_distance(target_name.lower(), candidate.lower()) 
                for candidate in unique_candidates]
    
    # Find nearest match
    nearest_index = np.argmin(distances)
    nearest_movie = unique_candidates[nearest_index]
    
    return {
        'movie_name': target_name, 
        'min_edit_distance': distances[nearest_index], 
        'nearest_movie': nearest_movie
    }


def extract_recommendation_list(text, movie_candidates=None):
    """
    Extract recommendation list from text output.
    
    Args:
        text: Text containing recommendations
        movie_candidates: Optional list of valid movie names for matching
        
    Returns:
        Tuple of (recommendation_list, preference_text)
    """
    try:
        # Try to split on first numbered item
        preference_text, recommendation_text = text.split('1.', maxsplit=1)
    except Exception as e:
        print(e)
        preference_text = ""
        # Fallback: replace commas with newlines for parsing
        recommendation_text = text.replace(',', '\n')
    
    # Extract and clean recommendation items using the consolidated function
    raw_recommendations = [
        clean_text_elements(item.strip()) for item in recommendation_text.split('\n')
    ]
    
    # Match against candidates if provided
    recommendation_list = ([find_nearest_movie(item, movie_candidates) for item in raw_recommendations] 
                         if movie_candidates is not None else raw_recommendations)
    
    return recommendation_list, preference_text


# ============================================================================
# METRICS CALCULATION
# ============================================================================

# Initialize ROUGE scorer
rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL', 'rougeLsum'], use_stemmer=True)


def calculate_metrics(prediction, ground_truth_answers):
    """
    Calculate comprehensive metrics for prediction evaluation.
    
    Args:
        prediction: The predicted text
        ground_truth_answers: Ground truth answer(s) - can be string or list
        
    Returns:
        Dictionary of calculated metrics
    """
    # Calculate basic metrics using maximum over ground truths
    metrics = {
        "exact_match": drqa_metric_max_over_ground_truths(drqa_exact_match_score, prediction, ground_truth_answers),
        "f1": drqa_metric_max_over_ground_truths(lambda x, y: f1_score(x, y)[0], prediction, ground_truth_answers),
        "substring_exact_match": drqa_metric_max_over_ground_truths(substring_exact_match_score, prediction, ground_truth_answers)
    }

    # Normalize ground truth answers for ROUGE calculation
    if isinstance(ground_truth_answers, str):
        answer_list = [ground_truth_answers]
    elif ground_truth_answers and isinstance(ground_truth_answers[0], list):
        answer_list = [answer for answer_sublist in ground_truth_answers for answer in answer_sublist]
    else:
        answer_list = ground_truth_answers

    # Calculate ROUGE scores
    rouge_scores = [rouge_scorer_instance.score(target=answer, prediction=prediction) for answer in answer_list]
    
    # Extract ROUGE metrics
    for rouge_type in rouge_scorer_instance.rouge_types:
        metrics[rouge_type + "_f1"] = max(score[rouge_type].fmeasure for score in rouge_scores)
        metrics[rouge_type + "_recall"] = max(score[rouge_type].recall for score in rouge_scores)

    return metrics


# ============================================================================
# DATASET-SPECIFIC POST-PROCESSING
# ============================================================================

def post_process(output, answer, dataset_config):
    """
    Apply dataset-specific post-processing to model outputs.
    
    Args:
        output: Model output dictionary
        answer: Ground truth answer
        dataset_config: Dataset configuration dictionary
        
    Returns:
        Tuple of (metrics_dict, additional_info_dict)
    """
    sub_dataset_name = dataset_config['sub_dataset']
    
    # Route to appropriate post-processing based on dataset type
    if 'icl' in sub_dataset_name:
        return _process_icl_dataset(output, answer)
    elif 'eventqa' in sub_dataset_name:
        return _process_eventqa_dataset(output, answer)
    elif 'infbench' in sub_dataset_name or 'longmemeval' in sub_dataset_name:
        return _process_infbench_longmemeval_dataset(output, answer, dataset_config)
    elif 'ruler' in sub_dataset_name or 'mermory_merging' in sub_dataset_name:
        return _process_ruler_memory_merging_dataset(output, answer, dataset_config)
    elif 'recsys' in sub_dataset_name:
        return _process_recsys_dataset(output, answer)
    else:
        return default_post_process(output, answer)


def _process_icl_dataset(output, answer):
    """Process in-context learning dataset outputs."""
    prediction = output["output"]
    parsed_prediction = parse_output(prediction)
    metrics = calculate_metrics(parsed_prediction, answer)
    return metrics, {"parsed_output": parsed_prediction}


def _process_eventqa_dataset(output, answer):
    """Process EventQA dataset outputs with recall calculation."""
    prediction = output["output"]
    
    # Calculate recall: fraction of answer elements found in prediction
    recall_score = sum(answer_element.lower() in prediction.lower() for answer_element in answer) / len(answer)
    
    # Convert to binary recall (1 if all elements found, 0 otherwise)
    binary_recall = int(recall_score == 1)
    
    # Calculate standard metrics
    parsed_prediction = parse_output(prediction)
    standard_metrics = calculate_metrics(parsed_prediction, answer)
    standard_metrics["eventqa_recall"] = binary_recall

    return standard_metrics, {"parsed_output": parsed_prediction}


def _process_infbench_longmemeval_dataset(output, answer, sub_dataset_name):
    """Process InfBench/LongMemEval dataset outputs."""
    
    if "choice_eng" in sub_dataset_name:
        return _process_choice_eng_dataset(output, answer)
    else:
        return default_post_process(output, answer)


def _process_choice_eng_dataset(output, answer):
    """Process choice_eng dataset with special substring matching."""
    prediction = output["output"]
    
    # Calculate standard metrics (excluding substring_exact_match initially)
    standard_metrics = calculate_metrics(prediction, answer)
    standard_metrics.pop("substring_exact_match", None)

    # Try with parsed output and take maximum scores
    parsed_prediction = parse_output(prediction)
    if parsed_prediction is not None:
        parsed_metrics = calculate_metrics(parsed_prediction, answer)
        parsed_metrics.pop("substring_exact_match", None)
        standard_metrics = {
            metric_name: max(original_score, parsed_metrics[metric_name]) 
            for metric_name, original_score in standard_metrics.items()
        }

    # Special substring matching for choice_eng (check if answer option is in prediction)
    substring_match = answer[1].lower() in prediction.lower()
    standard_metrics["substring_exact_match"] = substring_match
    
    # If substring match found, also set exact_match to True
    if substring_match:
        standard_metrics["exact_match"] = True
        
    return standard_metrics, {"parsed_output": parsed_prediction}


def _process_ruler_memory_merging_dataset(output, answer, sub_dataset_name):
    """Process Ruler/Memory merging dataset outputs."""
    
    if "ruler_niah" in sub_dataset_name:
        prediction = output["output"]
        recall_score = sum([
            answer_element.lower() in prediction.lower() 
            for answer_element in answer
        ]) / len(answer)
        metrics = {"ruler_recall": recall_score}
        return metrics, {"parsed_output": prediction}
    else:
        return default_post_process(output, answer)


def _process_recsys_dataset(output, answer):
    """Process recommendation system dataset outputs."""
    # Load movie entity mapping
    entity_mapping_path = "/home/wangyu/data/entity2id.json"
    name_to_id = json.load(open(entity_mapping_path))
    id_to_name = {entity_id: extract_movie_name(name) for name, entity_id in name_to_id.items()}

    # Get movie candidates and parse prediction
    prediction = output["output"]
    movie_candidates = list(id_to_name.values())
    
    predicted_list, _ = extract_recommendation_list(prediction, movie_candidates)
    predicted_movies = [item['nearest_movie'] for item in predicted_list]

    # Convert ground truth IDs to movie names / answer is a string with movie ids divided by comma
    ground_truth_ids = [int(movie_id.strip()) for movie_id in answer]
    ground_truth_movies = [id_to_name[movie_id] for movie_id in ground_truth_ids]

    # Calculate recall at different cutoffs
    recall_at_1 = sum([movie in predicted_movies[:1] for movie in ground_truth_movies]) / len(ground_truth_movies)
    recall_at_5 = sum([movie in predicted_movies[:5] for movie in ground_truth_movies]) / len(ground_truth_movies)
    recall_at_10 = sum([movie in predicted_movies[:10] for movie in ground_truth_movies]) / len(ground_truth_movies)
    
    metrics = {
        "recsys_recall@1": recall_at_1,
        "recsys_recall@5": recall_at_5,
        "recsys_recall@10": recall_at_10
    }
    
    return metrics, {"parsed_output": predicted_movies, "gt_movies": ground_truth_movies}


def default_post_process(output, answer):
    """
    Default post-processing function for model outputs.
    
    Args:
        output: Model output dictionary
        answer: Ground truth answer
        
    Returns:
        Tuple of (metrics_dict, additional_info_dict)
    """
    prediction = output["output"]
    metrics = calculate_metrics(prediction, answer)
    
    # Try parsing output and take maximum scores
    parsed_prediction = parse_output(prediction)
    if parsed_prediction is not None:
        parsed_metrics = calculate_metrics(parsed_prediction, answer)
        metrics = {metric_name: max(original_score, parsed_metrics[metric_name]) 
                  for metric_name, original_score in metrics.items()}
    
    return metrics, {"parsed_output": parsed_prediction}


# ============================================================================
# METRICS SUMMARIZATION
# ============================================================================

def metrics_summarization(output, query, answer, dataset_config, metrics, results, query_id=None, qa_pair_id=None):
    """
    Summarize metrics for a single query and update overall metrics and results.
    
    Args:
        output: Model output dictionary
        query: The input query
        answer: Ground truth answer
        dataset_config: Dataset configuration
        metrics: Running metrics dictionary
        results: List of result records
        query_id: Optional query identifier
        qa_pair_id: Optional qa_pair_id for the question
        
    Returns:
        Tuple of (updated_metrics, updated_results)
    """
    if output is None:
        logger.info("Skipping example because the model returned None")
        return metrics, results
    
    # Calculate dataset-specific metrics
    calculated_metrics, additional_info = post_process(output, answer, dataset_config)
    output.update({**additional_info, **calculated_metrics})
    
    # Update running metrics
    for metric_name, metric_value in calculated_metrics.items():
        metrics[metric_name].append(metric_value)

    # Update system metrics
    metrics["input_len"].append(output["input_len"])
    metrics["output_len"].append(output["output_len"])
    metrics["memory_construction_time"].append(output.get("memory_construction_time", 0))
    metrics["query_time_len"].append(output.get("query_time_len", 0))
    
    # Create result record
    result_record = {**output, "answer": answer, 'query': query}
    if query_id is not None:
        result_record["query_id"] = query_id
    if qa_pair_id is not None:
        result_record["qa_pair_id"] = qa_pair_id
    results.append(result_record)

    # Log debug information if enabled
    if dataset_config['debug']:
        logger.info(f"Input length: {output['input_len']}")
        logger.info(f"Answer: {answer}")
        logger.info(f"Output: {output['output']}")
        logger.info(f"Parsed output: {output['parsed_output']}")
                    
    return metrics, results

def _extract_answer_from_response(predicted_answer):
    """
    Extract the final answer from a ``\\boxed{...}`` expression, returning the
    original string when the marker is absent or malformed.
    """

    boxed_pattern = r'\\boxed\{([^}]*)\}'
    boxed_match = re.search(boxed_pattern, predicted_answer)
    if boxed_match:
        return boxed_match.group(1).strip()

    return predicted_answer.strip()

def evaluate_wrt_source(output, answer, sub_dataset_name):

    # Route to appropriate post-processing based on dataset type
    if 'icl' in sub_dataset_name:
        return _process_icl_dataset(output, answer)[0]['substring_exact_match']
    elif 'eventqa' in sub_dataset_name:
        return float(_process_eventqa_dataset(output, answer)[0]['eventqa_recall'])
    elif 'infbench' in sub_dataset_name:
        return _process_infbench_longmemeval_dataset(output, answer, sub_dataset_name)[0]['rougeL_f1']
    # elif 'longmemeval' in sub_dataset_name: 
    #     return default_post_process(output, answer)[0]['substring_exact_match']
    elif 'ruler' in sub_dataset_name or 'mermory_merging' in sub_dataset_name:
        # return _process_ruler_memory_merging_dataset(output, answer, dataset_config)
        return float(_process_ruler_memory_merging_dataset(output, answer, sub_dataset_name)[0]['substring_exact_match'])
    elif 'recsys' in sub_dataset_name:
        return _process_recsys_dataset(output, answer)[0]['recsys_recall@5']
    elif 'detectiveqa' in sub_dataset_name:
        extracted_answer = _extract_answer_from_response(output['output'])
        return float(answer == extracted_answer)
    else:
        return default_post_process(output, answer)[0]['substring_exact_match']
