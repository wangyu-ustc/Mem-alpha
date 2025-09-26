import json
import tiktoken
import numpy as np
from json_repair import repair_json
from nltk.metrics.distance import edit_distance
# Add rapidfuzz for faster string matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("rapidfuzz not available. Install with: pip install rapidfuzz")

def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens using tiktoken"""
    import traceback
    
    encoding = tiktoken.encoding_for_model(model)
    
    # Convert input to string if it's not already a string
    if not isinstance(text, str):
        print(f"!!!! WARNING: Non-string input to count_tokens: {repr(text)} (type: {type(text)})")
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
        length = len(encoding.encode(text))
    except Exception as e:
        print(f"!!!! ERROR when processing text: {text}")
        print(f"!!!! ERROR type: {type(e).__name__}: {e}")
        print("!!!! STACK TRACE:")
        traceback.print_stack()
        print("!!!! END STACK TRACE")
        return 0
    return len(encoding.encode(text))

def evaluate_eurlex(predicted_answers, gold_answers):

    with open("transform_TC/eurovoc_concepts.jsonl", "r") as f:
        eurovoc_concepts = [json.loads(line) for line in f]

    eurovoc_concepts = {concept["title"]:concept["id"] for concept in eurovoc_concepts}

    all_f1_scores = []
    for predicted_answer, gold_answer in zip(predicted_answers, gold_answers):

        # extract the list from predicted_answer:
        try:
            predicted_answer = eval(predicted_answer)
        except:
            pass

        if isinstance(predicted_answer, str):
            try:
                predicted_answer = json.loads(predicted_answer)
            except:
                pass

        if isinstance(predicted_answer, str):
            try:
                string = repair_json(string)
                predicted_answer = json_loads(string)
            except:
                pass

        if isinstance(predicted_answer, str):
            return 0
        
        assert isinstance(predicted_answer, list)
        gold_answer = json.loads(gold_answer)

        gold_answer_ids = [eurovoc_concepts[concept] for concept in gold_answer]
        predicted_answer_ids = []
        for concept in predicted_answer:
            # find the nearest concept in eurovoc_concepts using fast method:
            nearest_concept = find_nearest_concept_fast(concept, eurovoc_concepts)
            predicted_answer_ids.append(eurovoc_concepts[nearest_concept])

        # Now let's calculate the F-1 score:
        precision = len(set(predicted_answer_ids) & set(gold_answer_ids)) / len(predicted_answer_ids)
        recall = len(set(predicted_answer_ids) & set(gold_answer_ids)) / len(gold_answer_ids)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        all_f1_scores.append(f1)

    return np.mean(all_f1_scores)

def find_nearest_concept_fast(concept, eurovoc_concepts):
    """
    Find nearest concept using rapidfuzz for much faster string matching.
    Falls back to edit_distance if rapidfuzz is not available.
    """
    if RAPIDFUZZ_AVAILABLE:
        # Use rapidfuzz - typically 10-100x faster than edit_distance
        result = process.extractOne(concept, eurovoc_concepts.keys(), scorer=fuzz.ratio)
        return result[0] if result else list(eurovoc_concepts.keys())[0]
    else:
        # Fallback to original method
        return min(eurovoc_concepts.keys(), key=lambda x: edit_distance(x, concept))
