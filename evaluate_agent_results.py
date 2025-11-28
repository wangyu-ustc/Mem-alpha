import re
import os
import json
import logging
import numpy as np
from dotenv import load_dotenv
from rouge_score import rouge_scorer
from typing import List, Dict, Any
from openai import OpenAI

from memalpha.llm_agent.metrics import evaluate_wrt_source, _extract_answer_from_response

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentResultsEvaluator:
    """
    Evaluator for agent results that uses data source specific evaluation methods
    borrowed from long_context_eval.py
    """

    def __init__(self):
        # Initialize ROUGE scorer for booksum evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.client = OpenAI(
            base_url=os.getenv("QWEN_URL"),
            api_key=os.getenv("OPENROUTER_API_KEY", "EMPTY")
        )
        self.qwen_model = os.getenv("QWEN_MODEL_NAME")

    def _compute_rouge_score(self, prediction: str, reference: str) -> float:
        """Compute ROUGE score between prediction and reference text."""
        scores = self.rouge_scorer.score(reference, prediction)
        rouge1_f1 = scores['rouge1'].fmeasure
        rouge2_f1 = scores['rouge2'].fmeasure
        rougeL_f1 = scores['rougeL'].fmeasure

        return (rouge1_f1 + rouge2_f1 + rougeL_f1) / 3.0

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

        elif data_source == 'pubmed-rct' or "icl" in data_source:
            # PUBMED dataset evaluation: MUST be ONLY a single digit
            extracted_answer = _extract_answer_from_response(predicted_answer)

            # Remove quotes and strip whitespace
            extracted_answer = extracted_answer.strip('"\'').strip()

            # STRICT pattern: must be EXACTLY a single digit with nothing else
            single_digit_pattern = r'^\d+$'

            if isinstance(gold_answer, list):
                gold_answer = gold_answer[0]

            if not re.match(single_digit_pattern, extracted_answer):
                return 0.0

            gold_num = str(gold_answer).strip('"\'').strip()

            return 1.0 if extracted_answer == gold_num else 0.0

        elif data_source == 'squad' or data_source == 'hotpotqa':
            # Default: containment score for other data sources
            if isinstance(gold_answer, list):
                answer_text = str(gold_answer[0]) if gold_answer else ""
            else:
                answer_text = gold_answer.get('text', gold_answer) if isinstance(gold_answer, dict) else str(gold_answer)

            return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

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

        elif data_source == 'lme_train' or data_source == 'longmemeval_s*':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, gold_answer, predicted_answer)

            response = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )

            if "yes" in response.choices[0].message.content.strip().lower() and "no" not in response.choices[0].message.content.strip().lower():
                return 1.0
            else:
                return 0.0

        else:
            # memory agent bench
            return evaluate_wrt_source({'output': predicted_answer}, gold_answer, data_source)

    def evaluate_agent_results(self, agent_dir: str) -> Dict[str, Any]:
        """
        Evaluate results for a single agent directory

        Args:
            agent_dir: Path to agent results directory containing data_instance_info.json and results.json

        Returns:
            Dictionary containing evaluation metrics
        """
        # Load data source info
        data_instance_info_path = os.path.join(agent_dir, "data_instance_info.json")
        with open(data_instance_info_path, 'r') as f:
            data_instance_info = json.load(f)
        data_source = data_instance_info.get('data_source')

        if not data_source:
            raise ValueError(f"No data_source found in {data_instance_info_path}")

        # Load results
        results_path = os.path.join(agent_dir, "results.json")
        with open(results_path, 'r') as f:
            results = json.load(f)

        # Load memory state if it exists
        memory_state_path = os.path.join(agent_dir, "agent_state.json")
        total_memory_length = 0
        if os.path.exists(memory_state_path):
            try:
                with open(memory_state_path, 'r') as f:
                    memory_state = json.load(f)

                # Calculate total memory length
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")

                # Count tokens in core memory if exists
                if 'core' in memory_state and memory_state['core']:
                    if isinstance(memory_state['core'], str):
                        total_memory_length += len(tokenizer(memory_state['core']).input_ids)
                    elif isinstance(memory_state['core'], list):
                        for item in memory_state['core']:
                            if isinstance(item, str):
                                total_memory_length += len(tokenizer(item).input_ids)

                # Count tokens in semantic memory
                if 'semantic' in memory_state and memory_state['semantic']:
                    for item in memory_state['semantic']:
                        total_memory_length += len(tokenizer(list(item.values())[0]).input_ids)

                # Count tokens in episodic memory
                if 'episodic' in memory_state and memory_state['episodic']:
                    for item in memory_state['episodic']:
                        total_memory_length += len(tokenizer(list(item.values())[0]).input_ids)

            except Exception as e:
                logger.warning(f"Could not load memory state from {memory_state_path}: {e}")
                total_memory_length = 0

        if not results:
            logger.warning(f"No results found in {results_path}")
            return {
                "data_source": data_source,
                "error": "No results found"
            }

        # Group results by instance and evaluate
        instance_scores = {}

        for result in results:
            # Get instance identifier (use index if no explicit instance_id)
            instance_id = result.get('instance_id', result.get('question_id', len(instance_scores)))

            score = self._compute_score(
                data_source=data_source,
                predicted_answer=result['response'],
                gold_answer=result['answer'],
                question=result['question']
            )

            if isinstance(score, bool):
                score = 1.0 if score else 0.0

            result['score'] = score

            # Group scores by instance
            if instance_id not in instance_scores:
                instance_scores[instance_id] = []
            instance_scores[instance_id].append(score)

        # Calculate average score for each instance
        instance_avg_scores = []
        for instance_id, scores_list in instance_scores.items():
            instance_avg = np.mean(scores_list)
            instance_avg_scores.append(instance_avg)

        # Overall scores for compatibility with existing code
        scores = instance_avg_scores

        # Save updated results with scores
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate metrics
        metrics = {
            "data_source": data_source,
            "num_instances": len(instance_scores),
            "num_total_results": len(results),
            "avg_score_per_instance": np.mean(scores) if scores else 0.0,
            "min_instance_score": np.min(scores) if scores else 0.0,
            "max_instance_score": np.max(scores) if scores else 0.0,
            "std_instance_score": np.std(scores) if scores else 0.0,
            "total_memory_length": total_memory_length,
            # Keep backward compatibility
            "num_questions": len(instance_scores),
            "avg_score": np.mean(scores) if scores else 0.0,
            "min_score": np.min(scores) if scores else 0.0,
            "max_score": np.max(scores) if scores else 0.0,
            "std_score": np.std(scores) if scores else 0.0
        }

        return metrics

def evaluate_all_agents(base_dir: str) -> List[Dict[str, Any]]:
    """
    Evaluate results for all agent directories under the base directory

    Args:
        base_dir: Base directory containing agent result directories

    Returns:
        List of evaluation metrics for each agent
    """
    evaluator = AgentResultsEvaluator()
    all_metrics = []

    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        if 'data_instance_info.json' in files and 'results.json' in files:
            metrics = evaluator.evaluate_agent_results(root)
            metrics['agent_dir'] = root
            all_metrics.append(metrics)
    return all_metrics

def main():
    """Main function to run the evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate agent results')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing agent results')
    parser.add_argument('--output', type=str, default='evaluation_metrics.json',
                       help='Output file to save metrics')

    args = parser.parse_args()

    # Run evaluation
    all_metrics = evaluate_all_agents(args.base_dir)

    # Save metrics to base_dir
    output_path = os.path.join(args.base_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # Group metrics by data source
    grouped_metrics = {}
    for metrics in all_metrics:
        data_source = metrics['data_source']
        if data_source not in grouped_metrics:
            grouped_metrics[data_source] = []
        grouped_metrics[data_source].append(metrics)

    # Print summary grouped by data source
    print("\nEvaluation Summary by Data Source:")
    print("=" * 50)

    for data_source, metrics_list in grouped_metrics.items():
        print(f"\nData Source: {data_source}")
        print("-" * 30)

        # Calculate aggregate statistics for this data source
        total_instances = sum(m['num_instances'] for m in metrics_list)
        avg_scores_per_instance = [m['avg_score_per_instance'] for m in metrics_list]
        overall_avg = np.mean(avg_scores_per_instance) if avg_scores_per_instance else 0.0
        overall_min = min(m['min_instance_score'] for m in metrics_list)
        overall_max = max(m['max_instance_score'] for m in metrics_list)
        overall_std = np.std(avg_scores_per_instance) if len(avg_scores_per_instance) > 1 else 0.0

        # Calculate average memory length
        memory_lengths = [m.get('total_memory_length', 0) for m in metrics_list]
        avg_memory_length = np.mean(memory_lengths) if memory_lengths else 0.0

        print(f"Total agents: {len(metrics_list)}")
        print(f"Total instances: {total_instances}")
        print(f"Overall average score per instance: {overall_avg:.3f}")
        print(f"Overall min/max instance score: {overall_min:.3f}/{overall_max:.3f}")
        print(f"Standard deviation across agents: {overall_std:.3f}")
        print(f"Average total memory length: {avg_memory_length:.0f} tokens")

    print("\nMetrics saved to:", output_path)

if __name__ == "__main__":
    main()
