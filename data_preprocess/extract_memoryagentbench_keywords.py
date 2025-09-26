import json
import os
import pandas as pd
from typing import List, Dict
from openai import AzureOpenAI
from tqdm import tqdm
import argparse
import dotenv
from datasets import load_dataset
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

dotenv.load_dotenv()

class MemoryAgentBenchKeywordExtractor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2025-01-01-preview",
            azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
        )

    def extract_keywords_from_answers(self, answers: str) -> List[str]:
        """Extract keywords from answers using GPT-4.1"""

        prompt = f"""Analyze the following answers and extract the most important keywords. Focus on:

1. Character names (main and supporting characters)
2. Key events and plot points
3. Important locations/settings
4. Central themes and concepts
5. Significant objects or symbols
6. Time periods or dates mentioned
7. Key relationships between characters
8. Important actions or decisions
9. Specific facts, names, or details mentioned in the answers

Example:
Answers: "Elizabeth Bennet is the protagonist. Mr. Darcy is initially proud and disagreeable. The story takes place in Hertfordshire and later at Pemberley estate. The main themes include prejudice, social class, and marriage. Elizabeth initially rejects Darcy's proposal but later falls in love with him after understanding his true character."

Keywords: Elizabeth Bennet, Mr. Darcy, protagonist, proud, disagreeable, Hertfordshire, Pemberley estate, prejudice, social class, marriage, proposal, rejection, love, character development, misunderstanding

Now analyze these answers:
{answers}

Extract keywords/phrases that capture the essential information in these answers, make sure they are complete and cover all aspects mentioned.
Return ONLY a comma-separated list of keywords, nothing else.
Focus on concrete, specific terms rather than generic words.
Include both single words and short phrases (2-3 words max).
Prioritize proper nouns, specific events, and unique concepts mentioned in the answers."""

        messages = [
            {"role": "system", "content": "You are an expert at identifying key information from text answers."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model='gpt-4.1',
                messages=messages,
                temperature=0.1
            )

            keywords_text = response.choices[0].message.content.strip()
            # Clean and split keywords
            keywords = [k.strip() for k in keywords_text.split(',')]
            # Remove empty strings and duplicates while preserving order
            seen = set()
            unique_keywords = []
            for k in keywords:
                k_lower = k.lower()
                if k and k_lower not in seen:
                    seen.add(k_lower)
                    unique_keywords.append(k)

            return unique_keywords

        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return []

    def process_memoryagentbench_parquet(self,
                                         data_df: pd.DataFrame,
                                         output_path: str,
                                         sample_size: int = None):
        """Process MemoryAgentBench parquet data and extract keywords from all answers"""

        print("Processing data from Hugging Face dataset.")
        df = data_df

        if sample_size:
            df = df.head(sample_size)
            print(f"Processing {sample_size} samples")
        else:
            print(f"Processing all {len(df)} samples")

        # Always load existing mappings if the output file exists
        keyword_mappings = {}
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                keyword_mappings = json.load(f)
            print(f"Loaded {len(keyword_mappings)} existing keyword mappings from {output_path}")

        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keywords"):
            # Use row index as instance_id since there's no explicit instance_id column
            instance_id = str(idx)

            # Skip if already processed
            if instance_id in keyword_mappings:
                continue

            # Get answers from the row
            answers = row['answers']
            
            if not answers or pd.isna(answers):
                print(f"Warning: No answers found for row {idx}")
                continue

            # Convert answers to string if it's not already
            answers_str = str(answers)

            # Extract keywords
            keywords = self.extract_keywords_from_answers(answers_str)

            # Store mapping (keeping same format as booksum for consistency)
            keyword_mappings[instance_id] = {
                'summary': answers_str,
                'keywords': keywords,
                'num_keywords': len(keywords),
                'context': str(row['context']) if 'context' in row and not pd.isna(row['context']) else None,
                'questions': str(row['questions']) if 'questions' in row and not pd.isna(row['questions']) else None,
                'metadata': str(row['metadata']) if 'metadata' in row and not pd.isna(row['metadata']) else None
            }

            # Save after every API query to avoid losing progress
            with open(output_path, 'w') as f:
                json.dump(keyword_mappings, f, indent=2)

        # Print statistics
        total_entries = len(keyword_mappings)
        avg_keywords = sum(entry['num_keywords'] for entry in keyword_mappings.values()) / total_entries if total_entries > 0 else 0

        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        print(f"Total entries processed: {total_entries}")
        print(f"Average keywords per answer: {avg_keywords:.2f}")
        print(f"Output saved to: {output_path}")

        return keyword_mappings


def main():
    parser = argparse.ArgumentParser(description='Extract keywords from MemoryAgentBench answers')
    parser.add_argument('--output_path', type=str,
                        default='./data/memoryagentbench_answers_to_keywords_mapping.json',
                        help='Path to save keyword mappings')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to process (default: all)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Define Hugging Face dataset and file information
    dataset_name = "ai-hyz/MemoryAgentBench"
    file_path = "data/Long_Range_Understanding-00000-of-00001.parquet"
    parquet_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{file_path}"

    print(f"Attempting to download and load {parquet_url}...")
    try:
        # Load the parquet file directly from Hugging Face
        dataset = load_dataset('parquet', data_files={'train': parquet_url})
        df = dataset['train'].to_pandas()
        print("Dataset loaded successfully!")

    except Exception as e:
        print(f"Error loading dataset from Hugging Face: {e}")
        return

    # Initialize extractor
    extractor = MemoryAgentBenchKeywordExtractor()

    # Process the dataset
    keyword_mappings = extractor.process_memoryagentbench_parquet(
        data_df=df,
        output_path=args.output_path,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()