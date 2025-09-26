import json
import os
import pandas as pd
from typing import List, Dict
from openai import AzureOpenAI
from tqdm import tqdm
import argparse
import dotenv

dotenv.load_dotenv()

class BookSumKeywordExtractor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2025-01-01-preview",
            azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
        )

    def extract_keywords_from_summary(self, summary: str) -> List[str]:
        """Extract keywords from a summary using GPT-4.1"""

        prompt = f"""Analyze the following book summary and extract the most important keywords. Focus on:

1. Character names (main and supporting characters)
2. Key events and plot points
3. Important locations/settings
4. Central themes and concepts
5. Significant objects or symbols
6. Time periods or dates mentioned
7. Key relationships between characters
8. Important actions or decisions

Example:
Summary: "Elizabeth Bennet meets Mr. Darcy at a ball in Hertfordshire. Initially, she finds him proud and disagreeable. After learning about his past with Wickham and his role in separating Jane and Bingley, her dislike intensifies. However, when Darcy proposes and she rejects him, he writes a letter explaining his actions. Elizabeth realizes her prejudices and eventually falls in love with him after visiting Pemberley."

Keywords: Elizabeth Bennet, Mr. Darcy, ball, Hertfordshire, proud, Wickham, Jane, Bingley, proposal, rejection, letter, prejudices, Pemberley, love, Pride and Prejudice themes, marriage, social class, first impressions, misunderstanding, character growth

Now analyze this summary:
{summary}

Extract keywords/phrases that capture the essential information in this summary, make sure they are complete and cover all aspects of the story.
Return ONLY a comma-separated list of keywords, nothing else.
Focus on concrete, specific terms rather than generic words.
Include both single words and short phrases (2-3 words max).
Prioritize proper nouns, specific events, and unique concepts."""

        messages = [
            {"role": "system", "content": "You are an expert at identifying key information in text summaries."},
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

    def process_booksum_parquet(self,
                               parquet_path: str,
                               output_path: str,
                               sample_size: int = None):
        """Process booksum parquet file and extract keywords from all summaries"""

        print(f"Loading data from {parquet_path}")
        df = pd.read_parquet(parquet_path)

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
            instance_id = row['instance_id']

            # Skip if already processed
            if str(instance_id) in keyword_mappings:
                continue

            # Parse questions and answers
            questions_and_answers = json.loads(row['questions_and_answers'])

            # Find summarization question and answer
            summary = None
            for qa in questions_and_answers:
                if 'summarize' in qa['question'].lower():
                    summary = qa['answer']
                    break

            if not summary:
                print(f"Warning: No summary found for instance {instance_id}")
                continue

            # Extract keywords
            keywords = self.extract_keywords_from_summary(summary)

            # Store mapping
            keyword_mappings[str(instance_id)] = {
                'summary': summary,
                'keywords': keywords,
                'num_keywords': len(keywords)
            }

            # Save after every API query to avoid losing progress
            with open(output_path, 'w') as f:
                json.dump(keyword_mappings, f, indent=2)
            print(f"Saved {len(keyword_mappings)} entries to {output_path}")

        # Print statistics
        total_entries = len(keyword_mappings)
        avg_keywords = sum(entry['num_keywords'] for entry in keyword_mappings.values()) / total_entries if total_entries > 0 else 0

        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        print(f"Total entries processed: {total_entries}")
        print(f"Average keywords per summary: {avg_keywords:.2f}")
        print(f"Output saved to: {output_path}")

        return keyword_mappings

    def replace_answers_with_keywords(self,
                                     parquet_path: str,
                                     keyword_mapping_path: str,
                                     output_parquet_path: str):
        """Load processed booksum data and replace answers with extracted keywords"""
        
        print(f"Loading parquet data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        original_count = len(df)
        print(f"Loaded {original_count} rows")
        
        print(f"Loading keyword mappings from {keyword_mapping_path}")
        with open(keyword_mapping_path, 'r') as f:
            keyword_mappings = json.load(f)
        print(f"Loaded {len(keyword_mappings)} keyword mappings")
        
        # Track rows to delete
        rows_to_delete = []
        replaced_count = 0
        not_found_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Replacing answers with keywords"):
            instance_id = str(row['instance_id'])
            
            if instance_id not in keyword_mappings:
                not_found_count += 1
                print(f"Warning: No keyword mapping found for instance {instance_id} - will be deleted")
                rows_to_delete.append(idx)
                continue
            
            # Parse questions and answers
            questions_and_answers = json.loads(row['questions_and_answers'])
            
            # Get keywords for this instance
            keywords = keyword_mappings[instance_id]['keywords']
            keyword_string = ", ".join(keywords)
            
            # Replace answer in each Q&A pair (particularly the summarization one)
            modified = False
            for qa in questions_and_answers:
                if 'summarize' in qa['question'].lower():
                    qa['answer'] = keyword_string
                    modified = True
                    replaced_count += 1
            
            # Update the dataframe with modified Q&A
            if modified:
                df.at[idx, 'questions_and_answers'] = json.dumps(questions_and_answers)
        
        # Delete rows without keyword mappings
        if rows_to_delete:
            print(f"\nDeleting {len(rows_to_delete)} instances without keyword mappings...")
            df = df.drop(rows_to_delete)
            df = df.reset_index(drop=True)  # Reset index after deletion
        
        # Save the modified dataframe
        print(f"\nSaving modified data to {output_parquet_path}")
        df.to_parquet(output_parquet_path, index=False)
        
        # Print statistics
        print("\n" + "="*50)
        print("REPLACEMENT COMPLETE")
        print("="*50)
        print(f"Original rows: {original_count}")
        print(f"Rows deleted (no keyword mapping): {not_found_count}")
        print(f"Final rows saved: {len(df)}")
        print(f"Answers replaced with keywords: {replaced_count}")
        print(f"Output saved to: {output_parquet_path}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Extract keywords from booksum summaries or replace answers with keywords')
    parser.add_argument('--mode', type=str, choices=['extract', 'replace'], default='extract',
                       help='Mode: extract keywords or replace answers with keywords')
    parser.add_argument('--parquet_path', type=str,
                       default='/home/wangyu/work/Mem-alpha/data/memalpha/processed_booksum_data.parquet',
                       help='Path to the booksum parquet file')
    parser.add_argument('--output_path', type=str,
                       default='./data/booksum_summary_to_keywords_mapping.json',
                       help='Path to save keyword mappings (for extract mode)')
    parser.add_argument('--keyword_mapping_path', type=str,
                       default='/home/wangyu/work/Mem-alpha/data/booksum_summary_to_keywords_mapping.json',
                       help='Path to keyword mapping file (for replace mode)')
    parser.add_argument('--output_parquet_path', type=str,
                       default='/home/wangyu/work/Mem-alpha/data/memalpha/processed_booksum_data_with_keywords.parquet',
                       help='Path to save modified parquet file (for replace mode)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to process (default: all, only for extract mode)')

    args = parser.parse_args()

    # Initialize extractor
    extractor = BookSumKeywordExtractor()

    if args.mode == 'extract':
        # Process the dataset to extract keywords
        keyword_mappings = extractor.process_booksum_parquet(
            parquet_path=args.parquet_path,
            output_path=args.output_path,
            sample_size=args.sample_size
        )
    elif args.mode == 'replace':
        # Replace answers with keywords
        modified_df = extractor.replace_answers_with_keywords(
            parquet_path=args.parquet_path,
            keyword_mapping_path=args.keyword_mapping_path,
            output_parquet_path=args.output_parquet_path
        )


if __name__ == "__main__":
    main()
