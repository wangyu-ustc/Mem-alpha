#!/usr/bin/env python3
"""
Script to shorten answers in perltqa_en.json using OpenAI API.
This script loads all question-answer pairs and uses OpenAI to make answers as brief as possible.
"""

import json
import os
import time
from typing import Dict, List, Any
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

class AnswerShortener:
    def __init__(self):
        """Initialize the Azure OpenAI client."""
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")
        
        self.azure_client = AzureOpenAI(
            api_key=self.api_key, 
            api_version="2025-01-01-preview", 
            azure_endpoint="https://jplml-resource.cognitiveservices.azure.com"
        )

    def shorten_answer(self, question: str, answer: str) -> str:
        """
        Use OpenAI to shorten an answer to be as brief as possible.
        Retries up to 3 times, carrying error information to subsequent attempts.
        
        Args:
            question: The original question
            answer: The original answer to be shortened
            
        Returns:
            The shortened answer
        """
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Build the error context string for the prompt
                error_context = ""
                if last_error is not None:
                    error_context = f"{last_error} "

                prompt = f"""Extract the key information from the answer as keywords or short phrases that directly answer the question. The output should be optimized for evaluation where the ground truth answer will be checked using "ground_truth in prediction".

Return ONLY the essential keywords/phrases, separated by ";". Remove all unnecessary words, articles, and connecting phrases.

Examples:
- Question: "What is the name of the protagonist?"
  Answer: "The name of the protagonist is Joey Tribbiani."
  Keywords: "Joey Tribbiani"

- Question: "What is Wang Xiaoming's educational background?"
  Answer: "Wang Xiaoming graduated from Computer Science and Technology with a bachelor's degree."
  Keywords: "Bachelor; Computer Science Technology"

- Question: "What year was the company founded?"
  Answer: "The company was established in 1995."
  Keywords: "1995"

- Question: "What are his hobbies?"
  Answer: "He enjoys playing basketball and reading books."
  Keywords: "Basketball; Reading books"

- Question: What is Zhang Xiaohong’s educational background?
  Answer: Zhang Xiaohong’s educational background is in chemical and pharmaceutical engineering.
  Keywords: Chemical; Pharmaceutical; Engineering

Note that the keywords will be used for evaluation with the pattern "keyword in prediction for keyword in keywords.split(';')". Thus in the last example you should not use "Chemical Engineering; Pharmaceutical Engineering" as the keywords because in that case the predicted answer "Chemical and Pharmaceutical Engineering" will not be considered as correct.

Note that every keyword should be strictly less than 3 words. {error_context}

Question: {question}
Original Answer: {answer}

Keywords:"""

                response = self.azure_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.1
                )
                
                shortened_answer = response.choices[0].message.content.strip()

                # Validate the answer
                if ";" in shortened_answer:
                    keywords = shortened_answer.split(";")
                else:
                    keywords = [shortened_answer]

                # Check each keyword for token length
                for keyword in keywords:
                    keyword = keyword.strip()  # Strip whitespace
                    if len(keyword.split(" ")) > 3:
                        error_msg = f"The previous answer was '{shortened_answer}' and the keyword '{keyword}' had too many tokens ({len(keyword.split(' '))} > 3)."
                        last_error = error_msg
                        raise ValueError(error_msg)

                # If we get here, the answer is valid
                return shortened_answer
                
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                # If this was the last attempt, return the original answer
                if attempt == max_retries - 1:
                    print(f"All {max_retries} attempts failed for question: {question}")
                    return None  # Return original answer if all attempts fail
                
                # Add a small delay before retrying
                time.sleep(0.5)

    def load_json_data(self, file_path: str) -> List[Dict]:
        """Load the JSON data from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_json_data(self, data: List[Dict], file_path: str) -> None:
        """Save the modified JSON data to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def process_all_answers(self, input_file: str, output_file: str) -> None:
        """
        Process all question-answer pairs in the JSON file and shorten the answers.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file with shortened answers
        """
        print(f"Loading data from {input_file}...")
        data = self.load_json_data(input_file)
        
        total_questions = 0
        processed_questions = 0
        
        # Count total questions first
        for person_data in data:
            for person_name, person_info in person_data.items():
                # Count questions in all sections
                sections = ['profile', 'social_relationship', 'events', 'dialogues']
                for section in sections:
                    if section in person_info:
                        if section == 'profile':
                            # Profile section has direct Q&A pairs
                            total_questions += len(person_info[section])
                        else:
                            # Other sections have nested structure
                            for item in person_info[section]:
                                for key, qa_list in item.items():
                                    total_questions += len(qa_list)
        
        print(f"Found {total_questions} question-answer pairs to process...")
        
        # Process each person's data across all sections
        for person_data in data:
            for person_name, person_info in person_data.items():
                print(f"Processing {person_name}...")
                
                sections = ['profile', 'social_relationship', 'events', 'dialogues']
                
                for section in sections:
                    if section in person_info:
                        print(f"  Processing {section} section...")
                        
                        if section == 'profile':
                            # Profile section has direct Q&A pairs
                            for qa_pair in person_info[section]:
                                if 'Question' in qa_pair and 'Answer' in qa_pair:
                                    question = qa_pair['Question']
                                    original_answer = qa_pair['Answer']
                                    
                                    print(f"    Question: {question}")
                                    print(f"    Original: {original_answer}")
                                    
                                    # Shorten the answer using OpenAI
                                    shortened_answer = self.shorten_answer(question, original_answer)
                                    qa_pair['Answer'] = shortened_answer
                                    
                                    print(f"    Shortened: {shortened_answer}")
                                    print()
                                    
                                    processed_questions += 1
                                    print(f"Progress: {processed_questions}/{total_questions} ({processed_questions/total_questions*100:.1f}%)")
                                    
                                    # Add a small delay to avoid rate limiting
                                    time.sleep(0.1)
                        else:
                            # Other sections have nested structure
                            for item in person_info[section]:
                                for key, qa_list in item.items():
                                    for qa_pair in qa_list:
                                        if 'Question' in qa_pair and 'Answer' in qa_pair:
                                            question = qa_pair['Question']
                                            original_answer = qa_pair['Answer']
                                            
                                            print(f"    Question: {question}")
                                            print(f"    Original: {original_answer}")
                                            
                                            # Shorten the answer using OpenAI
                                            shortened_answer = self.shorten_answer(question, original_answer)
                                            qa_pair['Answer'] = shortened_answer
                                            
                                            print(f"    Shortened: {shortened_answer}")
                                            print()
                                            
                                            processed_questions += 1
                                            print(f"Progress: {processed_questions}/{total_questions} ({processed_questions/total_questions*100:.1f}%)")
                                            
                                            # Add a small delay to avoid rate limiting
                                            time.sleep(0.1)
        
        print(f"Saving shortened answers to {output_file}...")
        self.save_json_data(data, output_file)
        print("Done!")

def main():
    """Main function to run the answer shortening process."""
    input_file = "/home/mlp/work/Mem-alpha/data/perltqa_en.json"
    output_file = "/home/mlp/work/Mem-alpha/data/perltqa_en_shortened.json"
    
    shortener = AnswerShortener()
    shortener.process_all_answers(input_file, output_file)

if __name__ == "__main__":
    main()
