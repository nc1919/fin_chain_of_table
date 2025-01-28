import json
from tqdm import tqdm
import re


def clean_question(question):
    """
    Cleans a raw question by applying basic cleaning rules.
    Modify this function as needed for custom cleaning logic.
    """
    # Example cleaning rules
    question = question.strip()  # Remove leading/trailing spaces
    question = re.sub(r"\s+", " ", question)  # Replace multiple spaces with a single space
    question = question.lower()  # Convert to lowercase (optional)
    question = question.replace("?", "")  # Remove question marks (optional)
    return question


def create_tatqa_raw2clean_file(dataset_path, output_path):
    """
    Generates a raw-to-clean mapping file for the TAT-QA dataset.

    Args:
        dataset_path (str): Path to the TAT-QA dataset (JSON format).
        output_path (str): Path to save the raw-to-clean mapping file.
    """
    with open(dataset_path, "r") as f:
        data = json.load(f)

    raw2clean_mapping = []

    # Iterate through all entries and clean each question
    for entry in tqdm(data, desc="Creating raw-to-clean mapping"):
        for question_obj in entry.get("questions", []):
            raw_question = question_obj["question"]
            cleaned_question = clean_question(raw_question)

            # Add the mapping to the list
            raw2clean_mapping.append({
                "raw_question": raw_question,
                "cleaned_question": cleaned_question
            })

    # Save the raw-to-clean mappings to a JSON lines file
    with open(output_path, "w") as f:
        for mapping in raw2clean_mapping:
            f.write(json.dumps(mapping) + "\n")

    print(f"Raw-to-clean file created at: {output_path}")


# Example usage
dataset_path = "path/to/tatqa_dataset.json"  # Path to the TAT-QA dataset
output_path = "path/to/tatqa_raw2clean.jsonl"  # Output raw-to-clean mapping file

create_tatqa_raw2clean_file(dataset_path, output_path)