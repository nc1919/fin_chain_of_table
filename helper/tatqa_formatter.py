import json
import re
from tqdm import tqdm

def extract_clean_table(table):
    """
    Converts the table structure into a clean, formatted table.
    """
    return [
        [cell.strip() for cell in row]
        for row in table.get("table", [])
    ]

def convert_json_to_jsonl(input_path, output_path):
    """
    Converts the JSON input format to the JSONL output format.

    Args:
        input_path (str): Path to the input JSON file.
        output_path (str): Path to save the converted JSONL file.
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    output_data = []

    for entry in tqdm(data, desc="Processing entries"):
        # Process the table part
        table_id = entry["table"]["uid"]
        table_text = extract_clean_table(entry["table"])
        table_caption = "".join([p["text"] for p in entry.get("paragraphs", [])])

        for question in entry.get("questions", []):
            statement = question.get("question", "")
            answer = question.get("answer", [])
            derivation = question.get("derivation", "")
            answer_type = question.get("answer_type", "")
            answer_from = question.get("answer_from", "")
            scale = question.get("scale", "")

            formatted_entry = {
                "statement": statement,
                "label": 1,  # Default label for binary classification
                "table_caption": table_caption,
                "table_text": table_text,
                "table_id": table_id,
                "answer": answer,
                "derivation": derivation,
                "answer_type": answer_type,
                "answer_from": answer_from,
                "scale": scale,
            }
            output_data.append(formatted_entry)

    # Save to JSONL file
    with open(output_path, "w") as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")

    print(f"Converted JSON saved to {output_path}")

# Example Usage
input_path = "/Users/neeladmin/Desktop/University_of_Edinburgh/Computing/DISS/fin_chain_of_table-main/data/tatqa/tatqa_dataset_test_gold.json"  # Input TAT-QA dataset
output_path = "/Users/neeladmin/Desktop/University_of_Edinburgh/Computing/DISS/fin_chain_of_table-main/data/tatqa/converted_tatqa_dataset_test_gold.jsonl"  # Output TabFact-like dataset

convert_json_to_jsonl(input_path, output_path)
