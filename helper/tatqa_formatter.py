import json
from tqdm import tqdm

def tatqa_to_tabfact_format(tatqa_data_path, output_path):
    """
    Converts TAT-QA dataset tables into TabFact-like format.

    Args:
        tatqa_data_path (str): Path to the TAT-QA dataset.
        output_path (str): Path to save the converted dataset.
    """
    with open(tatqa_data_path, "r") as f:
        tatqa_data = json.load(f)

    tabfact_data = []

    for entry in tqdm(tatqa_data, desc="Converting TAT-QA to TabFact format"):
        table = entry.get("table", {}).get("table", [])
        table_uid = entry.get("table", {}).get("uid", "")
        paragraphs = entry.get("paragraphs", [])
        questions = entry.get("questions", [])

        # Create a table caption from paragraphs (if available)
        table_caption = " ".join(p.get("text", "") for p in paragraphs)

        # Convert TAT-QA table into TabFact-compatible format
        tabfact_table = [
            [str(cell) for cell in row]  # Convert all cells to strings
            for row in table
        ]

        # Derive a sample statement from questions and answers (optional)
        if questions:
            first_question = questions[0]
            statement = first_question.get("question", "")
            label = 1 if "yes" in statement.lower() or "true" in statement.lower() else 0
        else:
            statement = "Placeholder statement."
            label = -1  # Unknown label if no questions

        # Create TabFact entry
        tabfact_entry = {
            "statement": statement,
            "label": label,
            "table_caption": table_caption.strip(),
            "table_text": tabfact_table,
            "table_id": table_uid,
        }

        tabfact_data.append(tabfact_entry)

    # Save the converted dataset
    with open(output_path, "w") as f:
        json.dump(tabfact_data, f, indent=2)

    print(f"Converted dataset saved to {output_path}")


# Example Usage
tatqa_data_path = "/Users/neeladmin/Desktop/University_of_Edinburgh/Computing/DISS/fin_chain_of_table-main/data/tatqa/tatqa_dataset_test_gold.json"  # Input TAT-QA dataset
output_path = "/Users/neeladmin/Desktop/University_of_Edinburgh/Computing/DISS/fin_chain_of_table-main/data/tatqa/converted_tatqa_dataset_test_gold.json"  # Output TabFact-like dataset

tatqa_to_tabfact_format(tatqa_data_path, output_path)