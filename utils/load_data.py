# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from tqdm import tqdm

def load_tabfact_dataset(
    dataset_path,
    raw2clean_path,
    tag="test",
    first_n=-1,
):
    tabfact_statement_raw2clean_dict = {}
    with open(raw2clean_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            info = json.loads(line)
            tabfact_statement_raw2clean_dict[info["statement"]] = info["cleaned_statement"]

    dataset = []
    if first_n != -1:
        all_lines = []
        for line in open(dataset_path):
            all_lines.append(line)
            if len(all_lines) >= first_n: break
    else:
        all_lines = open(dataset_path).readlines()
    for i, line in tqdm(enumerate(all_lines), total=len(all_lines), desc=f"Loading tabfact-{tag} dataset"):
        info = json.loads(line)
        info["id"] = f"{tag}-{i}"
        info["chain"] = []
        if info["statement"] in tabfact_statement_raw2clean_dict:
            info["cleaned_statement"] = tabfact_statement_raw2clean_dict[
                info["statement"]
            ]
        else:
            info["cleaned_statement"] = info["statement"]
        dataset.append(info)
    return dataset

def load_tatqa_dataset(
    dataset_path: str,
    raw2clean_path: str,
    tag: str = "test",
    first_n: int = -1
):
    """
    Load the TAT-QA dataset in a format suitable for Chain-of-Table reasoning.

    Parameters:
        dataset_path (str): Path to the TAT-QA dataset file.
        tag (str): Tag to identify the dataset (e.g., 'train', 'test').
        first_n (int): Number of samples to load. -1 for all samples.

    Returns:
        list[dict]: Processed dataset.
    """
    dataset = []
    with open(dataset_path, 'r') as f:
        data = json.load(f)  # Assuming the dataset is a JSON array

    if first_n != -1:
        data = data[:first_n]

    for i, entry in tqdm(enumerate(data), total=len(data), desc=f"Loading TAT-QA-{tag} dataset"):
        # Extract table data
        table_text = entry.get("table", {}).get("table", [])
        # Process each question as a separate sample
        for question in entry.get("questions", []):
            sample = {
                "id": f"{tag}-{i}-{question['uid']}",
                "table_text": table_text,
                "statement": question.get("question", ""),
                "cleaned_statement": question.get("question", ""),  # Optional cleaning step
                "chain": [],  # Initialize empty chain
                "answer": question.get("answer", []),
                "answer_type": question.get("answer_type", "unknown"),
                "answer_from": question.get("answer_from", "unknown"),
                "scale": question.get("scale", "")
            }
            dataset.append(sample)

    return dataset

def wrap_input_for_demo(statement, table_caption, table_text, cleaned_statement=None):
    return {
        "statement": statement,
        "table_caption": table_caption,
        "table_text": table_text,
        "cleaned_statement": cleaned_statement if cleaned_statement is not None else statement,
        "chain": [],
    }

