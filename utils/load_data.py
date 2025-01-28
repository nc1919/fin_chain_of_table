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
import re

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

def load_tatqa_dataset(dataset_path, tag="test", first_n=-1):
    with open(dataset_path, "r") as f:
        data = json.load(f)  # <-- Load entire JSON as a list

    # If first_n is specified, limit the number of entries
    if first_n != -1:
        data = data[:first_n]

    dataset = []
    for i, info in tqdm(enumerate(data), total=len(data), desc=f"Loading TAT-QA-{tag} dataset"):
        info["id"] = f"{tag}-{i}"
        info["chain"] = []  # Placeholder for reasoning chain if needed

        dataset_entry = {
            "id": info["id"],
            "table_uid": info["table"].get("uid", ""),
            "table": info["table"].get("table", []),
            "paragraphs": [{p["uid"]: p["text"]} for p in info["paragraphs"]],
            "questions": [
                {
                    "uid": q["uid"],
                    "question": q["question"],
                    "answer": q.get("answer", []),
                    "answer_type": q.get("answer_type", ""),
                    "answer_from": q.get("answer_from", ""),
                    "derivation": q.get("derivation", ""),
                    "rel_paragraphs": q.get("rel_paragraphs", []),
                    "req_comparison": q.get("req_comparison", False),
                    "scale": q.get("scale", "")
                }
                for q in info["questions"]
            ]
        }
        dataset.append(dataset_entry)

    return dataset

def wrap_input_for_demo(statement, table_caption, table_text, cleaned_statement=None):
    return {
        "statement": statement,
        "table_caption": table_caption,
        "table_text": table_text,
        "cleaned_statement": cleaned_statement if cleaned_statement is not None else statement,
        "chain": [],
    }

def wrap_input_for_tatqa_demo(question, answer, table_text, paragraph):
    return {
        "question": question,
        "answer": answer,
        "table_text": table_text,
        "paragraph": paragraph,
        "chain": []
    }
    
