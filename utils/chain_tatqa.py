import copy
import os
import re
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from operations import *

# ------------------------------------------------------------------------
# Adapting the TabFact-like chain-of-table code to TAT-QA
# ------------------------------------------------------------------------

# Example placeholders for your adapted TAT-QA operations:
# from operations_tatqa import (
#     add_column_func,
#     select_row_func,
#     select_column_func,
#     group_column_func,
#     sort_column_func,
# )
#
# from utils.helper_tatqa import table2string

# ---------------------------------------
# 1. Utility & Helper Functions
# ---------------------------------------

def get_operation_name(string):
    """
    Extract the operation name from an expression like "f_xxxx(...)".
    """
    res = re.findall(r"f_(.*?)\(.*\)", string)[0]
    return res


def get_all_operation_names(string):
    """
    From a chain string "f_add_column(...) -> f_select_row(...) -> <END>",
    parse out ['add_column', 'select_row', '<END>'].
    """
    operation_names = []
    parts = string.split("->")
    for part in parts:
        part = part.strip()
        if part == "<END>":
            operation_names.append("<END>")
        else:
            res = re.findall(r"f_(.*?)\(.*\)", part)
            if res:
                operation_names.append(res[0])
    return operation_names


def get_table_info(sample, skip_op=[], first_n_op=None):
    """
    In TabFact code, this was used to build an intermediate structure
    reflecting the table state after each chain operation.

    For TAT-QA, you can incorporate paragraphs or additional metadata if needed.
    """
    # Because TAT-QA sample format differs, adapt as necessary:
    # sample["table"] is a list of rows or a structured table,
    # sample["chain"] is an empty placeholder or a previously generated chain.
    table = sample["table"]
    chain = sample.get("chain", [])

    if first_n_op is not None:
        chain = chain[:first_n_op]

    table_info = {
        "table": table,
        "act_chain": [],
        # If you need the paragraphs for some operations:
        "paragraphs": sample.get("paragraphs", []),
    }

    for operation in chain:
        operation_name = operation["operation_name"]
        act_func = get_act_func(operation_name)
        table_info = act_func(table_info, operation, skip_op=skip_op)

    return table_info


def get_act_func(name):
    """
    Map an operation name (like 'select_row') to the actual function.

    You should replace the placeholders below with your TAT-QA-specific
    operation implementations or correct references.
    """
    try:
        # If your function naming aligns with the old format, adapt as needed:
        return eval(f"{name}_act")  
    except:
        def _default_act(table_info, *args, **kwargs):
            return copy.deepcopy(table_info)
        return _default_act


# ---------------------------------------
# 2. Example Possible Next Operations
# ---------------------------------------
# Adjust these depending on TAT-QA logic.
possible_next_operation_dict = {
    "<init>": [
        "add_column",
        "select_row",
        "select_column",
        "group_column",
        "sort_column",
    ],
    "add_column": [
        "select_row",
        "select_column",
        "group_column",
        "sort_column",
        "<END>",
    ],
    "select_row": [
        "select_column",
        "group_column",
        "sort_column",
        "<END>",
    ],
    "select_column": [
        "group_column",
        "sort_column",
        "<END>",
    ],
    "group_column": [
        "sort_column",
        "<END>",
    ],
    "sort_column": [
        "<END>",
    ],
}


# ---------------------------------------
# 3. Prompt Generation and Next-Step Logic
# ---------------------------------------

def generate_prompt_for_next_step(
    sample,
    llm,
    llm_options=None,
    strategy="top",
    debug=False
):
    """
    Generate the next chain operation using the LLM (language model).
    
    - Adapt the prompt if TAT-QA needs a different style.
    - The example below reuses the style from TabFact code with minimal changes.
    """
    table_info = get_table_info(sample)
    act_chain = table_info["act_chain"]

    if debug:
        print("Act Chain: ", act_chain, flush=True)

    kept_act_chain = [x for x in act_chain if not x.startswith("skip")]
    kept_act_chain_str = " -> ".join(kept_act_chain)
    if kept_act_chain_str:
        kept_act_chain_str += " ->"

    skip_act_chain = [x for x in act_chain if x.startswith("skip")]
    skip_act_chain_op_names = []
    for op in skip_act_chain:
        op = op[len("skip ") :]
        op_name = get_operation_name(op)
        skip_act_chain_op_names.append(op_name)

    last_operation = (
        "<init>" if not kept_act_chain else get_operation_name(kept_act_chain[-1])
    )
    possible_next_operations = possible_next_operation_dict[last_operation]
    possible_next_operations = [
        x for x in possible_next_operations if x not in skip_act_chain_op_names
    ]

    if debug:
        print("Last Operation: ", last_operation, flush=True)
        print("Possible Next Operations: ", possible_next_operations, flush=True)

    # If there's only one valid operation, return it immediately.
    if len(possible_next_operations) == 1:
        log = {
            "act_chain": act_chain,
            "last_operation": last_operation,
            "possible_next_operations": possible_next_operations,
            "prompt": None,
            "response": None,
            "generate_operations": None,
            "next_operation": possible_next_operations[0],
        }
        return possible_next_operations[0], log

    # Build a prompt snippet for the TAT-QA table and question
    # Replace 'table2string' with your TAT-QA table formatting if needed.
    prompt = ""
    # Add your TAT-QA demonstration prompts or examples here if desired.

    # Convert the table to a string for prompting:
    # prompt += table2string(table_info["table"]) + "\n"

    # If you want to attach paragraphs:
    # for para_dict in sample["paragraphs"]:
    #     for uid, text in para_dict.items():
    #         prompt += f"Paragraph {uid}: {text}\n"

    # Because TAT-QA has multiple questions per sample, you might store the question
    # you are currently processing in 'sample["question"]' or something similar.
    # For demonstration:
    prompt += f"\nQuestion: {sample.get('question', 'N/A')}\n"

    # Show the user which operations are possible:
    _possible_next_operations_str = " or ".join(
        [f"f_{op}()" if op != "<END>" else op for op in possible_next_operations]
    )
    prompt += (
        f"The next operation must be one of {_possible_next_operations_str}.\n"
    )
    prompt += "Function Chain: " + kept_act_chain_str

    # Example LLM call:
    responses = llm.generate_plus_with_score(
        prompt,
        options=llm_options,
        end_str="\n\n"
    )

    # Strategy handling:
    if strategy == "top":
        response = responses[0][0]  # top-scoring response
        generate_operations = get_all_operation_names(response)
        next_operation = "<END>"
        for operation in generate_operations:
            if operation in possible_next_operations:
                next_operation = operation
                break
    else:  # e.g. 'voting'
        next_operation_conf_dict = defaultdict(float)
        for response, score in responses:
            generate_operations = get_all_operation_names(response)
            next_operation_ = None
            for operation in generate_operations:
                if operation in possible_next_operations:
                    next_operation_ = operation
                    break
            if next_operation_:
                next_operation_conf_dict[next_operation_] += np.exp(score)
        if len(next_operation_conf_dict) != 0:
            next_operation_conf_pairs = sorted(
                next_operation_conf_dict.items(), key=lambda x: x[1], reverse=True
            )
            next_operation = next_operation_conf_pairs[0][0]
        else:
            next_operation = "<END>"

    log = {
        "act_chain": act_chain,
        "last_operation": last_operation,
        "possible_next_operations": possible_next_operations,
        "prompt": prompt,
        "response": response,
        "generate_operations": generate_operations,
        "next_operation": next_operation,
    }

    return next_operation, log


# ---------------------------------------
# 4. Dynamic Chain Execution for One Q
# ---------------------------------------

def dynamic_chain_exec_one_sample(
    sample,
    llm,
    llm_options=None,
    strategy="top",
    debug=False,
    operation_parameter_dict=None,
):
    """
    Execute the chain-of-ops for a single TAT-QA question (sample).
    """
    if operation_parameter_dict is None:
        # Example: Adjust operation parameters for TAT-QA
        operation_parameter_dict = {
            "add_column": (
                "addColumn",
                add_column_func,  # your TAT-QA operation
                {},
                llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                ),
            ),
            "select_row": (
                "selectRow",
                select_row_func,  # your TAT-QA operation
                {},
                llm.get_model_options(
                    temperature=0.5,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                    n_sample=8,
                ),
            ),
            "select_column": (
                "selectColumn",
                select_column_func,  # your TAT-QA operation
                {},
                llm.get_model_options(
                    temperature=0.5,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                    n_sample=8,
                ),
            ),
            "group_column": (
                "groupColumn",
                group_column_func,  # your TAT-QA operation
                dict(skip_op=[]),
                llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                ),
            ),
            "sort_column": (
                "sortColumn",
                sort_column_func,  # your TAT-QA operation
                dict(skip_op=[]),
                llm.get_model_options(
                    temperature=0.0,
                    per_example_max_decode_steps=150,
                    per_example_top_p=1.0,
                ),
            ),
        }

    dynamic_chain_log = []
    current_sample = copy.deepcopy(sample)

    while True:
        # Generate next operation
        next_operation, log = generate_prompt_for_next_step(
            current_sample,
            llm=llm,
            llm_options=llm_options,
            strategy=strategy,
            debug=debug
        )
        dynamic_chain_log.append(log)

        if debug:
            print("Next operation:", next_operation)

        if next_operation == "<END>":
            break

        # Execute the chosen operation
        op_name, solver_func, kargs, op_llm_options = operation_parameter_dict[next_operation]
        table_info = get_table_info(current_sample)
        current_sample = solver_func(
            current_sample,
            table_info,
            llm=llm,
            llm_options=op_llm_options,
            **kargs
        )

    return current_sample, dynamic_chain_log


# ---------------------------------------
# 5. Batch Execution (Single Process)
# ---------------------------------------

def dynamic_chain_exec_for_tatqa_questions(
    dataset,
    llm,
    llm_options=None,
    strategy="voting",
    cache_dir="./cache_tatqa",
):
    """
    Iterate over the TAT-QA dataset, and for each table/question, run dynamic chain execution.
    This version uses a simple for-loop, storing results in a local structure.
    If you want caching (like TabFact code), adapt accordingly.
    """
    os.makedirs(cache_dir, exist_ok=True)

    result_samples = []
    dynamic_chain_log_list = []

    for data_entry in tqdm(dataset, desc="Processing TAT-QA dataset"):
        table_uid = data_entry["table_uid"]
        table = data_entry["table"]
        paragraphs = data_entry["paragraphs"]
        # TAT-QA often has multiple questions per table:
        for q in data_entry["questions"]:
            # Construct a single sample for chain execution
            sample_id = q["uid"]
            sample = {
                "id": sample_id,
                "table_uid": table_uid,
                "table": table,
                "paragraphs": paragraphs,
                "question": q["question"],
                "answer": q.get("answer", []),
                "answer_type": q.get("answer_type", ""),
                "chain": [],  # start with empty chain
            }
            cache_path = os.path.join(cache_dir, f"case-{sample_id}.pkl")
            if os.path.exists(cache_path):
                # Load from cache
                stored_data = pickle.load(open(cache_path, "rb"))
                proc_sample = stored_data["proc_sample"]
                log = stored_data["log"]
            else:
                # Perform chain execution
                proc_sample, log = dynamic_chain_exec_one_sample(
                    sample,
                    llm=llm,
                    llm_options=llm_options,
                    strategy=strategy
                )
                # Save to cache
                pickle.dump(
                    {"proc_sample": proc_sample, "log": log},
                    open(cache_path, "wb")
                )
            result_samples.append(proc_sample)
            dynamic_chain_log_list.append(log)

    return result_samples, dynamic_chain_log_list


# ---------------------------------------
# 6. Batch Execution (Multi-Process)
# ---------------------------------------

def _dynamic_chain_exec_tatqa_mp_core(arg):
    """
    Helper function for parallel (multiprocessing) execution.
    """
    (data_entry, question, llm, llm_options, strategy, cache_dir) = arg
    try:
        # Build sample
        sample_id = question["uid"]
        sample = {
            "id": sample_id,
            "table_uid": data_entry["table_uid"],
            "table": data_entry["table"],
            "paragraphs": data_entry["paragraphs"],
            "question": question["question"],
            "answer": question.get("answer", []),
            "answer_type": question.get("answer_type", ""),
            "chain": [],
        }
        cache_path = os.path.join(cache_dir, f"case-{sample_id}.pkl")
        if os.path.exists(cache_path):
            stored_data = pickle.load(open(cache_path, "rb"))
            proc_sample = stored_data["proc_sample"]
            log = stored_data["log"]
        else:
            proc_sample, log = dynamic_chain_exec_one_sample(
                sample,
                llm=llm,
                llm_options=llm_options,
                strategy=strategy
            )
            pickle.dump({"proc_sample": proc_sample, "log": log}, open(cache_path, "wb"))
        return sample_id, proc_sample, log
    except Exception as e:
        print(f"Error in question {question['uid']}: {e}", flush=True)
        return question["uid"], None, None


def dynamic_chain_exec_for_tatqa_mp(
    dataset,
    llm,
    llm_options=None,
    strategy="voting",
    cache_dir="./results_tatqa",
    n_proc=10,
    chunk_size=50,
):
    """
    Multi-processing version for TAT-QA chain-of-ops.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Prepare tasks
    tasks = []
    for data_entry in dataset:
        for q in data_entry["questions"]:
            tasks.append((data_entry, q, llm, llm_options, strategy, cache_dir))

    result_samples = {}
    dynamic_chain_log_list = {}

    with mp.Pool(n_proc) as p:
        for sample_id, proc_sample, log in tqdm(
            p.imap_unordered(_dynamic_chain_exec_tatqa_mp_core, tasks, chunksize=chunk_size),
            total=len(tasks),
            desc="Processing TAT-QA Multiprocess"
        ):
            result_samples[sample_id] = proc_sample
            dynamic_chain_log_list[sample_id] = log

    return result_samples, dynamic_chain_log_list