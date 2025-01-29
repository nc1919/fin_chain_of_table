import argparse
import json
import os
import pickle
from utils.load_data import load_tatqa_dataset
from utils.llm import ChatGPT
from utils.helper import *
from utils.evaluate import *
from utils.chain import *
from operations import *
from typing import Any, Dict, List, Tuple, Union

def main(
    dataset_path: str = "data/tatqa/converted_tatqa_dataset_dev.json",
    raw2clean_path: str = "data/tatqa/converted_tatqa_dataset_dev_raw2clean.jsonl",
    model_name: str = "gpt-4",
    result_dir: str = "results/tatqa",
    openai_api_key: str = None,
    first_n: int = 10,  # Process only 10 entries by default
    n_proc: int = 1,
    chunk_size: int = 1,
):
    dataset = load_tatqa_dataset(dataset_path, raw2clean_path, first_n=first_n)
    gpt_llm = ChatGPT(
        model_name=model_name,
        key=os.environ["OPENAI_API_KEY"] if openai_api_key is None else openai_api_key,
    )
    os.makedirs(result_dir, exist_ok=True)

    proc_samples, dynamic_chain_log_list = dynamic_chain_exec_with_cache_mp(
        dataset,
        llm=gpt_llm,
        llm_options=gpt_llm.get_model_options(
            temperature=0.0, per_example_max_decode_steps=200, per_example_top_p=1.0
        ),
        strategy="top",
        cache_dir=os.path.join(result_dir, "cache"),
        n_proc=n_proc,
        chunk_size=chunk_size,
    )
    fixed_chain = [
        (
            "simpleQuery_fewshot",
            simple_query,
            dict(use_demo=True),
            dict(
                temperature=0, per_example_max_decode_steps=200, per_example_top_p=1.0
            ),
        ),
    ]
    final_result, _ = fixed_chain_exec_mp(gpt_llm, proc_samples, fixed_chain)
    acc = tabfact_match_func_for_samples(final_result)

    # Save accuracy and results to files
    print("Accuracy:", acc)
    with open(os.path.join(result_dir, "result.txt"), "w") as result_file:
        result_file.write(f"Accuracy: {acc}\n")

    pickle.dump(
        final_result, open(os.path.join(result_dir, "final_result.pkl"), "wb")
    )
    pickle.dump(
        dynamic_chain_log_list, 
        open(os.path.join(result_dir, "dynamic_chain_log_list.pkl"), "wb")
    )

    # Save predictions to a JSON file
    predictions_path = os.path.join(result_dir, "predictions.json")
    with open(predictions_path, "w") as pred_file:
        json.dump(final_result, pred_file, indent=4)

    print(f"Predicted results saved to {predictions_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
