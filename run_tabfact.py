import fire
import os
import pickle
from utils.load_data import load_tabfact_dataset
from utils.llm import GeminiAPI
from utils.helper import *
from utils.evaluate import *
from utils.chain import *
from operations import *


def main(
    dataset_path: str = "data/tabfact/test.jsonl",
    raw2clean_path: str ="data/tabfact/raw2clean.jsonl",
    model_name: str = "gemini-1.5-flan",  # Update this as per your model name in Gemini
    result_dir: str = "results/tabfact",
    api_key: str = "",  # Updated the variable name to match Gemini's API key
    first_n=-1,
    n_proc=1,
    chunk_size=1,
):
    dataset = load_tabfact_dataset(dataset_path, raw2clean_path, first_n=first_n)
    
    # Initialize the GeminiAPI with model_name and api_key
    gpt_llm = GeminiAPI(
        model_name=model_name,
        key=os.environ["Gemini"] if api_key is None else api_key,  # Adapt to handle Gemini key
    )

    os.makedirs(result_dir, exist_ok=True)
 
    # Process the dataset with dynamic chain execution
    proc_samples, dynamic_chain_log_list = dynamic_chain_exec_with_cache_mp(
        dataset,
        llm=gpt_llm,
        llm_options=gpt_llm.get_model_options(
            temperature=0.0, per_example_max_decode_steps=200, n_sample=1  # Adjusted option keys
        ),
        strategy="top",
        cache_dir=os.path.join(result_dir, "cache"),
        n_proc=n_proc,
        chunk_size=chunk_size,
    )
    
    # Define a simple chain using Gemini LLM
    fixed_chain = [
        (
            "simpleQuery_fewshot",
            simple_query,
            dict(use_demo=True),
            dict(
                temperature=0, per_example_max_decode_steps=200, n_sample=1  # Adjusted for Gemini options
            ),
        ),
    ]

    # Execute the fixed chain
    final_result, _ = fixed_chain_exec_mp(gpt_llm, proc_samples, fixed_chain)

    # Evaluate accuracy with TabFact specific function
    acc = tabfact_match_func_for_samples(final_result)
    print("Accuracy:", acc)

    # Save results
    with open(os.path.join(result_dir, "result.txt"), "w") as f:
        f.write(f'Accuracy: {acc}')

    # Save final results and logs using pickle
    pickle.dump(final_result, open(os.path.join(result_dir, "final_result.pkl"), "wb"))
    pickle.dump(dynamic_chain_log_list, open(os.path.join(result_dir, "dynamic_chain_log_list.pkl"), "wb"))


if __name__ == "__main__":
    fire.Fire(main)
