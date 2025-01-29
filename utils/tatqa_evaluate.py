#!/usr/bin/python
import argparse
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union, List, Set
from pathlib import Path
from tqdm import tqdm
from scipy import stats

# -----------------------------
# 1) TAT-QA Utility Functions
# -----------------------------

def is_number(s: str) -> bool:
    """Check if string can be cast to a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def to_number(s: str):
    """Convert string to float or return None if not possible."""
    try:
        return float(s)
    except ValueError:
        return None

def normalize_answer(s: str) -> str:
    """
    Lower text, remove punctuation and articles, and trim extra whitespace.
    This normalization aids in partial matches for text-based answers.
    """
    import re
    import string

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    text = s.lower()
    text = remove_punc(text)
    text = remove_articles(text)
    text = " ".join(text.split())
    return text

def scale_to_num(scale: str) -> float:
    """Convert a scale string to a numerical multiplier."""
    if scale == "thousand":
        return 1e3
    elif scale == "million":
        return 1e6
    elif scale == "billion":
        return 1e9
    elif scale == "percent":
        # 'percent' can be treated as a multiplier of 1 if returning the raw fraction.
        return 1.0
    return 1.0

def extract_all_nums_from_str(s: str) -> List[float]:
    """
    Extract all numeric values from a string (including negatives and decimals).
    """
    import re
    nums = []
    for x in re.findall(r"-?\d+\.?\d*", s):
        try:
            nums.append(float(x))
        except:
            pass
    return nums

# -----------------------------
# 2) TAT-QA Metric Functions
# -----------------------------

def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    """
    Token-level F1 between two sets of tokens (predicted vs. gold).
    """
    intersection = len(gold_bag.intersection(predicted_bag))

    precision = 1.0 if not predicted_bag else intersection / float(len(predicted_bag))
    recall = 1.0 if not gold_bag else intersection / float(len(gold_bag))

    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    """
    For each span in the answer, normalize and convert to a set of tokens.
    Returns a tuple (list_of_normalized_strings, list_of_token_sets).
    """
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]

    normalized_spans: List[str] = []
    token_bags: List[Set[str]] = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags

def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Finds an optimal one-to-one alignment (via Hungarian algorithm) between 
    predicted bags and gold bags, returning the best F1 scores.
    """
    from scipy.optimize import linear_sum_assignment
    scores = np.zeros([len(gold), len(predicted)])
    for g_i, gold_item in enumerate(gold):
        for p_i, pred_item in enumerate(predicted):
            scores[g_i, p_i] = _compute_f1(pred_item, gold_item)

    # Negate scores to convert to a minimization problem for linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-scores)
    max_scores = np.zeros([max(len(gold), len(predicted))])

    for row, column in zip(row_ind, col_ind):
        # We take the maximum of any duplicates (unlikely but possible)
        max_scores[row] = max(max_scores[row], scores[row, column])

    return max_scores

def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    Computes (exact_match, f1_score) between predicted and gold answers.
    - exact_match: 1 if the sets of tokens match exactly for the first span, else 0
    - f1_score: token-level F1 across potential multiple spans
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    # Check exact match on the first normalized strings.
    # For single-span: set equality + same number of tokens => exact_match = 1.0
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    # Compute F1 via the Hungarian alignment of sets.
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1_score = float(np.mean(f1_per_bag))
    f1_score = round(f1_score, 2)

    return exact_match, f1_score

def extract_gold_answers(qa_annotation: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """
    Parse the gold question annotation to retrieve:
    - answer_type (span, multi-span, arithmetic, count, etc.)
    - gold_answers (list of correct answers)
    - scale (e.g., 'million', 'percent', etc.)
    """
    answer_type = qa_annotation.get('answer_type', "")
    scale = qa_annotation.get('scale', "")
    answer_content = qa_annotation.get('answer', "")
    gold_answers = []

    # Different TAT-QA answer types
    if answer_type in ['multi-span', 'span']:
        assert isinstance(answer_content, list), answer_content
        gold_answers = answer_content
    elif answer_type in ["arithmetic"]:
        gold_answers.append(str(answer_content))
    elif answer_type in ['count']:
        gold_answers.append(str(int(answer_content)))
    else:
        # fallback: treat as single textual answer
        gold_answers.append(str(answer_content))

    return answer_type, gold_answers, scale

def get_answer_str(answers: list, scale: str) -> List[str]:
    """
    Convert a list of gold answers plus an optional scale into normalized strings.
    If answers are numeric, multiply by scale_to_num(scale); if they're textual, just append the scale text (optional).
    """
    try:
        sorted_ans = sorted(answers, key=lambda x: str(x).lower())
        ans_temp = []
        for ans in sorted_ans:
            ans_str = str(ans).lower()
            if is_number(ans_str):
                ans_num = to_number(ans_str)
                if ans_num is None:
                    # Could not convert to float; treat as text
                    if scale:
                        ans_str = ans_str + " " + str(scale)
                else:
                    # If there's a '%' in the string, treat it as is, else apply numeric scale
                    if '%' in ans_str:
                        ans_str = f"{ans_num:.4f}"
                    else:
                        ans_str = f"{(round(ans_num, 2) * scale_to_num(scale)):.4f}"
            else:
                # If text, optionally append scale
                if scale:
                    ans_str = ans_str + " " + str(scale)
            ans_temp.append(ans_str)
        return [" ".join(ans_temp)]
    except Exception as e:
        print(f'get_answer_str error: {e}')
    return [""]

def metric_max_over_ground_truths(metric_fn,
                                  prediction_strings: List[str],
                                  ground_truth_strings: List[str]) -> Tuple[float, float]:
    """
    Compute maximum (EM, F1) among multiple ground-truth references.
    """
    scores_for_ground_truths = []
    for pred in prediction_strings:
        for gold in ground_truth_strings:
            score_em, score_f1 = metric_fn(pred, gold)
            scores_for_ground_truths.append((score_em, score_f1))
    if not scores_for_ground_truths:
        return 0.0, 0.0
    best_em = max(s[0] for s in scores_for_ground_truths)
    best_f1 = max(s[1] for s in scores_for_ground_truths)
    return best_em, best_f1

class TATEmAndF1:
    """
    Accumulates average Exact Match, F1, and scale correctness for TAT-QA.
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._scale_em = 0.0
        self._count = 0

    def __call__(self, ground_truth: dict,
                 prediction: Union[str, List, float, int, None],
                 pred_scale: str = ""):
        """
        Evaluate a single question's ground-truth vs. a predicted answer.
        """
        self._count += 1
        if not prediction:
            exact_match = 0.0
            f1_score = 0.0
        else:
            # Extract gold data
            gold_type, gold_answers, gold_scale = extract_gold_answers(ground_truth)
            ground_truth_answer_strings = get_answer_str(gold_answers, gold_scale)

            # Convert prediction to list of strings
            if not isinstance(prediction, list):
                prediction_list = [prediction]
            else:
                prediction_list = prediction
            prediction_strings = get_answer_str(prediction_list, pred_scale)

            exact_match, f1_score = metric_max_over_ground_truths(
                get_metrics,
                prediction_strings,
                ground_truth_answer_strings
            )

        true_scale = ground_truth.get("scale", "")
        scale_correct = 1.0 if pred_scale == true_scale else 0.0

        self._total_em += exact_match
        self._total_f1 += f1_score
        self._scale_em += scale_correct

    def get_overall_metric(self) -> Tuple[float, float, float]:
        """
        Returns the average EM, F1, and scale correctness across all examples so far.
        """
        if self._count == 0:
            return 0.0, 0.0, 0.0
        em = self._total_em / self._count
        f1 = self._total_f1 / self._count
        scale_score = self._scale_em / self._count
        return em, f1, scale_score

# -----------------------------
# 3) Main Evaluation Logic
# -----------------------------

def parse_pred_answer(qa: Dict[str, Any], llm_response_text: str) -> Tuple[List[str], str]:
    """
    An example parser that extracts a final predicted answer and scale from a single response string.
    Adjust as needed for your chain-of-table or model output format.
    """
    pred_answer = [llm_response_text.strip()]
    pred_scale = ""
    return pred_answer, pred_scale

def evaluate_json(golden_answers: List[Dict[str, Any]], llm_predictions: Dict[str, Any]) -> None:
    """
    Evaluate each TAT-QA question in `golden_answers` against predictions
    in `llm_predictions` (uid -> predicted answer).
    """
    em_and_f1 = TATEmAndF1()

    # Iterate through each TAT-QA entry (each can have multiple questions)
    for entry in tqdm(golden_answers, desc="Evaluating"):
        questions = entry.get("questions", [])
        for qa in questions:
            query_id = qa["uid"]
            pred_answer, pred_scale = None, None

            if query_id in llm_predictions:
                raw_prediction = llm_predictions[query_id]
                if isinstance(raw_prediction, str):
                    # Single string
                    pred_answer, pred_scale = parse_pred_answer(qa, raw_prediction)
                elif isinstance(raw_prediction, list) and len(raw_prediction) > 0:
                    # Multiple attempts, just pick the first or implement your own logic
                    pred_answer, pred_scale = parse_pred_answer(qa, raw_prediction[0])
                else:
                    # Unexpected format
                    pred_answer, pred_scale = [str(raw_prediction)], ""

            # Evaluate the question
            em_and_f1(ground_truth=qa, prediction=pred_answer, pred_scale=pred_scale)

    # Output final metrics
    global_em, global_f1, global_scale = em_and_f1.get_overall_metric()
    print("----")
    print(f"Exact-match accuracy: {global_em * 100:.2f}")
    print(f"F1 score: {global_f1 * 100:.2f}")
    print(f"Scale score: {global_scale * 100:.2f}")
    print(f"{global_em * 100:.2f}   &   {global_f1 * 100:.2f}")
    print("----")

def evaluate_prediction_file(gold_path: str, pred_path: str) -> None:
    """
    Loads TAT-QA gold data and your predictions, then runs the evaluation.
    """
    # 1) Load gold data
    with open(gold_path, "r", encoding='utf-8') as f:
        golden_answers = json.load(f)

    # 2) Load predictions in the expected format:
    #    [
    #      {"id": "question_uid1", "prediction": "some text or list of text"},
    #      {"id": "question_uid2", "prediction": ["attempt1", "attempt2"]},
    #      ...
    #    ]
    with open(pred_path, "r", encoding='utf-8') as f:
        predictions_list = json.load(f)

    # Convert into dict: uid -> prediction
    llm_predictions = {item["id"]: item["prediction"] for item in predictions_list}

    # 3) Evaluate
    evaluate_json(golden_answers, llm_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TAT-QA Evaluation Script')
    parser.add_argument("--gold_path",
                        type=str,
                        required=True,
                        help='Path to the TAT-QA gold file (JSON).')
    parser.add_argument("--pred_path",
                        type=str,
                        required=True,
                        help='Path to the prediction file (JSON).')
    args = parser.parse_args()

    evaluate_prediction_file(args.gold_path, args.pred_path)