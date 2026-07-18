"""Evaluate predictions against gold annotations (SemEval 2025 Task 10, Subtask 2).

Compares outputs/submission.txt with the gold subtask-2 annotations and prints
averaged sample F1 scores (pairs / narrative-only / subnarrative-only) and
macro F1 scores (narrative-only / subnarrative-only).
"""

from pathlib import Path

from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
LANG = "EN"  # "EN" or "PT"

GOLD_FILE = ROOT / "data" / "target_4_December_release" / LANG / "subtask-2-annotations.txt"
PRED_FILE = ROOT / "outputs" / "submission.txt"


def f1_for_sample(gold_set, pred_set):
    """F1 score between two label sets for a single article."""
    if not gold_set and not pred_set:
        return 1.0
    common = gold_set & pred_set
    precision = len(common) / len(pred_set) if pred_set else 0.0
    recall = len(common) / len(gold_set) if gold_set else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def read_labels(file_path):
    """Parse a tab-separated label file (article_id, narratives, subnarratives).

    Returns {article_id: (pair set, narrative set, subnarrative set)}.
    """
    data = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            article_id = parts[0].strip()
            narratives = [lab.strip() for lab in parts[1].split(";") if lab.strip()]
            subnarratives = [lab.strip() for lab in parts[2].split(";") if lab.strip()]
            data[article_id] = (set(zip(narratives, subnarratives)),
                                set(narratives), set(subnarratives))
    return data


def evaluate_files(gold_file, pred_file):
    """Compute and print all evaluation metrics; returns them as a tuple."""
    gold_data = read_labels(gold_file)
    pred_data = read_labels(pred_file)
    all_article_ids = set(gold_data) | set(pred_data)
    empty = (set(), set(), set())

    # Averaged sample F1: per-article F1, averaged over all articles.
    f1_pairs, f1_narrative, f1_subnarrative = [], [], []
    for article_id in all_article_ids:
        gold_pairs, gold_narr, gold_sub = gold_data.get(article_id, empty)
        pred_pairs, pred_narr, pred_sub = pred_data.get(article_id, empty)
        f1_pairs.append(f1_for_sample(gold_pairs, pred_pairs))
        f1_narrative.append(f1_for_sample(gold_narr, pred_narr))
        f1_subnarrative.append(f1_for_sample(gold_sub, pred_sub))

    avg_f1_pairs = sum(f1_pairs) / len(f1_pairs) if f1_pairs else 0.0
    avg_f1_narrative = sum(f1_narrative) / len(f1_narrative) if f1_narrative else 0.0
    avg_f1_subnarrative = sum(f1_subnarrative) / len(f1_subnarrative) if f1_subnarrative else 0.0

    # Macro F1: binary indicator vectors over the universal label set, per class.
    all_narrative_labels = sorted({lab for _, narr, _ in gold_data.values() for lab in narr} |
                                  {lab for _, narr, _ in pred_data.values() for lab in narr})
    all_subnarrative_labels = sorted({lab for _, _, sub in gold_data.values() for lab in sub} |
                                     {lab for _, _, sub in pred_data.values() for lab in sub})

    y_true_narr, y_pred_narr, y_true_sub, y_pred_sub = [], [], [], []
    for article_id in all_article_ids:
        _, gold_narr, gold_sub = gold_data.get(article_id, empty)
        _, pred_narr, pred_sub = pred_data.get(article_id, empty)
        y_true_narr.append([1 if lab in gold_narr else 0 for lab in all_narrative_labels])
        y_pred_narr.append([1 if lab in pred_narr else 0 for lab in all_narrative_labels])
        y_true_sub.append([1 if lab in gold_sub else 0 for lab in all_subnarrative_labels])
        y_pred_sub.append([1 if lab in pred_sub else 0 for lab in all_subnarrative_labels])

    macro_f1_narrative = f1_score(y_true_narr, y_pred_narr, average="macro", zero_division=0)
    macro_f1_subnarrative = f1_score(y_true_sub, y_pred_sub, average="macro", zero_division=0)

    print(f"Averaged sample F1 (narrative:subnarrative pairs): {avg_f1_pairs:.4f}")
    print(f"Averaged sample F1 (narrative only):               {avg_f1_narrative:.4f}")
    print(f"Averaged sample F1 (subnarrative only):            {avg_f1_subnarrative:.4f}")
    print(f"Macro F1 (narrative only):                         {macro_f1_narrative:.4f}")
    print(f"Macro F1 (subnarrative only):                      {macro_f1_subnarrative:.4f}")

    return avg_f1_pairs, avg_f1_narrative, avg_f1_subnarrative, macro_f1_narrative, macro_f1_subnarrative


if __name__ == "__main__":
    evaluate_files(GOLD_FILE, PRED_FILE)
