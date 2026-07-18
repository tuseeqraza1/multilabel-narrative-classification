"""Run inference with the trained model and write a submission file.

Loads the model and label mappings from models/final_model/, predicts narrative
and subnarrative labels for every article in the target release, enforces
hierarchical consistency, and writes outputs/submission.txt.
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from training import MultiLabelClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
LANG = "EN"  # "EN" or "PT"

ARTICLES_DIR = ROOT / "data" / "target_4_December_release" / LANG / "raw-documents"
MODEL_DIR = ROOT / "models" / "final_model"
OUTPUT_FILE = ROOT / "outputs" / "submission.txt"

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 512

# Label selection thresholds (tuned on the dev split)
PRIMARY_THRESHOLD_NARRATIVE = 0.60
PRIMARY_THRESHOLD_SUBNARRATIVE = 0.75
FALLBACK_THRESHOLD = 0.40


def load_mappings(model_dir):
    """Load the id -> label mappings saved during training."""
    with open(model_dir / "narrative_mapping.json", encoding="utf-8") as f:
        narrative_label2id = json.load(f)
    with open(model_dir / "subnarrative_mapping.json", encoding="utf-8") as f:
        subnarrative_label2id = json.load(f)
    narrative_id2label = {int(v): k for k, v in narrative_label2id.items()}
    subnarrative_id2label = {int(v): k for k, v in subnarrative_label2id.items()}
    return narrative_id2label, subnarrative_id2label


def load_model(model_dir, device="cpu"):
    """Rebuild the model architecture and load the trained weights."""
    narrative_id2label, subnarrative_id2label = load_mappings(model_dir)
    model = MultiLabelClassifier(MODEL_NAME, len(narrative_id2label), len(subnarrative_id2label))

    safetensors_path = model_dir / "model.safetensors"
    bin_path = model_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location=device)
    else:
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    model.load_state_dict(state_dict)
    return model.to(device), narrative_id2label, subnarrative_id2label


def select_labels(probabilities, primary_threshold, fallback_threshold):
    """Pick labels above the primary threshold; if none qualify, force the top
    prediction and optionally a second one above the fallback threshold."""
    preds = (probabilities > primary_threshold).int().tolist()
    if sum(preds) == 0:
        sorted_indices = torch.argsort(probabilities, descending=True)
        preds[sorted_indices[0]] = 1
        if len(probabilities) > 1 and probabilities[sorted_indices[1]].item() > fallback_threshold:
            preds[sorted_indices[1]] = 1
    return preds


def enforce_hierarchical_consistency(narrative_labels, subnarrative_labels):
    """Subtask-2 convention: no (or only 'Other') narrative -> subnarrative is
    ['Other']; every other predicted narrative must have at least one matching
    subnarrative, otherwise 'Narrative: Other' is appended."""
    if not narrative_labels or narrative_labels == ["Other"]:
        return ["Other"]

    result = subnarrative_labels.copy()
    for narrative in narrative_labels:
        if narrative == "Other":
            continue
        prefix = narrative + ":"
        if not any(sub.startswith(prefix) for sub in result):
            result.append(narrative + ": Other")
    return result


def predict_article(text, model, tokenizer, narrative_id2label, subnarrative_id2label, device):
    """Predict narrative and subnarrative labels for one article text."""
    encoding = tokenizer(text, truncation=True, padding="max_length",
                         max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"].to(device),
                        attention_mask=encoding["attention_mask"].to(device))

    narrative_probs = torch.sigmoid(outputs["narrative_logits"].squeeze())
    subnarrative_probs = torch.sigmoid(outputs["subnarrative_logits"].squeeze())

    narrative_preds = select_labels(narrative_probs, PRIMARY_THRESHOLD_NARRATIVE, FALLBACK_THRESHOLD)
    subnarrative_preds = select_labels(subnarrative_probs, PRIMARY_THRESHOLD_SUBNARRATIVE, FALLBACK_THRESHOLD)

    narrative_labels = [narrative_id2label[i] for i, p in enumerate(narrative_preds) if p]
    subnarrative_labels = [subnarrative_id2label[i] for i, p in enumerate(subnarrative_preds) if p]
    subnarrative_labels = enforce_hierarchical_consistency(narrative_labels, subnarrative_labels)
    return narrative_labels, subnarrative_labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model, narrative_id2label, subnarrative_id2label = load_model(MODEL_DIR, device=device)
    model.eval()

    files = sorted(ARTICLES_DIR.glob("*.txt"))
    print(f"Running inference on {len(files)} articles from {ARTICLES_DIR}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for path in files:
            text = path.read_text(encoding="utf-8")
            narratives, subnarratives = predict_article(
                text, model, tokenizer, narrative_id2label, subnarrative_id2label, device)
            out.write(f"{path.name}\t{';'.join(narratives)}\t{';'.join(subnarratives)}\n")

    print(f"Submission file saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
