"""Train the hierarchical multi-label narrative classifier (SemEval 2025 Task 10, Subtask 2).

Reads the subtask-2 annotations and article texts, fine-tunes a multilingual BERT
model with two classification heads (narrative + subnarrative), and saves the
final model, tokenizer and label mappings to models/final_model/.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
LANG = "EN"  # "EN" or "PT"

TRAIN_DIR = ROOT / "data" / "training_data_16_October_release" / LANG
ANNOTATIONS_FILE = TRAIN_DIR / "subtask-2-annotations.txt"
ARTICLES_DIR = TRAIN_DIR / "raw-documents"
MODEL_DIR = ROOT / "models" / "final_model"
OUTPUT_DIR = ROOT / "outputs" / "output"  # trainer checkpoints/logs

MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 512
NUM_EPOCHS = 30


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------
def split_labels(entry):
    """Split a ';'-separated label string into a clean list."""
    return [lab.strip() for lab in str(entry).split(";") if lab.strip()]


def compute_label_mapping(df, column):
    """Map every distinct label in `column` to an integer id (sorted for determinism)."""
    labels = sorted({lab for entry in df[column] for lab in split_labels(entry)})
    return {label: idx for idx, label in enumerate(labels)}


def compute_pos_weights(df, column, label2id):
    """Per-label positive weights ((N - count) / count) for the imbalanced BCE/focal loss."""
    counts = np.zeros(len(label2id))
    for entry in df[column]:
        for lab in split_labels(entry):
            if lab in label2id:
                counts[label2id[lab]] += 1
    n = len(df)
    weights = [(n - c) / c if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)


def compute_sample_weights(df, column, label2id):
    """Per-sample weights (inverse label frequency) for oversampling rare classes."""
    counts = np.zeros(len(label2id))
    for entry in df[column]:
        for lab in split_labels(entry):
            if lab in label2id:
                counts[label2id[lab]] += 1
    sample_weights = []
    for entry in df[column]:
        labs = split_labels(entry)
        weights = [1.0 / counts[label2id[lab]] for lab in labs if counts[label2id[lab]] > 0]
        sample_weights.append(np.mean(weights) if weights else 1.0)
    return sample_weights


def augment_text(text, drop_prob=0.05):
    """Randomly drop words to make the model more robust to noise."""
    words = text.split()
    kept = [w for w in words if random.random() > drop_prob]
    return " ".join(kept) if kept else text


# ---------------------------------------------------------------------------
# Focal loss (down-weights easy examples, focuses on hard ones)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        loss = (1 - pt) ** self.gamma * bce
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class NarrativeDataset(Dataset):
    """Reads an article file per row and builds multi-hot label vectors."""

    def __init__(self, dataframe, articles_dir, tokenizer, max_length,
                 narrative_label2id, subnarrative_label2id, augment=False):
        self.data = dataframe.reset_index(drop=True)
        self.articles_dir = Path(articles_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.narrative_label2id = narrative_label2id
        self.subnarrative_label2id = subnarrative_label2id
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = (self.articles_dir / row["article_id"]).read_text(encoding="utf-8")
        if self.augment:
            text = augment_text(text)

        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )

        narrative_vector = torch.zeros(len(self.narrative_label2id))
        for lab in split_labels(row["narrative"]):
            if lab in self.narrative_label2id:
                narrative_vector[self.narrative_label2id[lab]] = 1.0

        subnarrative_vector = torch.zeros(len(self.subnarrative_label2id))
        for lab in split_labels(row["subnarrative"]):
            if lab in self.subnarrative_label2id:
                subnarrative_vector[self.subnarrative_label2id[lab]] = 1.0

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "narrative_labels": narrative_vector,
            "subnarrative_labels": subnarrative_vector,
        }


def collate_fn(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


# ---------------------------------------------------------------------------
# Model: BERT encoder + two heads with hierarchical conditioning
# ---------------------------------------------------------------------------
class MultiLabelClassifier(nn.Module):
    """The subnarrative head sees the narrative logits, so fine-grained predictions
    are conditioned on the coarse-level prediction (hierarchical conditioning)."""

    def __init__(self, model_name, num_narrative_labels, num_subnarrative_labels,
                 pos_weight_narrative=None, pos_weight_subnarrative=None,
                 use_focal_loss=False, gamma=2.0, alpha=0.5,
                 hierarchical_mapping=None, narrative_id2label=None,
                 consistency_weight=0.5, subnarrative_loss_weight=1.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)

        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.alpha = alpha
        self.pos_weight_narrative = pos_weight_narrative
        self.pos_weight_subnarrative = pos_weight_subnarrative

        # narrative id -> list of subnarrative ids belonging to it (for consistency loss)
        self.hierarchical_mapping = hierarchical_mapping
        self.narrative_id2label = narrative_id2label
        self.consistency_weight = consistency_weight
        self.subnarrative_loss_weight = subnarrative_loss_weight

        self.narrative_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, num_narrative_labels),
        )
        self.subnarrative_head = nn.Sequential(
            nn.Linear(hidden_size + num_narrative_labels, hidden_size), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, num_subnarrative_labels),
        )

    def _loss_fn(self, pos_weight, device):
        pos_weight = pos_weight.to(device) if pos_weight is not None else None
        if self.use_focal_loss:
            return FocalLoss(gamma=self.gamma, alpha=self.alpha, pos_weight=pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, input_ids, attention_mask, narrative_labels=None, subnarrative_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None \
            else outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        narrative_logits = self.narrative_head(pooled)
        subnarrative_logits = self.subnarrative_head(torch.cat([pooled, narrative_logits], dim=1))

        loss = None
        if narrative_labels is not None and subnarrative_labels is not None:
            device = narrative_logits.device
            narrative_labels = narrative_labels.to(device)
            subnarrative_labels = subnarrative_labels.to(device)

            narrative_loss = self._loss_fn(self.pos_weight_narrative, device)(
                narrative_logits, narrative_labels)
            subnarrative_loss = self._loss_fn(self.pos_weight_subnarrative, device)(
                subnarrative_logits, subnarrative_labels)
            loss = narrative_loss + self.subnarrative_loss_weight * subnarrative_loss

            # Consistency loss: if a narrative is active, at least one of its
            # subnarratives should get a high probability.
            if self.hierarchical_mapping and self.narrative_id2label:
                sub_probs = torch.sigmoid(subnarrative_logits)
                consistency_terms = []
                for sample_idx in range(narrative_labels.shape[0]):
                    for narr_idx in range(narrative_labels.shape[1]):
                        if narrative_labels[sample_idx, narr_idx] != 1:
                            continue
                        if self.narrative_id2label.get(narr_idx, "").lower() == "other":
                            continue
                        group = self.hierarchical_mapping.get(narr_idx, [])
                        if group:
                            consistency_terms.append(1 - torch.max(sub_probs[sample_idx, group]))
                if consistency_terms:
                    loss = loss + self.consistency_weight * torch.stack(consistency_terms).mean()

        return {
            "loss": loss,
            "narrative_logits": narrative_logits,
            "subnarrative_logits": subnarrative_logits,
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred, threshold=0.7):
    """Sample and macro F1 for both label levels, thresholding sigmoid probabilities."""
    logits_narrative = eval_pred.predictions["narrative_logits"]
    logits_subnarrative = eval_pred.predictions["subnarrative_logits"]
    labels_narrative = np.array(eval_pred.label_ids["narrative_labels"])
    labels_subnarrative = np.array(eval_pred.label_ids["subnarrative_labels"])

    preds_narrative = (torch.sigmoid(torch.tensor(logits_narrative)) > threshold).int().numpy()
    preds_subnarrative = (torch.sigmoid(torch.tensor(logits_subnarrative)) > threshold).int().numpy()

    return {
        "f1_narrative": f1_score(labels_narrative, preds_narrative, average="samples", zero_division=0),
        "f1_subnarrative": f1_score(labels_subnarrative, preds_subnarrative, average="samples", zero_division=0),
        "macro_f1_narrative": f1_score(labels_narrative, preds_narrative, average="macro", zero_division=0),
        "macro_f1_subnarrative": f1_score(labels_subnarrative, preds_subnarrative, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Trainer with oversampling of rare classes
# ---------------------------------------------------------------------------
class CustomTrainerWithSampler(Trainer):
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights

    def get_train_dataloader(self):
        if self.sample_weights is None:
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(
            self.sample_weights, num_samples=len(self.sample_weights), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
        logits = {
            "narrative_logits": outputs["narrative_logits"].detach().cpu(),
            "subnarrative_logits": outputs["subnarrative_logits"].detach().cpu(),
        }
        labels = {
            "narrative_labels": inputs["narrative_labels"],
            "subnarrative_labels": inputs["subnarrative_labels"],
        }
        return outputs["loss"], logits, labels


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(ANNOTATIONS_FILE, sep="\t", header=None,
                     names=["article_id", "narrative", "subnarrative"])
    print(f"Loaded {len(df)} annotated articles from {ANNOTATIONS_FILE}")

    narrative_label2id = compute_label_mapping(df, "narrative")
    subnarrative_label2id = compute_label_mapping(df, "subnarrative")
    narrative_id2label = {v: k for k, v in narrative_label2id.items()}
    print(f"{len(narrative_label2id)} narrative labels, {len(subnarrative_label2id)} subnarrative labels")

    # Save the label mappings next to the model so inference can reuse them.
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_DIR / "narrative_mapping.json", "w", encoding="utf-8") as f:
        json.dump(narrative_label2id, f, ensure_ascii=False, indent=2)
    with open(MODEL_DIR / "subnarrative_mapping.json", "w", encoding="utf-8") as f:
        json.dump(subnarrative_label2id, f, ensure_ascii=False, indent=2)

    # Group subnarrative ids under their parent narrative ("Narrative: Subnarrative" format).
    hierarchical_mapping = {}
    for narr_label, narr_id in narrative_label2id.items():
        if narr_label.lower() == "other":
            continue
        group = [sub_id for sub_label, sub_id in subnarrative_label2id.items()
                 if sub_label.startswith(narr_label + ":")]
        if group:
            hierarchical_mapping[narr_id] = group

    # 90/10 train-dev split with a fixed seed for reproducibility.
    train_df, dev_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train samples: {len(train_df)}, dev samples: {len(dev_df)}")

    pos_weight_narrative = compute_pos_weights(train_df, "narrative", narrative_label2id)
    pos_weight_subnarrative = compute_pos_weights(train_df, "subnarrative", subnarrative_label2id)
    sample_weights = compute_sample_weights(train_df, "narrative", narrative_label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = NarrativeDataset(train_df, ARTICLES_DIR, tokenizer, MAX_LENGTH,
                                     narrative_label2id, subnarrative_label2id, augment=True)
    dev_dataset = NarrativeDataset(dev_df, ARTICLES_DIR, tokenizer, MAX_LENGTH,
                                   narrative_label2id, subnarrative_label2id, augment=False)

    model = MultiLabelClassifier(
        MODEL_NAME,
        num_narrative_labels=len(narrative_label2id),
        num_subnarrative_labels=len(subnarrative_label2id),
        pos_weight_narrative=pos_weight_narrative,
        pos_weight_subnarrative=pos_weight_subnarrative,
        use_focal_loss=True,
        gamma=1.5,
        alpha=0.5,
        hierarchical_mapping=hierarchical_mapping,
        narrative_id2label=narrative_id2label,
        consistency_weight=0.7,
        subnarrative_loss_weight=1.0,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # keep only the 2 most recent checkpoints (~2 GB each)
        logging_steps=50,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=500,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="f1_subnarrative",
        lr_scheduler_type="cosine",
    )

    trainer = CustomTrainerWithSampler(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_fn,
        sample_weights=sample_weights,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    model.encoder.config.save_pretrained(str(MODEL_DIR))
    print(f"Model, tokenizer and label mappings saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
