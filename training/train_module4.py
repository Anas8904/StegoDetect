"""
StegoDetect - Training Script for Module 4 (NLP Payload Classifier)

Trains both TF-IDF+SVM and optionally CodeBERT classifiers.

Usage:
    python training/train_module4.py --device cuda
    python training/train_module4.py --skip_codebert
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from modules.module4_nlp import (
    PayloadClassifier,
    TFIDFClassifier,
    CodeBERTClassifier,
    LABEL_NAMES,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


# ──────────────────────────────────────────────────────────────────────
# Text preprocessing
# ──────────────────────────────────────────────────────────────────────

def clean_payload_text(text):
    """Clean payload text for NLP training."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\x00', '')
    text = text.strip()
    # Truncate very long payloads (memory safety)
    text = text[:10000]
    return text


# ──────────────────────────────────────────────────────────────────────
# Part A: Train TF-IDF + SVM
# ──────────────────────────────────────────────────────────────────────

def train_tfidf_svm():
    """Train TF-IDF + LinearSVC classifier."""
    print("\n" + "=" * 60)
    print("Part A: Training TF-IDF + SVM Classifier")
    print("=" * 60)

    corpus_dir = config.NLP_CORPUS_DIR

    # Load training data
    train_path = corpus_dir / "train.csv"
    val_path = corpus_dir / "val.csv"

    if not train_path.exists():
        print(f"[WARN] Training corpus not found at {train_path}")
        print("       Build it first with: python data_prep/build_nlp_corpus.py")
        return None

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"  Train samples: {len(train_df)}")

    val_df = None
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        print(f"  Val samples:   {len(val_df)}")

    # Clean texts
    train_df['text'] = train_df['text'].apply(clean_payload_text)
    train_df = train_df[train_df['text'].str.len() > 0]

    if val_df is not None:
        val_df['text'] = val_df['text'].apply(clean_payload_text)
        val_df = val_df[val_df['text'].str.len() > 0]

    # Train
    classifier = TFIDFClassifier()
    print("\nTraining TF-IDF + SVM...")
    start = time.time()
    classifier.train(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
    )
    elapsed = time.time() - start
    print(f"  Training completed in {elapsed:.1f}s")

    # Evaluate on validation set
    if val_df is not None and len(val_df) > 0:
        print("\nEvaluating on validation set...")
        val_texts = val_df['text'].tolist()
        val_labels = val_df['label'].tolist()

        val_preds = []
        for text in val_texts:
            pred, _ = classifier.classify(text)
            val_preds.append(pred)

        acc = accuracy_score(val_labels, val_preds)
        f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"  Accuracy:    {acc*100:.2f}%")
        print(f"  Weighted F1: {f1:.4f}")

        # Classification report
        label_names = [LABEL_NAMES[i] for i in sorted(set(val_labels))]
        report = classification_report(
            val_labels, val_preds,
            target_names=label_names,
            digits=4,
        )
        print(f"\n{report}")

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("TF-IDF+SVM Confusion Matrix")
        fig.tight_layout()
        cm_path = config.RESULTS_DIR / "tfidf_confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"  Confusion matrix saved to {cm_path}")

    # Save model
    classifier.save()
    return classifier


# ──────────────────────────────────────────────────────────────────────
# Part B: Train CodeBERT
# ──────────────────────────────────────────────────────────────────────

def train_codebert(args):
    """Fine-tune CodeBERT classifier on obfuscated payloads."""
    print("\n" + "=" * 60)
    print("Part B: Fine-tuning CodeBERT")
    print("=" * 60)

    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
    except ImportError:
        print("[WARN] transformers library not available. Skipping CodeBERT training.")
        return None

    corpus_dir = config.NLP_CORPUS_DIR
    train_path = corpus_dir / "train.csv"
    val_path = corpus_dir / "val.csv"

    if not train_path.exists():
        print(f"[WARN] Training corpus not found at {train_path}")
        return None

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None

    # Clean texts
    train_df['text'] = train_df['text'].apply(clean_payload_text)
    if val_df is not None:
        val_df['text'] = val_df['text'].apply(clean_payload_text)

    # Filter to obfuscated samples if enough exist
    if 'obfuscation_type' in train_df.columns:
        obf_df = train_df[train_df['obfuscation_type'].isin(['base64', 'zip'])]
        if len(obf_df) >= 500:
            print(f"  Using {len(obf_df)} obfuscated samples for CodeBERT")
            train_df = obf_df
        else:
            print(f"  Only {len(obf_df)} obfuscated samples, using all {len(train_df)} samples")

    device = torch.device(args.device if args.device != "auto" else str(config.DEVICE))
    model_name = config.NLP_CONFIG["codebert_model_name"]
    max_length = config.NLP_CONFIG["codebert_max_length"]
    batch_size = config.NLP_CONFIG.get("codebert_batch_size", 8)
    epochs = args.codebert_epochs
    lr = config.NLP_CONFIG.get("codebert_lr", 2e-5)

    print(f"  Model:      {model_name}")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=config.NLP_NUM_CLASSES,
    ).to(device)

    # Dataset
    class PayloadDataset(Dataset):
        def __init__(self, texts, labels, tok, ml):
            self.texts = texts
            self.labels = labels
            self.tok = tok
            self.ml = ml

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tok(
                self.texts[idx], truncation=True, max_length=self.ml,
                padding='max_length', return_tensors='pt',
            )
            return {
                'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_dataset = PayloadDataset(
        train_df['text'].tolist(), train_df['label'].tolist(),
        tokenizer, max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_df is not None and len(val_df) > 0:
        val_dataset = PayloadDataset(
            val_df['text'].tolist(), val_df['label'].tolist(),
            tokenizer, max_length,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_f1 = 0.0
    save_path = str(config.CODEBERT_DIR)

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"    Epoch {epoch+1} Step {step+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")

        # Validate
        if val_loader:
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())

            f1 = f1_score(all_labels, all_preds, average='weighted')
            acc = accuracy_score(all_labels, all_preds)
            print(f"  Val - Acc: {acc*100:.2f}%  F1: {f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                print(f"  >> Best model saved! (F1: {best_f1:.4f})")

    if not val_loader:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Model saved to {save_path}")

    return True


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train StegoDetect Module 4 NLP")
    parser.add_argument("--skip_codebert", action="store_true",
                        help="Skip CodeBERT fine-tuning")
    parser.add_argument("--codebert_epochs", type=int,
                        default=config.NLP_CONFIG.get("codebert_epochs", 5))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Part A: TF-IDF + SVM
    tfidf = train_tfidf_svm()

    # Part B: CodeBERT
    if not args.skip_codebert:
        train_codebert(args)
    else:
        print("\n[INFO] Skipping CodeBERT training (--skip_codebert)")

    # Final evaluation with combined classifier
    print("\n" + "=" * 60)
    print("Module 4 Training Complete!")
    print("=" * 60)

    # Save metrics
    metrics = {
        "tfidf_trained": tfidf is not None,
        "codebert_trained": not args.skip_codebert,
        "timestamp": datetime.now().isoformat(),
    }
    metrics_path = config.RESULTS_DIR / "module4_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
