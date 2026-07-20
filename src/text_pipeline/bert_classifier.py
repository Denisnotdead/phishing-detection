"""DistilBERT fine-tuned for binary phishing classification."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"


class PhishingEmailDataset(Dataset):
    """PyTorch Dataset that tokenises text strings for DistilBERT input."""

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        max_length: int = 128,
        labels: Optional[list[int]] = None,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class PhishingBERTClassifier:
    """DistilBERT-based phishing classifier with training loop and save/load support."""

    def __init__(
        self,
        max_length: int = 128,
        batch_size: int = 16,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        patience: int = 3,
        models_dir: str | Path = "models",
    ):
        self.max_length = max_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.models_dir = Path(models_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("PhishingBERTClassifier using device: %s", self.device)

        # Halve batch size on CPU to avoid memory pressure
        self.batch_size = batch_size if self.device.type == "cuda" else batch_size // 2

        self._tokenizer = None
        self._model = None

    def _load_pretrained(self) -> None:
        """Download and initialise the DistilBERT tokeniser and model."""
        logger.info("Loading %s tokeniser and model...", MODEL_NAME)
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
        self._model.to(self.device)

    def _make_loader(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        shuffle: bool = False,
    ) -> DataLoader:
        """Build a DataLoader from a labelled or unlabelled DataFrame."""
        texts = df["text"].fillna("").astype(str).tolist()
        labels = df[label_col].astype(int).tolist() if label_col in df.columns else None

        dataset = PhishingEmailDataset(
            texts,
            tokenizer=self._tokenizer,
            max_length=self.max_length,
            labels=labels,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        label_col: str = "label",
    ) -> "PhishingBERTClassifier":
        """Fine-tune DistilBERT with early stopping based on validation loss."""
        self._load_pretrained()

        train_loader = self._make_loader(df_train, label_col, shuffle=True)
        val_loader   = self._make_loader(df_val, label_col, shuffle=False)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        total_steps  = len(train_loader) * self.epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # FP16 on CUDA cuts VRAM usage roughly in half
        scaler = GradScaler("cuda", enabled=(self.device.type == "cuda"))

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state_dict = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, scaler)
            val_loss, val_acc = self._eval_epoch(val_loader)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch, self.epochs, train_loss, val_loss, val_acc,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Keep best weights in CPU memory
                best_state_dict = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                logger.info("  New best val_loss=%.4f, saving weights.", best_val_loss)
            else:
                epochs_without_improvement += 1
                logger.info(
                    "  No improvement for %d epoch(s) (patience=%d).",
                    epochs_without_improvement, self.patience,
                )
                if epochs_without_improvement >= self.patience:
                    logger.info("Early stopping triggered.")
                    break

        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        logger.info("DistilBERT fine-tuning complete.")
        return self

    def _train_epoch(self, loader, optimizer, scheduler, scaler) -> float:
        """Run one full training epoch and return the average loss."""
        self._model.train()
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()

            with autocast("cuda", enabled=(self.device.type == "cuda")):
                outputs = self._model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _eval_epoch(self, loader) -> tuple[float, float]:
        """Run one evaluation pass and return (average loss, accuracy)."""
        self._model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with autocast("cuda", enabled=(self.device.type == "cuda")):
                    outputs = self._model(**batch)

                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                labels = batch["labels"].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict binary labels (0 or 1) for each row in df."""
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return class probability estimates; column 1 is the phishing probability."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        texts = df["text"].fillna("").astype(str).tolist()
        dataset = PhishingEmailDataset(
            texts,
            tokenizer=self._tokenizer,
            max_length=self.max_length,
            labels=None,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self._model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = self._model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def evaluate(self, df: pd.DataFrame, label_col: str = "label") -> dict:
        """Compute classification metrics on a labelled test DataFrame."""
        y_true = df[label_col].values.astype(int)
        y_pred = self.predict(df)
        y_prob = self.predict_proba(df)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Legitimate", "Phishing"]
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        logger.info(
            "BERT evaluation  acc=%.4f  f1=%.4f  roc_auc=%.4f",
            metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )
        return metrics

    def save(self, name: str = "bert_phishing") -> Path:
        """Save fine-tuned model and tokeniser to models_dir/<name>/ in HuggingFace format."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been fitted yet.")

        save_path = self.models_dir / name
        save_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(save_path))
        self._tokenizer.save_pretrained(str(save_path))

        logger.info("DistilBERT checkpoint saved to %s", save_path)
        return save_path

    def load(self, name: str = "bert_phishing") -> "PhishingBERTClassifier":
        """Load a saved checkpoint from models_dir/<name>/."""
        load_path = self.models_dir / name

        self._tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(load_path)
        )
        self._model.to(self.device)

        logger.info("DistilBERT checkpoint loaded from %s", load_path)
        return self
