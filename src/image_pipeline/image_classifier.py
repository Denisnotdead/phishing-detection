"""EfficientNet-B0 fine-tuned for binary phishing screenshot classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logger = logging.getLogger(__name__)

INPUT_SIZE = 224


class PhishingImageDataset(Dataset):
    """Dataset that loads images from file paths and applies a torchvision transform."""

    def __init__(
        self,
        image_paths: list[str | Path],
        transform,
        labels: Optional[list[int]] = None,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(image)
        item = {"pixel_values": tensor}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_train_transform() -> transforms.Compose:
    """Build the augmentation transform used during training.

    Augmentations simulate real screenshot variation: rotations, crops, brightness/saturation shifts.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE + 32, INPUT_SIZE + 32)),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform() -> transforms.Compose:
    """Build the deterministic transform for validation and inference.

    Uses ImageNet statistics matching the pretrained EfficientNet weights.
    """
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class PhishingImageClassifier:
    """EfficientNet-B0 phishing screenshot classifier with two-phase fine-tuning.

    Public interface mirrors PhishingBERTClassifier for interchangeable use in the ensemble.
    """

    def __init__(
        self,
        batch_size: int = 16,
        epochs: int = 20,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 2,
        patience: int = 5,
        models_dir: str | Path = "models",
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.models_dir = Path(models_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("PhishingImageClassifier using device: %s", self.device)

        self._model: Optional[nn.Module] = None

    def _build_model(self) -> nn.Module:
        """Load pretrained EfficientNet-B0 and replace the classifier head for binary classification.

        All layers start frozen; they are selectively unfrozen in fit().
        """
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        for param in model.parameters():
            param.requires_grad = False

        # Replace 1000-class head with 2-class head
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 2),
        )

        return model.to(self.device)

    def _unfreeze_top_blocks(self, model: nn.Module) -> None:
        """Unfreeze the last two EfficientNet feature blocks and classifier after warmup."""
        blocks = list(model.features.children())
        for block in blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

    def _make_loader(
        self,
        image_paths: list[str | Path],
        labels: Optional[list[int]],
        transform,
        shuffle: bool = False,
    ) -> DataLoader:
        """Build a DataLoader from paths, labels, and a transform."""
        dataset = PhishingImageDataset(image_paths, transform=transform, labels=labels)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=(self.device.type == "cuda"),
        )

    def fit(
        self,
        train_paths: list[str | Path],
        train_labels: list[int],
        val_paths: list[str | Path],
        val_labels: list[int],
    ) -> "PhishingImageClassifier":
        """Fine-tune EfficientNet-B0 with warmup then top-block unfreezing."""
        self._model = self._build_model()
        train_transform = build_train_transform()
        eval_transform  = build_eval_transform()

        train_loader = self._make_loader(train_paths, train_labels, train_transform, shuffle=True)
        val_loader   = self._make_loader(val_paths, val_labels, eval_transform, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state_dict = None

        for epoch in range(1, self.epochs + 1):

            # After warmup, unfreeze deeper layers and rebuild optimizer with new params
            if epoch == self.warmup_epochs + 1:
                logger.info("Epoch %d: unfreezing top EfficientNet blocks.", epoch)
                self._unfreeze_top_blocks(self._model)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self._model.parameters()),
                    lr=self.learning_rate * 0.1,
                    weight_decay=1e-4,
                )
                scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))

            train_loss = self._train_epoch(train_loader, optimizer, criterion, scaler)
            val_loss, val_acc = self._eval_epoch(val_loader, criterion)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch, self.epochs, train_loss, val_loss, val_acc,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state_dict = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                logger.info("  New best val_loss=%.4f, checkpoint saved.", best_val_loss)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info("Early stopping after %d epochs.", epoch)
                    break

        if best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        logger.info("EfficientNet training complete.")
        return self

    def _train_epoch(self, loader, optimizer, criterion, scaler) -> float:
        """Run one training epoch and return average loss."""
        self._model.train()
        total_loss = 0.0

        for batch in loader:
            images = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self._model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _eval_epoch(self, loader, criterion) -> tuple[float, float]:
        """Run one evaluation pass and return (average loss, accuracy)."""
        self._model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                images = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = self._model(images)
                    loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    def predict_proba(self, image_paths: list[str | Path]) -> np.ndarray:
        """Return class probability estimates; column 1 is the phishing probability."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        loader = self._make_loader(
            image_paths, labels=None, transform=build_eval_transform(), shuffle=False
        )

        self._model.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                images = batch["pixel_values"].to(self.device)
                with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                    logits = self._model(images)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, image_paths: list[str | Path]) -> np.ndarray:
        """Predict binary labels (0 or 1) for a list of images."""
        return (self.predict_proba(image_paths)[:, 1] >= 0.5).astype(int)

    def evaluate(
        self,
        image_paths: list[str | Path],
        labels: list[int],
    ) -> dict:
        """Compute classification metrics for a labelled set of images."""
        y_true = np.array(labels)
        y_pred = self.predict(image_paths)
        y_prob = self.predict_proba(image_paths)[:, 1]

        metrics = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc":   float(roc_auc_score(y_true, y_prob)),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Legitimate", "Phishing"]
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        logger.info(
            "EfficientNet evaluation  acc=%.4f  f1=%.4f  roc_auc=%.4f",
            metrics["accuracy"], metrics["f1"], metrics["roc_auc"],
        )
        return metrics

    def save(self, name: str = "efficientnet_phishing") -> Path:
        """Save model weights to models_dir/<name>.pt."""
        if self._model is None:
            raise RuntimeError("Nothing to save — model has not been fitted yet.")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.models_dir / f"{name}.pt"
        torch.save(self._model.state_dict(), save_path)
        logger.info("EfficientNet weights saved to %s", save_path)
        return save_path

    def load(self, name: str = "efficientnet_phishing") -> "PhishingImageClassifier":
        """Load model weights from models_dir/<name>.pt."""
        load_path = self.models_dir / f"{name}.pt"
        self._model = self._build_model()
        self._model.load_state_dict(
            torch.load(load_path, map_location=self.device)
        )
        logger.info("EfficientNet weights loaded from %s", load_path)
        return self
