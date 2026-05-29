"""SHAP explainability for XGBoost and attention visualisation for DistilBERT."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REPORTS_DIR = Path("reports")


class XGBExplainer:
    """SHAP-based explainability for PhishingXGBClassifier.

    Produces a global summary bar chart and per-sample waterfall plots for the
    top phishing and top legitimate examples.
    """

    def __init__(self, clf, reports_dir: str | Path = DEFAULT_REPORTS_DIR):
        self._clf = clf
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _build_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Build the same feature matrix that XGBoost was trained on."""
        return self._clf._build_features(df)

    def _get_feature_names(self) -> list[str]:
        """Return feature names stored on the classifier after training."""
        return list(getattr(self._clf, "_feature_names", []))

    def compute_shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for every row in df; returns phishing-class values."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "shap is required for explainability. "
                "Install it with: pip install shap"
            ) from exc

        if self._clf._explainer is None:
            raise RuntimeError(
                "The classifier does not have a SHAP explainer attached. "
                "Make sure the model was trained before calling this method."
            )

        X = self._build_feature_matrix(df)
        shap_values = self._clf._explainer.shap_values(X)

        # TreeExplainer returns a list of arrays (one per class) for binary classification;
        # take the phishing-class values (index 1)
        if isinstance(shap_values, list):
            return shap_values[1]
        return shap_values

    def get_top_features(
        self, df: pd.DataFrame, top_n: int = 20
    ) -> pd.DataFrame:
        """Return a DataFrame of the top N features by mean absolute SHAP value."""
        shap_vals = self.compute_shap_values(df)
        feature_names = self._get_feature_names()

        mean_abs = np.abs(shap_vals).mean(axis=0)

        if len(feature_names) != len(mean_abs):
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)

        return importance_df

    def plot_summary(
        self,
        df: pd.DataFrame,
        top_n: int = 20,
        filename: str = "shap_summary.png",
    ) -> Path:
        """Save a horizontal bar chart of top N features by mean absolute SHAP value."""
        importance_df = self.get_top_features(df, top_n=top_n)

        fig, ax = plt.subplots(figsize=(9, 6))

        n = len(importance_df)
        colours = plt.cm.Blues(np.linspace(0.4, 0.9, n))[::-1]

        ax.barh(
            importance_df["feature"][::-1],
            importance_df["mean_abs_shap"][::-1],
            color=colours,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(
            f"XGBoost Feature Importance (top {n}, SHAP)",
            fontsize=13,
            fontweight="bold",
            pad=14,
        )
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        plt.tight_layout()
        save_path = self.reports_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("SHAP summary plot saved to %s", save_path)
        return save_path

    def _plot_waterfall_for_row(
        self,
        row_index: int,
        shap_vals_row: np.ndarray,
        feature_names: list[str],
        feature_values: np.ndarray,
        base_value: float,
        title: str,
        filename: str,
        top_n: int = 12,
    ) -> Path:
        """Draw a manual waterfall plot for a single sample.

        Drawn manually for broad SHAP version compatibility and full styling control.
        Red bars push toward phishing; blue bars push toward legitimate.
        """
        order = np.argsort(np.abs(shap_vals_row))[::-1]
        top_idx  = order[:top_n]
        rest_idx = order[top_n:]

        top_shap   = shap_vals_row[top_idx]
        top_names  = [feature_names[i] for i in top_idx]
        top_vals   = feature_values[top_idx]
        rest_total = shap_vals_row[rest_idx].sum() if len(rest_idx) > 0 else 0.0

        labels = [f"{n}={v:.3g}" for n, v in zip(top_names, top_vals)]
        contributions = list(top_shap)

        if rest_idx.size > 0:
            labels.append(f"other ({len(rest_idx)} features)")
            contributions.append(rest_total)

        cumulative = base_value
        starts = []
        for c in contributions:
            starts.append(cumulative)
            cumulative += c

        fig, ax = plt.subplots(figsize=(9, max(5, 0.45 * len(labels) + 2)))

        for i, (label, start, contrib) in enumerate(zip(labels, starts, contributions)):
            colour = "#d73027" if contrib > 0 else "#4575b4"
            ax.barh(
                i, contrib, left=start,
                color=colour, edgecolor="white", linewidth=0.5, height=0.6,
            )
            sign = "+" if contrib >= 0 else ""
            ax.text(
                start + contrib + (0.002 if contrib >= 0 else -0.002),
                i, f"{sign}{contrib:.3f}",
                va="center", ha="left" if contrib >= 0 else "right",
                fontsize=8,
            )

        ax.axvline(base_value, color="grey", linestyle="--", linewidth=1, label=f"base = {base_value:.3f}")
        ax.axvline(cumulative, color="black", linestyle="-", linewidth=1.2, label=f"pred = {cumulative:.3f}")

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("SHAP value contribution", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        red_patch  = mpatches.Patch(color="#d73027", label="pushes towards phishing")
        blue_patch = mpatches.Patch(color="#4575b4", label="pushes towards legitimate")
        ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")

        plt.tight_layout()
        save_path = self.reports_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Waterfall plot saved to %s", save_path)
        return save_path

    def plot_waterfall_phishing(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_prob: np.ndarray,
        filename: str = "shap_waterfall_phishing.png",
        top_n: int = 12,
    ) -> Path:
        """Save a waterfall plot for the highest-confidence true-positive phishing prediction."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required. Install with: pip install shap") from exc

        y_true_arr = np.asarray(y_true)
        tp_mask = (y_true_arr == 1) & (y_prob >= 0.5)
        if tp_mask.sum() == 0:
            tp_mask = y_true_arr == 1

        tp_probs   = np.where(tp_mask, y_prob, -1.0)
        chosen_idx = int(np.argmax(tp_probs))

        shap_vals     = self.compute_shap_values(df)
        feature_names = self._get_feature_names()
        X             = self._build_feature_matrix(df)

        base_value = float(self._clf._explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])

        return self._plot_waterfall_for_row(
            row_index=chosen_idx,
            shap_vals_row=shap_vals[chosen_idx],
            feature_names=feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])],
            feature_values=X[chosen_idx],
            base_value=base_value,
            title=f"SHAP Waterfall: Phishing Example (prob={y_prob[chosen_idx]:.1%})",
            filename=filename,
            top_n=top_n,
        )

    def plot_waterfall_legitimate(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_prob: np.ndarray,
        filename: str = "shap_waterfall_legitimate.png",
        top_n: int = 12,
    ) -> Path:
        """Save a waterfall plot for the highest-confidence true-negative legitimate prediction."""
        try:
            import shap
        except ImportError as exc:
            raise ImportError("shap is required. Install with: pip install shap") from exc

        y_true_arr = np.asarray(y_true)
        tn_mask = (y_true_arr == 0) & (y_prob < 0.5)
        if tn_mask.sum() == 0:
            tn_mask = y_true_arr == 0

        tn_probs   = np.where(tn_mask, y_prob, 2.0)
        chosen_idx = int(np.argmin(tn_probs))

        shap_vals     = self.compute_shap_values(df)
        feature_names = self._get_feature_names()
        X             = self._build_feature_matrix(df)

        base_value = float(self._clf._explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])

        return self._plot_waterfall_for_row(
            row_index=chosen_idx,
            shap_vals_row=shap_vals[chosen_idx],
            feature_names=feature_names if feature_names else [f"f{i}" for i in range(X.shape[1])],
            feature_values=X[chosen_idx],
            base_value=base_value,
            title=f"SHAP Waterfall: Legitimate Example (prob={y_prob[chosen_idx]:.1%})",
            filename=filename,
            top_n=top_n,
        )

    def explain_all(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_prob: np.ndarray,
        top_n: int = 20,
    ) -> dict:
        """Generate summary plot, phishing waterfall, and legitimate waterfall in one call."""
        summary_path = self.plot_summary(df, top_n=top_n)
        wf_phish     = self.plot_waterfall_phishing(df, y_true, y_prob)
        wf_legit     = self.plot_waterfall_legitimate(df, y_true, y_prob)
        top_features = self.get_top_features(df, top_n=top_n)

        return {
            "summary_plot":          str(summary_path),
            "waterfall_phishing":    str(wf_phish),
            "waterfall_legitimate":  str(wf_legit),
            "feature_importance_df": top_features,
        }


class BERTAttentionVisualizer:
    """Attention-based explainability for PhishingBERTClassifier.

    Extracts last-block attention weights, averages across heads, and renders
    a colour-coded word-importance grid saved to the reports directory.
    """

    def __init__(self, clf, reports_dir: str | Path = DEFAULT_REPORTS_DIR):
        self._clf = clf
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _get_attention_and_tokens(
        self, text: str, max_length: int = 128
    ) -> tuple[list[str], np.ndarray]:
        """Run one forward pass and return tokens with their CLS-attention scores.

        Reloads the model with attn_implementation='eager' because the default SDPA backend
        does not support output_attentions=True (raises 'tuple index out of range').
        """
        import torch
        from transformers import AutoModelForSequenceClassification

        model     = self._clf._model
        tokenizer = self._clf._tokenizer

        if model is None or tokenizer is None:
            raise RuntimeError(
                "Model or tokenizer is not loaded. "
                "Call clf.load() before using the visualizer."
            )

        model_name_or_path = model.config._name_or_path
        device = next(model.parameters()).device

        eager_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=2,
            attn_implementation="eager",
        ).to(device)
        eager_model.load_state_dict(model.state_dict())
        eager_model.eval()

        encoding = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        try:
            with torch.no_grad():
                outputs = eager_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )

            if outputs.attentions is None:
                raise ValueError(
                    "outputs.attentions is None even with attn_implementation='eager'. "
                    f"Output object type: {type(outputs)}. "
                    f"Output fields: {[k for k in outputs.keys() if outputs[k] is not None] if hasattr(outputs, 'keys') else 'N/A'}."
                )

            # outputs.attentions: tuple, one tensor per block, shape (batch, heads, seq, seq)
            last_attn = outputs.attentions[-1]

        except (IndexError, TypeError) as exc:
            n_outputs = len(outputs) if hasattr(outputs, "__len__") else "unknown"
            attn_val  = getattr(outputs, "attentions", "attribute missing")
            raise RuntimeError(
                f"Failed to extract attention weights: {exc}. "
                f"outputs.attentions = {attn_val!r}. "
                f"Number of output elements: {n_outputs}. "
                "This usually means the model is using SDPA or Flash Attention. "
                "Make sure attn_implementation='eager' was applied correctly."
            ) from exc

        # Average across heads; use CLS column (how much each token attended to CLS)
        mean_attn = last_attn[0].mean(dim=0).cpu().numpy()
        cls_col   = mean_attn[:, 0]

        token_ids = input_ids[0].cpu().numpy()
        tokens    = tokenizer.convert_ids_to_tokens(token_ids)

        return tokens, cls_col

    def _tokens_to_importance(
        self,
        tokens: list[str],
        attn_scores: np.ndarray,
        merge_wordpieces: bool = True,
    ) -> tuple[list[str], np.ndarray]:
        """Merge WordPiece sub-tokens back into words and sum their attention scores."""
        if not merge_wordpieces:
            return tokens, attn_scores

        words, scores = [], []
        current_word, current_score = None, 0.0

        for token, score in zip(tokens, attn_scores):
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                if current_word is not None:
                    words.append(current_word)
                    scores.append(current_score)
                    current_word, current_score = None, 0.0
                continue

            if token.startswith("##"):
                if current_word is not None:
                    current_word += token[2:]
                    current_score += score
                else:
                    current_word  = token[2:]
                    current_score = score
            else:
                if current_word is not None:
                    words.append(current_word)
                    scores.append(current_score)
                current_word  = token
                current_score = score

        if current_word is not None:
            words.append(current_word)
            scores.append(current_score)

        return words, np.array(scores, dtype=float)

    def visualize(
        self,
        text: str,
        label: Optional[str] = None,
        predicted_prob: Optional[float] = None,
        filename: str = "bert_attention.png",
        max_length: int = 128,
        top_n_words: int = 30,
    ) -> Path:
        """Save a colour-coded word importance chart for one input text.

        Background colour scales from white (no attention) to deep red (high attention).
        """
        tokens, attn = self._get_attention_and_tokens(text, max_length=max_length)
        words, importance = self._tokens_to_importance(tokens, attn)

        if len(words) == 0:
            logger.warning("No words found in text after tokenization. Skipping plot.")
            return self.reports_dir / filename

        words      = words[:top_n_words]
        importance = importance[:top_n_words]

        norm_imp = importance / (importance.max() + 1e-10)

        n_words = len(words)
        cols    = min(n_words, 8)
        rows    = (n_words + cols - 1) // cols
        fig_w   = max(cols * 1.6, 8)
        fig_h   = max(rows * 0.9 + 1.5, 3)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.axis("off")

        cmap = plt.cm.Reds

        for idx, (word, imp) in enumerate(zip(words, norm_imp)):
            col    = idx % cols
            row    = rows - 1 - idx // cols
            colour = cmap(0.15 + 0.75 * imp)

            rect = plt.Rectangle(
                (col + 0.05, row + 0.05), 0.9, 0.8,
                facecolor=colour, edgecolor="white", linewidth=1.5,
            )
            ax.add_patch(rect)

            text_colour = "white" if imp > 0.6 else "black"
            ax.text(
                col + 0.5, row + 0.45, word,
                ha="center", va="center",
                fontsize=8.5, color=text_colour, fontweight="bold",
            )
            ax.text(
                col + 0.5, row + 0.13, f"{imp:.2f}",
                ha="center", va="center",
                fontsize=6.5, color=text_colour, alpha=0.85,
            )

        title = "DistilBERT Attention Importance"
        if label is not None:
            title += f"  |  True label: {label}"
        if predicted_prob is not None:
            title += f"  |  Phishing prob: {predicted_prob:.1%}"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.02, shrink=0.4)
        cbar.set_label("Normalised attention score", fontsize=8)

        plt.tight_layout()
        save_path = self.reports_dir / filename
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("BERT attention plot saved to %s", save_path)
        return save_path

    def visualize_top_phishing(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_prob: np.ndarray,
        bert_clf,
        filename: str = "bert_attention_phishing.png",
        max_length: int = 128,
    ) -> Path:
        """Save an attention visualisation for the highest-confidence phishing sample."""
        y_true_arr = np.asarray(y_true)
        tp_mask    = (y_true_arr == 1) & (y_prob >= 0.5)
        if tp_mask.sum() == 0:
            tp_mask = y_true_arr == 1

        tp_probs   = np.where(tp_mask, y_prob, -1.0)
        chosen_idx = int(np.argmax(tp_probs))

        text = str(df.iloc[chosen_idx]["text"]) if "text" in df.columns else ""
        prob = float(y_prob[chosen_idx])

        return self.visualize(
            text=text,
            label="Phishing",
            predicted_prob=prob,
            filename=filename,
            max_length=max_length,
        )

    def visualize_top_legitimate(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_prob: np.ndarray,
        bert_clf,
        filename: str = "bert_attention_legitimate.png",
        max_length: int = 128,
    ) -> Path:
        """Save an attention visualisation for the highest-confidence legitimate sample."""
        y_true_arr = np.asarray(y_true)
        tn_mask    = (y_true_arr == 0) & (y_prob < 0.5)
        if tn_mask.sum() == 0:
            tn_mask = y_true_arr == 0

        tn_probs   = np.where(tn_mask, y_prob, 2.0)
        chosen_idx = int(np.argmin(tn_probs))

        text = str(df.iloc[chosen_idx]["text"]) if "text" in df.columns else ""
        prob = float(y_prob[chosen_idx])

        return self.visualize(
            text=text,
            label="Legitimate",
            predicted_prob=prob,
            filename=filename,
            max_length=max_length,
        )

    def explain_both(
        self,
        df: pd.DataFrame,
        y_true: pd.Series,
        y_prob: np.ndarray,
        bert_clf,
    ) -> dict:
        """Produce attention plots for both the top phishing and top legitimate sample."""
        phish_path = self.visualize_top_phishing(df, y_true, y_prob, bert_clf)
        legit_path = self.visualize_top_legitimate(df, y_true, y_prob, bert_clf)

        return {
            "attention_phishing":   str(phish_path),
            "attention_legitimate": str(legit_path),
        }
