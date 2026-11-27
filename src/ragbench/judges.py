"""Built-in faithfulness judges.

:class:`TokenOverlap` needs nothing installed. It is a lexical baseline: a
claim counts as supported when its content words appear in one retrieved
context. It cannot see paraphrase or negation, so treat it as the floor a
real judge has to beat, not as a measure of truth.

:class:`NLI` runs a natural-language-inference model over each (context,
claim) pair. It needs ``pip install 'ragbench-eval[nli]'``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

_WORD = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*")
# Negations (no, nor, not) are kept as content words, so "does not apply" needs a "not".
_STOPWORDS = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
        "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
        "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
        "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
        "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "now", "of", "off",
        "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
        "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
        "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
        "yourself", "yourselves",
    }
)  # fmt: skip


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


class TokenOverlap:
    """Support = share of the claim's content words found in its best-matching context."""

    def __init__(self, min_words: int = 1) -> None:
        self.min_words = int(min_words)

    def support(self, claim: str, contexts: list[str], question: str = "") -> float:
        words = content_words(claim)
        if len(words) < self.min_words:
            # Nothing checkable ("Yes.", "See below."): neither support nor contradiction.
            return 1.0
        best = 0.0
        for ctx in contexts:
            overlap = len(words & content_words(ctx)) / len(words)
            best = max(best, overlap)
        return best


class NLI:
    """Support = the highest entailment probability of the claim given any one context.

    Uses a natural-language-inference cross-encoder from the Hugging Face Hub,
    run locally with ``transformers`` (``pip install 'ragbench-eval[nli]'``).
    Each claim is the hypothesis and each retrieved context is a premise; the
    model's probability that the premise entails the claim is its support. The
    entailment class is read from the model's own label names, so any
    three-way NLI model works.

    The default model is DeBERTa-v3-base fine-tuned on MNLI, FEVER and ANLI.
    """

    def __init__(
        self,
        model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        batch_size: int = 16,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        self.model_name = model
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = device
        self._loaded: tuple[Any, Any, int] | None = None

    def _load(self) -> tuple[Any, Any, int]:
        if self._loaded is None:
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise ImportError("NLI needs: pip install 'ragbench-eval[nli]'") from exc
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.to(self.device).eval()
            self._loaded = (tokenizer, model, entailment_index(model.config.id2label))
        return self._loaded

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """P(entailment) for each (premise, hypothesis) pair."""
        import torch

        tokenizer, model, entail = self._load()

        def score(chunk: list[tuple[str, str]]) -> list[float]:
            enc = tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation="only_first",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(model(**enc).logits.float(), dim=-1)
            return probs[:, entail].tolist()

        return by_length(pairs, self.batch_size, score)

    def support_batch(self, items: list[tuple[str, list[str], str]]) -> list[float]:
        pairs: list[tuple[str, str]] = []
        owner: list[int] = []
        for i, (claim, contexts, _question) in enumerate(items):
            for ctx in contexts:
                pairs.append((ctx, claim))
                owner.append(i)
        best = [0.0] * len(items)
        if pairs:
            for i, p in zip(owner, self._predict(pairs), strict=True):
                best[i] = max(best[i], min(1.0, max(0.0, float(p))))
        return best

    def support(self, claim: str, contexts: list[str], question: str = "") -> float:
        return self.support_batch([(claim, contexts, question)])[0]


def by_length(
    pairs: list[tuple[str, str]],
    batch_size: int,
    score: Callable[[list[tuple[str, str]]], list[float]],
) -> list[float]:
    """Score ``pairs`` in batches of similar length; return scores in input order.

    Sorting first means each padded batch is mostly real tokens.
    """
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    out = [0.0] * len(pairs)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        scores = score([pairs[i] for i in idx])
        if len(scores) != len(idx):
            raise ValueError(f"model returned {len(scores)} scores for {len(idx)} pairs")
        for i, value in zip(idx, scores, strict=True):
            out[i] = float(value)
    return out


def entailment_index(id2label: Mapping[Any, str]) -> int:
    """Index of the entailment class in a model's ``id2label`` config."""
    found = [int(i) for i, name in id2label.items() if str(name).lower().startswith("entail")]
    if len(found) != 1:
        raise ValueError(f"cannot find one entailment label in {dict(id2label)}")
    return found[0]
