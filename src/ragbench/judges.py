"""Built-in faithfulness judges.

:class:`TokenOverlap` needs nothing installed. It is a lexical baseline: a
claim counts as supported when its content words appear in one retrieved
context. It cannot see paraphrase or negation, so treat it as the floor a
real judge has to beat, not as a measure of truth.
"""

from __future__ import annotations

import re

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
