"""
Step G — Phrase assembly layer.

Converts a stream of raw word predictions (each with label + confidence)
into a clean, deduplicated phrase.  Also provides template matching for
known target phrases.

Can be used standalone or imported by other inference scripts.

Usage (CLI demo):
    python inference/phrase_assembler.py

Public API:
    assembler = PhraseAssembler()
    assembler.push("hello", confidence=0.82)
    assembler.push("how_are_you", confidence=0.91)
    assembler.push("today", confidence=0.77)
    print(assembler.phrase())        # "hello how are you today"
    print(assembler.best_template()) # ("hello how are you today", 1.0)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional


# ── Display form for each class label ─────────────────────────────────────────
DISPLAY = {
    "hello":       "hello",
    "how":         "how",
    "how_are_you": "how are you",
    "you":         "you",
    "today":       "today",
}

# ── Known target phrases ───────────────────────────────────────────────────────
# Each entry: (display_text, ordered token list)
TARGET_PHRASES: list[tuple[str, list[str]]] = [
    ("Hi, how are you today?",   ["hello", "how_are_you", "today"]),
    ("Hello, how are you?",      ["hello", "how_are_you"]),
    ("How are you today?",       ["how_are_you", "today"]),
    ("How are you?",             ["how_are_you"]),
    ("Hello!",                   ["hello"]),
    ("How?",                     ["how"]),
    ("You today?",               ["you", "today"]),
]


class PhraseAssembler:
    """
    Accumulates word predictions and builds a clean phrase.

    Design decisions:
    - Duplicate suppression: same word repeated consecutively is collapsed.
    - Cooldown window: new pushes within `cooldown_s` seconds of the last
      confirmed token are ignored (prevents the same sign from flooding).
    - Max token history: configurable sliding window.
    - Template matching: substring-order match against TARGET_PHRASES.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        cooldown_s: float = 2.0,
        max_tokens: int = 10,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.cooldown_s = cooldown_s
        self.max_tokens = max_tokens

        self._tokens: deque[str] = deque(maxlen=max_tokens)
        self._last_push_time: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, label: str, confidence: float = 1.0) -> bool:
        """
        Offer a new word prediction.  Returns True if accepted into the phrase.

        Rejection conditions:
        - confidence below threshold
        - within cooldown window since last accepted token
        - same label as the most recent token (dedup)
        """
        if confidence < self.confidence_threshold:
            return False

        now = time.monotonic()
        if now - self._last_push_time < self.cooldown_s:
            return False

        if self._tokens and self._tokens[-1] == label:
            return False

        self._tokens.append(label)
        self._last_push_time = now
        return True

    def phrase(self) -> str:
        """Return the current assembled phrase as a display string."""
        return " ".join(DISPLAY.get(t, t) for t in self._tokens)

    def tokens(self) -> list[str]:
        """Return the raw token list."""
        return list(self._tokens)

    def clear(self) -> None:
        """Reset phrase and cooldown."""
        self._tokens.clear()
        self._last_push_time = 0.0

    def best_template(self) -> Optional[tuple[str, float]]:
        """
        Find the best-matching target phrase.

        Matching: for each template, count how many of its tokens appear
        in order (as a subsequence) in the current token list.
        Score = matched_tokens / template_length.

        Returns (display_text, score) or None if no tokens yet.
        """
        if not self._tokens:
            return None

        current = self.tokens()
        best_score = 0.0
        best_text  = None

        for display, tmpl in TARGET_PHRASES:
            score = _subsequence_score(current, tmpl)
            if score > best_score:
                best_score = score
                best_text  = display

        return (best_text, best_score) if best_text else None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _subsequence_score(tokens: list[str], template: list[str]) -> float:
    """
    Return fraction of template tokens found as an ordered subsequence in tokens.
    Score is in [0, 1]; 1.0 means the entire template is matched in order.
    """
    if not template:
        return 0.0
    ti = 0
    for tok in tokens:
        if ti < len(template) and tok == template[ti]:
            ti += 1
    return ti / len(template)


# ── CLI demo ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Phrase Assembler — interactive demo")
    print("Commands: enter label names (hello / how / how_are_you / you / today)")
    print("          'clear' to reset,  'quit' to exit\n")
    print(f"Classes: {list(DISPLAY.keys())}\n")

    asm = PhraseAssembler(confidence_threshold=0.0, cooldown_s=0.0)

    while True:
        try:
            raw = input("Sign > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if raw == "quit":
            break
        if raw == "clear":
            asm.clear()
            print("  [cleared]")
            continue
        if raw not in DISPLAY and raw != "":
            print(f"  Unknown label '{raw}'. Choose from: {list(DISPLAY.keys())}")
            continue

        if raw:
            accepted = asm.push(raw, confidence=1.0)
            print(f"  Accepted: {accepted}")

        phrase = asm.phrase()
        match  = asm.best_template()
        print(f"  Phrase : {phrase or '(empty)'}")
        if match:
            print(f"  Match  : {match[0]}  (score={match[1]:.2f})")
        print()


if __name__ == "__main__":
    main()
