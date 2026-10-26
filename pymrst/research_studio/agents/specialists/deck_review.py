"""Deck structure, keyword, include, unit, and coverage review specialist."""

from . import DeterministicSpecialist


class DeckReviewSpecialist(DeterministicSpecialist):
    name = "deck_review"
    tool_name = "analyze_deck_review"


__all__ = ["DeckReviewSpecialist"]
