"""
query_processor.py

Preprocesses engineering queries before retrieval.

Responsibilities
----------------
1. Normalize whitespace
2. Lowercase conversion
3. Engineering abbreviation expansion
4. Unit normalization
5. Stopword removal (optional)
6. Return normalized query
"""

from __future__ import annotations

import re
from typing import Dict


class QueryProcessor:

    ENGINEERING_TERMS: Dict[str, str] = {

        "ahu": "air handling unit",

        "cfm": "cubic feet per minute",

        "fpm": "feet per minute",

        "esp": "external static pressure",

        "rpm": "revolutions per minute",

        "kw": "kilowatt",

        "tr": "ton refrigeration",

        "dx": "direct expansion",

        "vfd": "variable frequency drive",

        "hvac": "heating ventilation air conditioning"
    }

    def __init__(self):

        pass

    @staticmethod
    def normalize_whitespace(text: str) -> str:

        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def remove_special_characters(text: str) -> str:

        return re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)

    def expand_abbreviations(self, text: str) -> str:

        words = []

        for word in text.split():

            key = word.lower()

            if key in self.ENGINEERING_TERMS:

                words.append(self.ENGINEERING_TERMS[key])

            else:

                words.append(word)

        return " ".join(words)

    @staticmethod
    def lowercase(text: str) -> str:

        return text.lower()

    def process(self, query: str) -> str:

        query = self.lowercase(query)

        query = self.remove_special_characters(query)

        query = self.normalize_whitespace(query)

        query = self.expand_abbreviations(query)

        query = self.normalize_whitespace(query)

        return query