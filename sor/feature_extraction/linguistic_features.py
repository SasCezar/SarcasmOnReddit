import re
from collections import Counter
from typing import Dict
from typing import List, Tuple

import nltk
import spacy
from stanfordcorenlp import StanfordCoreNLP
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from feature_extraction.feature_extraction import AbstractFeatureExtractor

POSList = List[Tuple[str, str]]


class StanfordPOSFeatureExtraction(AbstractFeatureExtractor):
    def __init__(self, server: str = "http://127.0.0.1", port: int = 9000, lang: str = "en"):
        """
        :param server:
        :param port:
        :param lang:
        """
        super().__init__()
        self._tag_map = {"#": "HASHTAG", "$": "DOLLAR_SIGN", ".": "FINAL_PUNCT", ",": "COMMA",
                         ":": "MID_SENTENCE_PUNCT", "(": "LEFT_BRACKET", ")": "RIGHT_BRACKET",
                         "“": "LEFT_QUOTE", "”": "RIGHT_QUOTE"}

        self._tagger = StanfordCoreNLP(server, port=port, lang=lang)

    def run(self, text: str) -> Dict:
        if isinstance(text, float):
            return {}
        tags = self._tagger.pos_tag(text)
        tag_count = self._count_pos(tags)
        return tag_count

    def _rename_tags(self, tags: POSList) -> POSList:
        result = []
        for word, tag in tags:
            tag = self._tag_map[tag] if tag in self._tag_map else tag
            result.append((word, str(tag)))

        return result

    def _count_pos(self, tags: POSList) -> Dict:
        tags = [tag for word, tag in tags]
        return dict(Counter(tags))


class SpacyPOSTagger(AbstractFeatureExtractor):
    def __init__(self):
        super().__init__()
        self._tagger = spacy.load('en_core_web_sm')
        self._tags = {
            "PUNCT": 0, "SYM": 0, "X": 0, "ADJ": 0, "VERB": 0, "CONJ": 0, "NUM": 0, "DET": 0, "ADV": 0, "ADP": 0,
            "NOUN": 0, "PROPN": 0, "PART": 0, "PRON": 0, "SPACE": 0, "INTJ": 0
        }

    def run(self, text):
        doc = self._tagger(text)
        tags = [token.pos_ for token in doc]
        tag_count = self._count_pos(tags)
        return tag_count

    def _count_pos(self, tags: POSList) -> Dict:
        return dict(Counter(tags))


class VaderSentimentExtraction(AbstractFeatureExtractor):
    def __init__(self):
        super().__init__()
        self._analyzer = SentimentIntensityAnalyzer()

    def run(self, text: str) -> Dict:
        sentiment = self._analyzer.polarity_scores(text)
        return sentiment


class FormattingStatsExtraction(AbstractFeatureExtractor):
    def __init__(self):
        super().__init__()
        self._enlonged_re = re.compile(r"(.)\1{2}")

    def run(self, text):
        res = {'word_count': self._wordcount(text), 'char_count': len(text),
               'capital_case_percent': self._capital_case_count(text) / len(text),
               'enlonged_words_count': self._enlonged_count(text)}
        # res['capital_case_count'] = self._capital_case_count(text)

        return res

    @staticmethod
    def _wordcount(text):
        return len(nltk.word_tokenize(text))

    @staticmethod
    def _capital_case_count(text):
        return sum(1 for c in text if c.isupper())

    def _enlonged_count(self, text):
        return len([word for word in nltk.word_tokenize(text) if self._enlonged_re.search(word)])
