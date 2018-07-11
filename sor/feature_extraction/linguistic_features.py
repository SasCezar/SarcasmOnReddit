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


def _uppercase_for_dict_keys(lower_dict):
    upper_dict = {}
    for k, v in lower_dict.items():
        if isinstance(v, dict):
            v = _uppercase_for_dict_keys(v)
        upper_dict[k.upper()] = v
    return upper_dict


class SpacyPOSTagger(AbstractFeatureExtractor):
    def __init__(self, normalize=False):
        super().__init__()
        self._tagger = spacy.load('en_core_web_sm')
        self._tags = {
            "PUNCT": 0, "SYM": 0, "X": 0, "ADJ": 0, "VERB": 0, "CONJ": 0, "NUM": 0, "DET": 0, "ADV": 0, "ADP": 0,
            "NOUN": 0, "PROPN": 0, "PART": 0, "PRON": 0, "SPACE": 0, "INTJ": 0
        }
        self._tags = self._append_key(self._tags, "POS")
        self._norm = normalize

    def run(self, text):
        res = {}
        res.update(self._tags)
        doc = self._tagger(text)
        tags = [token.pos_ for token in doc]
        tag_count = self._count(tags)
        tag_count = self._append_key(tag_count, "POS")
        if self._norm:
            tag_count = self._normalize(tag_count)
        res.update(tag_count)
        return res

    @staticmethod
    def _normalize(d: Dict, target=1.0) -> Dict:
        raw = sum(d.values())
        factor = target / raw
        return {key: value * factor for key, value in d.items()}

    @staticmethod
    def _count(tags: POSList) -> Dict:
        return dict(Counter(tags))


class VaderSentimentExtraction(AbstractFeatureExtractor):
    def __init__(self):
        super().__init__()
        self._analyzer = SentimentIntensityAnalyzer()

    def run(self, text: str) -> Dict:
        sentiment = self._analyzer.polarity_scores(text)
        sentiment = self._append_key(_uppercase_for_dict_keys(sentiment), "VADER")
        return sentiment


class FormattingStatsExtraction(AbstractFeatureExtractor):
    def __init__(self, normalize=False):
        super().__init__()
        self._enlonged_re = re.compile(r"(.)\1{2,}")
        self._norm = normalize

    def run(self, text):
        res = _uppercase_for_dict_keys({'WORD_COUNT': self._wordcount(text), 'CHAR_COUNT': len(text),
                                        'CAPITAL_CASE_PERCENT': self._capital_case_count(text),
                                        'ENLONGED_WORD_COUNT': self._enlonged_count(text)})
        # res['capital_case_count'] = self._capital_case_count(text)
        res = self._append_key(res, "STATS")
        return res

    @staticmethod
    def _wordcount(text):
        res = len(nltk.word_tokenize(text))
        return res

    def _capital_case_count(self, text):
        count = sum(1 for c in text if c.isupper())
        if self._norm:
            count = count / len(text)

        return count

    def _enlonged_count(self, text):
        tokens = nltk.word_tokenize(text)
        count = len([word for word in tokens if self._enlonged_re.search(word)])
        if self._norm:
            count = count / len(tokens)
        return count
