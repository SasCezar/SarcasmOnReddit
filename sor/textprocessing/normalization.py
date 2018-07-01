import re
import string

import treetaggerwrapper
from nltk import SnowballStemmer

from textprocessing.textprocessor import AbstractTextProcessor


class Stemmer(AbstractTextProcessor):
    def __init__(self, language):
        self._stemmer = SnowballStemmer(language)
        self._splitregex = re.compile(r"""[\w']+|[^\s\w]""")

    def run(self, text):
        stemmed_text = self._stem_text(text)
        return stemmed_text

    def _stem_text(self, text):
        return ' '.join([self._stemmer.stem(w) for w in self._splitregex.findall(text)])


class Lemmer(AbstractTextProcessor):
    def __init__(self, language):
        self._lemmer = treetaggerwrapper.TreeTagger(TAGLANG=language).tag_text

    def run(self, text):
        lemmatized_text = self._text_lemmatization(text)
        return lemmatized_text

    def _text_lemmatization(self, text):
        """
        Given a text returns the lemmized version using the TreeTagger lemmatizer.

        :param text:
        :return:
        """
        lemmed_text = ""
        tagged_text = self._lemmer(text)
        for lemmed_word in tagged_text:
            original, tag, lemmed = lemmed_word.split()
            word = " " + lemmed if lemmed not in string.punctuation else lemmed

            lemmed_text += word

        return lemmed_text
