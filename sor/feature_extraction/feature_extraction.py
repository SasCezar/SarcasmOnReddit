from abc import abstractmethod, ABC
from typing import Dict


class AbstractFeatureExtractor(ABC):
    """
    Implements an abstract element in a pipeline
    """

    def __init__(self):
        pass

    @abstractmethod
    def run(self, obj):
        """
        Executes the algorithm on the input, the result depends on the algorithm.

        :param obj: A generic input, depends on the implementation of the algorithm.
        :return: The features
        """
        pass

    @staticmethod
    def _append_key(d: Dict, string):
        return {string + "_" + key: d[key] for key in d}
