import logging
from abc import ABC, abstractmethod
from importlib import import_module
from typing import Dict, List


class AbstractPipeline(ABC):
    """
    Defines an abstract execution pipeline. Implements the common function that are needed to perform a generic computation
    """

    def __init__(self, algorithms: List = None, callables: List = None, **kwargs):
        """
        Constructor of the class. Accepts a dictionary of configurations that will be used to define the workflow of
        the class.

        :param algorithms:
        :param callables:
        :param kwargs:
        """
        self._kwargs = kwargs
        self._allowed = set()  # all these keys will be initialized as class attributes
        self._initialize()
        if not algorithms and not callables:
            raise ValueError()
        self._algorithms = algorithms
        if callables:
            self._callables = callables
        else:
            self._callables = []
            self._build()

    @abstractmethod
    def run(self, item):
        pass

    def get_config(self):
        configs = {}
        for name, callable_object in self._callables:
            object_config = callable_object.configuration()
            configs[name] = object_config

        return configs

    def _build(self):
        """
        Builds the objects that the class will use
        :return:
        """
        logging.info("Loading modules...")
        for algorithm in self._algorithms:
            algorithm_class = algorithm['class']
            algorithm_parameters = algorithm.get('parameters', {})
            logging.info("Loading {}".format(algorithm_class))
            algorithm_name, algorithm_instance = self._object_instancer(algorithm_class, algorithm_parameters)
            self._callables.append((algorithm_name, algorithm_instance))

    def _object_instancer(self, algorithm: str, algorithm_parameters: Dict):
        algorithm_name, algorithm_object = self._import_class(algorithm)
        algorithm_instance = algorithm_object(**algorithm_parameters)
        return algorithm_name, algorithm_instance

    def _initialize(self):
        """
        Set up the class defining the attributes
        :return:
        """
        # initialize all allowed keys to false
        for key in self._allowed:
            self.__setattr__(key, False)

        for key, value in self._kwargs.items():
            if key in self._allowed:
                self.__setattr__(key, value)

        return self

    @staticmethod
    def _import_class(module: str):
        """
        Imports the class from the specified module needed for the execution
        :param module:
        :return:
        """
        module_name = ".".join(module.split('.')[:-1])
        my_module = import_module("{}".format(module_name))
        class_name = module.split('.')[-1]
        model = getattr(my_module, class_name)
        return class_name, model


class SimplePipeline(AbstractPipeline):
    def run(self, item):
        for name, callable_object in self._callables:
            item = callable_object.run(item)
        return item


class SplitPipeline(AbstractPipeline):
    def run(self, item):
        res = {}
        for name, callable_object in self._callables:
            res[name] = callable_object.run(item)
        return res
