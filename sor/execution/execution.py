import logging
import math
import multiprocessing
from abc import ABC, abstractmethod
from queue import Queue
from typing import Callable

from execution.pipeline import AbstractPipeline


class AbstractExecution(ABC):
    @abstractmethod
    def execute(self, func: Callable, items, **kwargs):
        pass


class SimpleExecution(AbstractExecution):
    def execute(self, func: Callable, items, **kwargs):
        res = []
        size = len(items)
        i = 0
        for item in items:
            i += 1
            res.append(func(item))
            logging.info("Percent complete: {} - Item number: {}".format(i / size * 100, i))
        return res


class ParallelExecution(AbstractExecution):
    def execute(self, func: Callable, items, **kwargs):
        result = Queue()
        processes = []
        chunksize = int(math.ceil(len(items) / float(kwargs['threads'])))
        logging.info("Initializing processes")

        for i in range(kwargs['threads']):
            p = multiprocessing.Process(target=func,
                                        args=(items[chunksize * i:chunksize * (i + 1)], result))
            processes.append(p)
            p.start()

        logging.info("Joining results")
        webpages = []
        for x in range(len(processes)):
            webpages += result.get()

        logging.info("Waiting processes to end")
        for p in processes:
            p.join()

        return result


class AbstractExecBlock(ABC):
    def __init__(self, pipeline: AbstractPipeline):
        self._pipeline = pipeline

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
