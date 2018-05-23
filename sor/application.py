import os

from pandas import DataFrame

import config
from execution.execution import SimpleExecution
from execution.my_executions import FeatureExtractionExecBlock
from execution.pipeline import SplitPipeline
from sorio.reddit import load_dataset

DATASET = "../data/test-balanced.tsv"


def run(item):
    item = item['label']
    pid = os.getpid()
    print("{} - {}".format(item, pid))


def main():
    feature_extraction_pipeline = SplitPipeline(config.FEATURE_EXTRACTION_ALGORITHMS)
    df = load_dataset(DATASET)
    dataset = df.to_dict('records')

    dataset = dataset

    execblock = FeatureExtractionExecBlock(feature_extraction_pipeline)
    features_extraction = SimpleExecution().execute(execblock, dataset)

    len(features_extraction)
    outdf = DataFrame(features_extraction)
    outdf.to_csv("C:/Users/sasce/desktop/out_2.csv")


if __name__ == '__main__':
    main()
