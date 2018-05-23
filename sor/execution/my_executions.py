from execution.execution import AbstractExecBlock

from execution.pipeline import AbstractPipeline


class FeatureExtractionExecBlock(AbstractExecBlock):
    def __init__(self, pipeline: AbstractPipeline):
        super().__init__(pipeline)

    def __call__(self, item):
        # res = {'id': item.index}
        res = {}
        comment_features = self._pipeline.run(str(item['comment']))
        comment_features = {"comment_" + subkey: comment_features[key][subkey] for key in comment_features
                            for subkey in comment_features[key]}

        parent_features = self._pipeline.run(str(item['parent']))
        parent_features = {"parent_" + subkey: parent_features[key][subkey] for key in parent_features
                           for subkey in parent_features[key]}

        res.update(comment_features)
        res.update(parent_features)
        return res
