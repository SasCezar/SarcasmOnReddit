import logging

import pandas
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.model_selection import cross_val_score

import config
from execution.execution import SimpleExecution
from execution.my_executions import FeatureExtractionExecBlock
from execution.pipeline import SplitPipeline
from sorio.reddit import load_dataset

SET = "train"
DATASET = "../data/{}-balanced.tsv".format(SET)
EXTRACT_FEATURES = True
CLASSIFY = False
FOLDS = 10
FILENAME = "../output/features_{}.csv".format(SET)


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    df = load_dataset(DATASET)

    # cleaning_pipeline = SimplePipeline(config.FEATURE_EXTRACTION_ALGORITHMS)
    #
    # if "clean" in DATASET:
    #     df['comment'] = df['comment'].apply(cleaning_pipeline.run)
    #     df['parent'] = df['parent'].apply(cleaning_pipeline.run)
    #
    #     df.to_csv("../data/{}-balanced_clean.tsv".format(SET))

    if EXTRACT_FEATURES:
        feature_extraction_pipeline = SplitPipeline(config.FEATURE_EXTRACTION_ALGORITHMS)

        dataset = df.to_dict('records')
        execblock = FeatureExtractionExecBlock(feature_extraction_pipeline)
        features = SimpleExecution().execute(execblock, dataset)

        outdf = pandas.DataFrame(features)
        outdf.to_csv(FILENAME)

    if CLASSIFY:
        x = pandas.read_csv(FILENAME)
        x = x[x.columns.drop(list(x.filter(regex='PARENT')))]
        x.fillna(0, inplace=True)
        enlonged = x.filter(regex='_POS_').as_matrix()
        x = x.loc[:, (x != 0).any(axis=0)]
        x.corr()
        y = pandas.DataFrame({'label': df['label']})

        comments = [str(c) for c in df['comment']]
        vectorizer = HashingVectorizer(stop_words='english', analyzer='word', n_features=4096)
        hv = vectorizer.transform(comments)
        x = hstack((hv, enlonged), format='csr')
        print(x.shape)
        print(y.shape)

        clf = RandomForestClassifier(n_estimators=200)
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 400, 200, ), random_state=1)
        #
        scores = cross_val_score(clf, x, y, cv=FOLDS, n_jobs=-1)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        """
        pca = PCA(n_components=100)
        principal_components = pca.fit_transform(x, y)
        print("PCA Computed")
        scores = cross_val_score(clf, principal_components, y, cv=10, n_jobs=-1)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        """

        """
        clf = LogisticRegression()
        scores = cross_val_score(clf, x, y, cv=FOLDS, n_jobs=8)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        """
        clf = RandomizedLogisticRegression()
        clf.fit(x, y)
        print(clf.get_support())
        print(sum(clf.get_support()))

        rfe = RFE(estimator=clf, n_features_to_select=1024, step=1)
        """
        rfecv = RFECV(estimator=clf, step=1, cv=KFold(5),
                      scoring='accuracy', n_jobs=-1)
        rfecv.fit(x, y)
        print("Optimal number of features : %d" % rfecv.n_features_)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()
        """
        # print("SVM")
        # svc = svm.SVC()
        # scores = cross_val_score(svc, x, y, cv=2, n_jobs=-1)
        # print(scores)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
