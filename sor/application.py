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
from visualization.plots import pos_bar_plot, subreddit_frequency, sarcasm_count, POS_TAGS, STATS_TAGS, stats_bar_plot, \
    SENTIMENT_TAGS, sentiment_bar_plot

SET = "train"
DATASET = "../data/{}-balanced.tsv".format(SET)
EXTRACT_FEATURES = False
CLASSIFY = False
VISUALIZE = True
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

    if VISUALIZE:
        x = pandas.read_csv("../output/features_{}.csv".format(SET))
        labels = [str(x) for x in list(df['label'])]
        subreddits = [str(x) for x in list(df['subreddit'])]
        print(len(subreddits))
        print(len(set(subreddits)))
        comments = [x for x in list(df['comment'])]
        x['label'] = pandas.Series(labels, index=x.index)
        x['subreddit'] = pandas.Series(subreddits, index=x.index)
        x['comment'] = pandas.Series(comments, index=x.index)
        x = x[x.columns.drop(list(x.filter(regex='PARENT')))]
        pos_bar_plot(x, POS_TAGS, "pos")
        stats_bar_plot(x, STATS_TAGS, "stats")
        subreddit_frequency(x)
        sarcasm_count(x)
        sentiment_bar_plot(x, SENTIMENT_TAGS, "sentiment")

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
