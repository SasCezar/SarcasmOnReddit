import logging
from collections import Counter

import pandas

import config
from execution.execution import SimpleExecution
from execution.my_executions import FeatureExtractionExecBlock
from execution.pipeline import SplitPipeline, SimplePipeline
from sorio.reddit import load_dataset
from visualization.plots import pos_bar_plot, subreddit_frequency, POS_TAGS, STATS_TAGS, stats_bar_plot, \
    SENTIMENT_TAGS, sentiment_bar_plot, sentiment_donut, line_plot, multi_line_plot, remap_sentiment, create_wordcloud

SET = "train"
DATASET = "../data/{}-balanced.tsv".format(SET)
EXTRACT_FEATURES = False
WC = False
VISUALIZE = True
FOLDS = 10
FILENAME = "../output/features_{}.csv".format(SET)
TESTFILR = "../output/features_{}.csv".format("TEST")


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    df = load_dataset(DATASET)

    print("Num record " + str(len(list(df['subreddit']))))
    print("Num subreddit " + str(len(set(list(df['subreddit'])))))

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
        comments = [x for x in list(df['comment'])]
        x['label'] = pandas.Series(labels, index=x.index)
        x['subreddit'] = pandas.Series(subreddits, index=x.index)
        x['comment'] = pandas.Series(comments, index=x.index)
        x = x[x.columns.drop(list(x.filter(regex='PARENT')))]
        pos_bar_plot(x, POS_TAGS, "pos")
        stats_bar_plot(x, STATS_TAGS, "stats")
        subreddit_frequency(x)
        # sarcasm_count(x)
        sentiment_bar_plot(x, SENTIMENT_TAGS, "sentiment")
        sentiment_donut(x)
        line_plot()
        multi_line_plot()

    if WC:
        labels = list(df['label'])
        cleaning_pipeline = SimplePipeline(config.PREPROCESSING_ALGORITHMS)
        comments = [str(cleaning_pipeline.run(x)).lower() for x in list(df['comment'])]
        # comments = [str(x).lower() for x in list(df['comment'])]

        compund = list(pandas.read_csv("../output/features_{}.csv".format(SET))['COMMENT_VADER_COMPOUND'])

        sentiment = list(map(remap_sentiment, compund))

        df = pandas.DataFrame({"comment": comments, "label": labels, "sentiment": sentiment})

        for l in [0, 1]:

            d = df.loc[df['label'] == l]
            text = " ".join(d['comment'])
            create_wordcloud(text, "{}_wc.png".format(l))
            count = Counter(text.split())
            freqs = [(i, count.get(i) / len(text.split()) * 100.0) for i, c in count.most_common(10)]
            print("Voc size:{} - {} - Freq: {}".format(len(count), l, count.most_common(10)))
            # create_wordcloud(text, "{}_{}_wc.png".format(s, l))

            for s in ["POS", "NEG", "NEU"]:
                d = df.loc[(df['sentiment'] == s) & (df['label'] == l)]
                text = " ".join(d['comment'])
                count = Counter(text.split())
                print("Voc size:{} - {} - {} - Freq: {}".format(len(count), s, l, count.most_common(10)))
                # create_wordcloud(text, "{}_{}_wc.png".format(s, l))

                d = df.loc[df['sentiment'] == s]
                text = " ".join(d['comment'])
                count = Counter(text.split())
                print("Voc size:{} - {} - Freq: {}".format(len(count), s, count.most_common(10)))
                # create_wordcloud(text, "{}_wc.png".format(s))

        # with open("{}_wordcloud.csv".format(SET), "wt", encoding="utf8", newline="") as outf:
        #     writer = csv.writer(outf)
        #     writer.writerow(["comment", "label", "sentiment"])
        #     for comment, label, sentiment in zip(comments, labels, compund):
        #         writer.writerow([cleaning_pipeline.run(comment).lower(), label, remap_sentiment(sentiment)])


if __name__ == '__main__':
    main()
