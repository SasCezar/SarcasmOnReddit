import csv
import logging
import pickle
from typing import List

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore

import config
from execution.pipeline import SimplePipeline
from sorio.reddit import load_dataset
from visualization.plots import topics_word_bar, sarcasm_topic_frequency

algorithms = {"lemmed": {
    "class": "textprocessing.normalization.Lemmer",
    "parameters": {
        "language": "en"
    }
},
    "stemmed": {
        "class": "textprocessing.normalization.Stemmer",
        "parameters": {
            "language": "english"
        }
    }}


def clean_documents(documents, t):
    result = []
    i = 0
    size = len(documents)
    algs = config.PREPROCESSING_ALGORITHMS + [algorithms[t]]
    preprocessing_pipeline = SimplePipeline(algs)
    for document in documents:
        i += 1
        clean_document = preprocessing_pipeline.run(document)
        tokens = [x.lower() for x in clean_document.split() if len(x) > 2]
        result.append(tokens)
        logger.info("Processing document: {} of {} - Percent {}".format(i, size, i / size * 100))
    return result


def find_num_topics(documents: List, SET):
    # texts = clean_documents([str(doc) for doc in documents])
    PREPROCESSINGs = ["lemmed", "stemmed"]
    step = 10
    for PREPROCESSING in PREPROCESSINGs:
        # texts = clean_documents([str(doc) for doc in documents], PREPROCESSING)
        # pickle.dump(texts, open("{}_{}_text".format(SET, PREPROCESSING), "wb"))
        texts = pickle.load(open("{}_{}_text".format(SET, PREPROCESSING), "rb"))
        logger.info("Clean complete")
        dictionary = Dictionary(texts)
        print('Number of unique tokens: %d' % len(dictionary))
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        print('Number filtered of unique tokens: %d' % len(dictionary))
        logger.info("Dictionary complete")
        corpus = [dictionary.doc2bow(text) for text in texts]
        logger.info("Corpus complete")

        with open("results_{}_{}_step_{}_filtered.csv".format(SET, step, PREPROCESSING), "at", encoding="utf8",
                  newline="") as outf:
            writer = csv.writer(outf)
            for i in range(250, 350, step):
                res = [i]
                for j in range(0, 1):
                    model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=200, num_topics=i, workers=3)
                    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                    res.append(coherence_model.get_coherence())

                writer.writerow(res)
                outf.flush()


def label_documents(documents: List, LVL, SET):
    OPTIMAL_TOPICS = 10
    PREPROCESSINGs = ["lemmed"]
    for PREPROCESSING in PREPROCESSINGs:
        # texts = clean_documents([str(doc) for doc in documents], PREPROCESSING)
        # pickle.dump(texts, open("{}_{}_{}_text".format(SET, LVL, PREPROCESSING), "wb"))
        texts = pickle.load(open("{}_{}_{}_text".format(SET, LVL, PREPROCESSING), "rb"))
        dictionary = Dictionary(texts)
        print('Number of unique tokens: %d' % len(dictionary))
        dictionary.filter_extremes(no_below=20, no_above=0.5)
        print('Number filtered of unique tokens: %d' % len(dictionary))

        logger.info("Clean complete")
        logger.info("Dictionary complete")
        corpus = [dictionary.doc2bow(text) for text in texts]
        logger.info("Corpus complete")
        # model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=1000, num_topics=OPTIMAL_TOPICS, workers=3)

        # model.save("{}_{}_{}_model_cv_coherence_{}".format(LVL, OPTIMAL_TOPICS, PREPROCESSING))

        model = LdaMulticore.load("{}_{}_1000_model_cv_coherence_{}".format(LVL, OPTIMAL_TOPICS, PREPROCESSING))
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        print("Coherence {}".format(coherence_model.get_coherence()))
        topics = []
        corpus = [dictionary.doc2bow(text) for text in texts]
        for document in corpus:
            topic = model.get_document_topics(document)
            topics.append(topic)

        with open("topics_results_{}_{}_{}.csv".format(SET, LVL, PREPROCESSING), "wt", encoding="utf8",
                  newline="") as outf:
            writer = csv.writer(outf)
            for document_topics in topics:
                sorted_topics = sorted(document_topics, key=lambda x: -x[1])
                if sorted_topics:
                    best = [sorted_topics[0][0]]
                else:
                    best = [-1]
                    print("NONE ERROR")
                writer.writerow(best)

        x = model.show_topics(num_topics=OPTIMAL_TOPICS, num_words=10, formatted=True)
        # Below Code Prints Topics and Words
        with open("topics_keywords_{}_{}_{}".format(SET, LVL, PREPROCESSING), "wt", encoding="utf8",
                  newline="") as outf:
            writer = csv.writer(outf)
            for t in x:
                topic = t[0]
                words = [(word_score.split("*")[1].strip()[1:-1], float(word_score.split("*")[0].strip())) for
                         word_score in
                         t[1].split("+")]
                sort_words = sorted(words, key=lambda z: z[1])
                print(str(topic) + " " + str(sort_words))
                words = [w for w, s in sort_words]
                score = [s for w, s in sort_words]
                topics_word_bar(words, score, t[0])
                writer.writerow([topic] + [sort_words])

    return


TOPIC_MAP = {"0": "Movies",
             "1": "Crime/News",
             "2": "Sport",
             "3": "Lifestyle",
             "4": "Entertainment",
             "5": "Human Rights",
             "6": "Hybrid",
             "7": "PC/Gaming",
             "8": "Politics",
             "9": "Technology"}

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    SET = "train"
    DATASET = "../../data/{}-balanced.tsv".format(SET)

    df = load_dataset(DATASET)
    LVL = "parent"
    comments = list(df[LVL])
    subreddits = list(df["subreddit"])
    labels = list(df["label"])

    label_documents(comments, LVL, SET)

    with open("topics_results_{}_{}_{}.csv".format(SET, LVL, "lemmed"), "rt", encoding="utf8") as inf:
        topics = [line.strip() for line in inf]

    print(len(topics))
    print(len(subreddits))
    print(len(labels))

    with open("{}_{}_{}_topics_subreddit.csv".format(SET, LVL, "lemmed"), "wt", encoding="utf8", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["topic", "topic_name", "subreddit", "label"])
        for topic, subreddit, label in zip(topics, subreddits, labels):
            writer.writerow([topic, TOPIC_MAP[topic], subreddit, label])

    with open("{}_{}_{}_topics_subreddit_top_10.csv".format(SET, LVL, "lemmed"), "wt", encoding="utf8",
              newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["topic", "topic_name", "subreddit", "label"])
        for topic, subreddit, label in zip(topics, subreddits, labels):
            if subreddit.lower() in ["askreddit", "politics", "worldnews", "leagueoflegends", "pcmasterrace", "funny",
                                     "news", "pics", "todayilearned", "nfl"]:
                writer.writerow([topic, TOPIC_MAP[topic], subreddit, label])

    # topics_bar(topics)
    sarcasm_topic_frequency(topics, labels)

    # cleaning_pipeline = SimplePipeline(config.PREPROCESSING_ALGORITHMS)
    # comments = [str(cleaning_pipeline.run(x)).lower() for x in list(df['comment'])]
    # comments = [str(x).lower() for x in list(df['comment'])]
    #
    # df = pandas.DataFrame({"comments": comments, "topic": topics})
    # for i in range(0, 10):
    #     d = df.loc[df['topic'] == str(i)]
    #     text = " ".join(list(d['comments']))
    #     create_wordcloud(text, "topic_{}_wc.png".format(i))
