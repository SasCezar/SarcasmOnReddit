import itertools
from collections import Counter

import matplotlib.pyplot as plt
import numpy
import numpy as np
import plotly
import plotly.graph_objs as go
from PIL import Image
from pandas import DataFrame
from wordcloud import WordCloud

IMAGES = False

SARCASM_COLOR = 'rgba(0,200,0,1)'
NO_SARCASM_COLOR = 'rgba(192,192,192,1)'

POS_TAGS = ['PUNCT', 'SYM', 'X', 'ADJ', 'VERB', 'CONJ', 'NUM', 'DET', 'ADV',
            'ADP', 'NOUN', 'PROPN', 'PART', 'PRON', 'SPACE', 'INTJ']

STATS_TAGS = ['WORD_COUNT', 'CHAR_COUNT', 'CAPITAL_CASE_PERCENT', 'ENLONGED_WORD_COUNT']

SENTIMENT_TAGS = ["POS", "NEG", "NEU"]
POS_COL = "rgb(84, 204, 20)"
NEU_COL = "rgb(255, 220, 0)"
NEG_COL = "rgb(255, 43, 23)"


def pos_bar_plot(df: DataFrame, X, name):
    groupdf = df.groupby(['label']).sum()
    Xs = [y for x in X for y in groupdf.columns.values if y.endswith(x)]
    y_scarcasm = [groupdf[x]['1'] for x in Xs]
    y_no_sarcasm = [groupdf[x]['0'] for x in Xs]

    count_sarcasm = sum(y_scarcasm)
    count_no_sarcasm = sum(y_no_sarcasm)

    plotly.tools.set_credentials_file(username='CezarAngeloSas', api_key='CgrETanZk0GtjkjDVXxs')

    no_sarcasm = go.Bar(
        x=X,
        y=[i / count_no_sarcasm for i in y_no_sarcasm],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    sarcasm = go.Bar(
        x=X,
        y=[i / count_sarcasm for i in y_scarcasm],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [no_sarcasm, sarcasm]
    layout = go.Layout(
        title="Part-of-Speech distribution",
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-bar.png'.format(name), height=650, width=1800) if IMAGES else None


def stats_bar_plot(df: DataFrame, X, name):
    groupdf = df.groupby(['label']).mean()
    Xs = [y for x in X for y in groupdf.columns.values if y.endswith("STATS_" + x)]
    y_scarcasm = [groupdf[x]['1'] for x in Xs]
    y_no_sarcasm = [groupdf[x]['0'] for x in Xs]

    plotly.tools.set_credentials_file(username='Giuseppe69', api_key='QXmVzu3sdvpiERas32kL')

    no_sarcasm = go.Bar(
        x=X,
        y=[i for i in y_no_sarcasm],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    sarcasm = go.Bar(
        x=X,
        y=[i for i in y_scarcasm],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [no_sarcasm, sarcasm]
    layout = go.Layout(
        title="Formatting stats",
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-bar.png'.format(name), height=650, width=1800) if IMAGES else None


def remap_sentiment(sentiment):
    if sentiment <= -0.05:
        return "NEG"
    if sentiment >= 0.05:
        return "POS"

    return "NEU"


def sentiment_correlation(df: DataFrame):
    poss = list(df['COMMENT_VADER_POS'])
    negs = list(df['COMMENT_VADER_NEG'])
    neus = list(df['COMMENT_VADER_NEU'])

    labels = list(df['label'])
    res = {"0": [], "1": []}
    for pos, neg, neu, label in zip(poss, negs, neus, labels):
        print(pos + neg + neu)
        if abs(pos - neg) <= 0.3 and (pos + neg) >= 0.7:
            score = pos + neg
            res[label].append(score)

    X = ["Instability"]

    y_sarcasm = [sum(res['1']) / (len(res['0']) + len(res['1']))]
    y_no_sarcasm = [sum(res['0']) / (len(res['0']) + len(res['1']))]

    no_sarcasm_p = go.Bar(
        x=X,
        y=y_no_sarcasm,
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    sarcasm_p = go.Bar(
        x=X,
        y=y_sarcasm,
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [sarcasm_p, no_sarcasm_p]
    layout = go.Layout(
        title="Instability",
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-bar.html'.format("instability"))
    plotly.plotly.image.save_as(fig, '{}-bar.png'.format("instability"), height=650, width=1800)


def sentiment_bar_plot(df: DataFrame, X, name):
    compound_sentiment = list(df['COMMENT_VADER_COMPOUND'])
    labels = list(df['label'])
    comments = list(df['comment'])

    text_sentiment = [remap_sentiment(sentiment) for sentiment in compound_sentiment]

    sarcasm = []
    no_sarcasm = []

    i = 10
    for sentiment, label, comment in zip(text_sentiment, labels, comments):
        if label == '1' and sentiment == "NEG" and i:
            print(comment)
            i -= 1
        if label == '0':
            no_sarcasm.append(sentiment)
        else:
            sarcasm.append(sentiment)

    y_sarcasm = Counter(sarcasm)
    y_no_sarcasm = Counter(no_sarcasm)

    plotly.tools.set_credentials_file(username='cezar.sas', api_key='8FM8AIu4kqaIjuw4UwQE')

    no_sarcasm_p = go.Bar(
        x=X,
        y=[y_no_sarcasm[x] / (y_sarcasm[x] + y_no_sarcasm[x]) for x in X],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    sarcasm_p = go.Bar(
        x=X,
        y=[y_sarcasm[x] / (y_sarcasm[x] + y_no_sarcasm[x]) for x in X],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [no_sarcasm_p, sarcasm_p]
    layout = go.Layout(
        title="Sentiment count",
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-bar.png'.format(name), height=650, width=1800)

    sarcasm = []
    no_sarcasm = []

    sarcasm_scores = {}
    no_sarcasm_scores = {}

    for x in SENTIMENT_TAGS:
        sarcasm_scores[x] = []
        no_sarcasm_scores[x] = []

    for sentiment, text, label, comment in zip(compound_sentiment, text_sentiment, labels, comments):
        if label == '0':
            no_sarcasm.append(sentiment)
            no_sarcasm_scores[text].append(abs(sentiment))
        else:
            sarcasm.append(sentiment)
            sarcasm_scores[text].append(abs(sentiment))

    sentiment_intensity_sarcasm = [sum(sarcasm_scores[x]) / (len(sarcasm_scores[x]) + len(no_sarcasm_scores[x]))
                                   for x in SENTIMENT_TAGS]
    no_sentiment_intensity_sarcasm = [sum(no_sarcasm_scores[x]) / (len(sarcasm_scores[x]) + len(no_sarcasm_scores[x]))
                                      for x in SENTIMENT_TAGS]

    sarcasm_p = go.Bar(
        x=SENTIMENT_TAGS,
        y=sentiment_intensity_sarcasm,
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(SENTIMENT_TAGS))
    )
    no_sarcasm_p = go.Bar(
        x=SENTIMENT_TAGS,
        y=no_sentiment_intensity_sarcasm,
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(SENTIMENT_TAGS))
    )

    data = [no_sarcasm_p, sarcasm_p]
    layout = go.Layout(
        title="Sentiment intensity",
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-multi-intensity-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-multi-intensity-bar.png'.format(name), height=650,
                                width=1800) if IMAGES else None

    sarcasm_intensity_l = [abs(x) for x in sarcasm if abs(x)]
    no_sarcasm_intensity_l = [abs(x) for x in no_sarcasm if abs(x)]
    sarcasm_intensity = sum(sarcasm_intensity_l) / len(compound_sentiment)
    no_sarcasm_intensity = sum(no_sarcasm_intensity_l) / len(compound_sentiment)

    sarcasm_p = go.Bar(
        x=['Avarage Sentiment'],
        y=[sarcasm_intensity],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR])
    )
    no_sarcasm_p = go.Bar(
        x=['Avarage Sentiment'],
        y=[no_sarcasm_intensity],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR])
    )

    data = [no_sarcasm_p, sarcasm_p]
    layout = go.Layout(
        title="Avarage Sentiment",
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='{}-intensity-bar.html'.format(name))
    plotly.plotly.image.save_as(fig, '{}-intensity-bar.png'.format(name), height=650, width=1800) if IMAGES else None


def subreddit_frequency(df: DataFrame):
    n = 10
    mc = Counter(list(df['subreddit'])).most_common(n)
    X = [x for x, y in list(mc)]

    size = len(df['subreddit'])
    tot = sum([y for x, y in mc])
    print("Top {} subreddits have {} comments, with a percent of {}".format(n, tot, tot / size))

    subreddits = df['subreddit']
    labels = df['label']

    y = []
    y_no = []
    for subreddit, label in zip(subreddits, labels):
        if subreddit in X:
            if label == '1':
                y.append(subreddit)
            else:
                y_no.append(subreddit)
    y = Counter(y)
    y_no = Counter(y_no)

    total = {}
    for k in X:
        total[k] = y[k] + y_no[k]

    y_scarcasm = [y[x] / total[x] for x in X]
    y_no_sarcasm = [y_no[x] / total[x] for x in X]

    subreddits_posts = go.Bar(
        x=X,
        y=[total[k] for k in total],
        name="Posts",
        marker=dict(
            color=['rgb(0,128,128)'] * len(X))
    )

    data = [subreddits_posts]
    layout = go.Layout(
        title='Number of comments for subreddit',
        barmode='group'
        # barmode='stack',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='subreddits-count-bar.html')
    plotly.plotly.image.save_as(fig, 'subreddits-count-bar.png', height=650, width=1800) if IMAGES else None

    no_sarcasm = go.Bar(
        x=X,
        y=y_no_sarcasm,
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    sarcasm = go.Bar(
        x=X,
        y=y_scarcasm,
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [no_sarcasm, sarcasm]
    layout = go.Layout(
        title='Distribution of sarcasm on subreddits',
        barmode='group'
        # barmode='stack',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='subreddits-sarcasm-stacked-bar.html')
    plotly.plotly.image.save_as(fig, 'subreddits-sarcasm-stacked-bar.png', height=650, width=1800) if IMAGES else None


def sarcasm_count(df: DataFrame):
    sarcasm = list(df.loc[df['label'] == '1'])
    no_sarcasm = list(df.loc[df['label'] == '0'])

    size = len(df)
    sarcasm_percent = str(len(sarcasm) / size)
    no_sarcasm_percent = str(len(no_sarcasm) / size)
    fig = {
        "data": [
            {
                "values": [sarcasm_percent, no_sarcasm_percent],
                "labels": ['Sarcasm', 'No Sarcasm'],
                "domain": {"x": [0, 1]},
                "name": "Percentage",
                "hoverinfo": "label+percent",
                "hole": .5,
                "type": "pie",
                'marker': {'colors': [SARCASM_COLOR, NO_SARCASM_COLOR]}
            }
        ],
        "layout": {
            "annotations": [
                {
                    "font": {
                        "size": 50
                    },
                    "showarrow": False,
                    "text": "Comments",
                    "x": 0.5,
                    "y": 0.5
                }
            ]

        }
    }

    plotly.offline.plot(fig, filename="word_tweets_donut.html")
    plotly.plotly.image.save_as(fig, 'word_tweets_donut.png', height=1000, width=1000) if IMAGES else None


def sentiment_donut(df: DataFrame):
    compound_sentiment = list(df['COMMENT_VADER_COMPOUND'])
    d_sent = [remap_sentiment(sentiment) for sentiment in compound_sentiment]
    pos = list([x for x in d_sent if x == "POS"])
    neu = list([x for x in d_sent if x == 'NEU'])
    neg = list([x for x in d_sent if x == 'NEG'])

    size = len(df)
    pos = str(len(pos) / size)
    neu = str(len(neu) / size)
    neg = str(len(neg) / size)

    fig = {
        "data": [
            {
                "values": [pos, neu, neg],
                "labels": ['POS', 'NEU', "NEG"],
                "domain": {"x": [0, 1]},
                "name": "Percentage",
                "hoverinfo": "label+percent",
                "hole": .5,
                "type": "pie",
                'marker': {'colors': [POS_COL, NEU_COL, NEG_COL]}
            }
        ],
        "layout": {
            "annotations": [
                {
                    "font": {
                        "size": 50
                    },
                    "showarrow": False,
                    "text": "",
                    "x": 0.5,
                    "y": 0.5
                }
            ]

        }
    }

    plotly.offline.plot(fig, filename="sentiment_donut.html")
    plotly.plotly.image.save_as(fig, 'sentiment_donut.png', height=1000, width=1000) if IMAGES else None


def line_plot():
    x = list(range(5, 255, 5))
    y = ["0.139093156", "0.144544842", "0.152398069", "0.185507624", "0.194224805", "0.194824002", "0.229285958",
         "0.247820547", "0.264407167", "0.275729953", "0.293775545", "0.317267362", "0.316181705", "0.329200448",
         "0.337833842", "0.359410367", "0.398491773", "0.409198164", "0.414637729", "0.427013555", "0.444729406",
         "0.457188012", "0.456761488", "0.463772105", "0.459410639", "0.463291314", "0.461429812", "0.451097374",
         "0.448617192", "0.438081814", "0.424349906", "0.424182862", "0.416271641", "0.411371919", "0.401029436"]

    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=''
    )

    data = [trace]
    layout = go.Layout(
        title='LDA Cv coherence using \'comment\' for different num of topics',
    )
    fig = dict(data=data, layout=layout)

    plotly.offline.plot(fig, filename="comment_topics_scatter.html")
    plotly.plotly.image.save_as(fig, 'comment_topics_scatter.png', height=650, width=1800) if IMAGES else None

    x = list(range(5, 75, 5))
    y = ["0.524239344", "0.532366949", "0.543829242", "0.543414296", "0.520849409", "0.501486787", "0.476610324",
         "0.455165027"]

    trace = go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=''
    )

    data = [trace]
    layout = go.Layout(
        title='LDA Cv coherence using \'parent\' for different num of topics',
    )
    fig = dict(data=data, layout=layout)

    plotly.offline.plot(fig, filename="parent_topics_scatter.html")
    plotly.plotly.image.save_as(fig, 'parent_topics_scatter.png', height=650, width=1800) if IMAGES else None


def multi_line_plot():
    x = list(range(5, 50, 5))
    miny = ["0.570600742", "0.617305215", "0.589788195", "0.576111981", "0.576524354", "0.54093114", "0.542133348",
            "0.538892902", "0.508340894"]
    y = ["0.612798631", "0.648314678", "0.624370912", "0.617182989", "0.587022568", "0.565596072", "0.558024734",
         "0.551900116", "0.530636077"]
    maxy = ["0.643953598", "0.674723348", "0.649077212", "0.653521721", "0.608184915", "0.58866727", "0.575485932",
            "0.565725347", "0.550627344"]
    # Create traces
    trace0 = go.Scatter(
        x=x,
        y=miny,
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=4,
            dash='dot'),
        name="Min"
    )
    trace1 = go.Scatter(
        x=x,
        y=y,
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=4),
        name="Avg"
    )
    trace2 = go.Scatter(
        x=x,
        y=maxy,
        line=dict(
            color=('rgb(124,252,0)'),
            width=4,
            dash='dot'),
        name="Max"
    )
    data = [trace0, trace1, trace2]
    layout = go.Layout(
        title='LDA Cv coherence using \'parent\' for different num of topics',
    )
    fig = dict(data=data, layout=layout)

    plotly.offline.plot(fig, filename="parent_multi_topics_scatter.html")
    plotly.plotly.image.save_as(fig, 'parent_multi_topics_scatter.png', height=650, width=1800) if IMAGES else None


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


def topics_bar(topics):
    topic_count = Counter(topics)

    sort_count = [(k, topic_count[k]) for k in sorted(topic_count, key=topic_count.get, reverse=True)]

    x = [TOPIC_MAP[k] for k, count in sort_count]
    y = [count for k, count in sort_count]

    subreddits_posts = go.Bar(
        x=x,
        y=y,
        name="No Sarcasm",
        marker=dict(
            color=['rgb(0,128,128)'] * len(x))
    )

    data = [subreddits_posts]
    layout = go.Layout(
        title='Distribution of comments on new topics',
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)

    plotly.offline.plot(fig, filename="topics_count_bar.html")
    plotly.plotly.image.save_as(fig, 'topics_count_bar.png', height=650, width=1800) if IMAGES else None


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    numpy.set_printoptions(precision=2)

    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png", bbox_inches='tight')


def create_wordcloud(text, filename):
    # create numpy araay for wordcloud mask image
    mask = np.array(Image.open("cloud.png"))

    # create set of stopwords

    # create wordcloud object
    wc = WordCloud(background_color="white", max_words=200, mask=mask)

    # generate wordcloud
    wc.generate(text)

    # save wordcloud
    wc.to_file(filename)


def topics_word_bar(words, score, name):
    plotly.tools.set_credentials_file(username='Giuseppe69', api_key='QXmVzu3sdvpiERas32kL')
    data = [go.Bar(
        x=score,
        y=words,
        orientation='h',
        name="Topic '{}' words".format(TOPIC_MAP[str(name)])
    )]

    layout = go.Layout(
        barmode='stack',
        title='Topic \'{}\' words'.format(TOPIC_MAP[str(name)])
    )
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="topic_{}_word_bar.html".format(name))
    plotly.plotly.image.save_as(fig, 'topic_{}_word_bar.png'.format(name), height=700, width=350) if IMAGES else None


def sarcasm_topic_frequency(topics, labels):
    plotly.tools.set_credentials_file(username='Giuseppe69', api_key='QXmVzu3sdvpiERas32kL')
    X = list(set(topics))
    y = []
    y_no = []
    for topic, label in zip(topics, labels):
        if str(label) == '1':
            y.append(topic)
        else:
            y_no.append(topic)

    y = Counter(y)
    y_no = Counter(y_no)

    total = {}
    for k in X:
        total[k] = y[k] + y_no[k]

    y_scarcasm = [y[x] / total[x] for x in X]
    y_no_sarcasm = [y_no[x] / total[x] for x in X]

    no_sarcasm = go.Bar(
        x=[TOPIC_MAP[z] for z in X],
        y=y_no_sarcasm,
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    sarcasm = go.Bar(
        x=[TOPIC_MAP[z] for z in X],
        y=y_scarcasm,
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [no_sarcasm, sarcasm]
    layout = go.Layout(
        title='Distribution of sarcasm on topics',
        barmode='group'
        # barmode='stack',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='topics-sarcasm-stacked-bar.html')
    plotly.plotly.image.save_as(fig, 'topics-sarcasm-stacked-bar.png', height=650, width=1800) if IMAGES else None
