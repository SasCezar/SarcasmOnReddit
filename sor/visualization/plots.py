import plotly
import plotly.graph_objs as go
from pandas import DataFrame

IMAGES = False

SARCASM_COLOR = 'rgba(0,200,0,1)'
NO_SARCASM_COLOR = ''


def pos_bar_plot(df: DataFrame):
    X = ["CC", "CD", "DT", "FW", "IN", "JJ", "JJR", "JJS", "MD", "NN", "NNP", "NNPS", "NNS", "POS", "PRP", "PRP$", "RB",
         "RBR", "RP", "TO", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]

    df = df.filter(items=X + ['label'])
    groupdf = df.groupby(['label']).sum()
    y_scarcasm = [groupdf[x]['1'] for x in X]
    y_no_sarcasm = [groupdf[x]['0'] for x in X]

    count_sarcasm = sum(y_scarcasm)
    count_no_sarcasm = sum(y_no_sarcasm)

    plotly.tools.set_credentials_file(username='cezar.sas', api_key='8FM8AIu4kqaIjuw4UwQE')

    sarcasm = go.Bar(
        x=X,
        y=[i / count_no_sarcasm for i in y_no_sarcasm],
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    no_sarcasm = go.Bar(
        x=X,
        y=[i / count_sarcasm for i in y_scarcasm],
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [sarcasm, no_sarcasm]
    layout = go.Layout(
        barmode='group',
    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='pos-bar.html')
    plotly.plotly.image.save_as(fig, 'pos-bar.png', height=650, width=1800) if IMAGES else None


def subreddit_frequency(df: DataFrame):
    df = df.filter(items=['subreddit', 'label'])
    X = list(set(df['subreddit']))
    groupdf = df.groupby(['subreddit']).count()
    y_scarcasm = [groupdf[x]['1'] for x in X]
    y_no_sarcasm = [groupdf[x]['0'] for x in X]

    plotly.tools.set_credentials_file(username='cezar.sas', api_key='8FM8AIu4kqaIjuw4UwQE')

    sarcasm = go.Bar(
        x=X,
        y=y_no_sarcasm,
        name="No Sarcasm",
        marker=dict(
            color=[NO_SARCASM_COLOR] * len(X))
    )
    no_sarcasm = go.Bar(
        x=X,
        y=y_scarcasm,
        name="Sarcasm",
        marker=dict(
            color=[SARCASM_COLOR] * len(X))
    )

    data = [sarcasm, no_sarcasm]
    layout = go.Layout(
        title='Distribution of sarcasm on subreddits',
        barmode='stack',
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
                "labels": ['Sarcasm', 'iPhone'],
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
                        "size": 20
                    },
                    "showarrow": False,
                    "text": "Tweets",
                    "x": 0.5,
                    "y": 0.5
                }
            ]

        }
    }

    plotly.offline.plot(fig, filename="word_tweets_donut.html")
    plotly.plotly.image.save_as(fig, 'word_tweets_donut.png', height=1000, width=1000) if IMAGES else None
