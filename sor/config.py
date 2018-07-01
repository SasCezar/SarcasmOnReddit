FEATURE_EXTRACTION_ALGORITHMS = [
    {
        'class': 'feature_extraction.linguistic_features.SpacyPOSTagger',
        'parameters': {
            'normalize': 0
        }
    },
    {
        'class': 'feature_extraction.linguistic_features.VaderSentimentExtraction',
        'parameters': {
        }
    },
    {
        'class': 'feature_extraction.linguistic_features.FormattingStatsExtraction',
        'parameters': {
            'normalize': 0
        }
    }
]

PREPROCESSING_ALGORITHMS_FEATURES = [
    {
        "class": "textprocessing.cleaner.MailCleaner"
    },
    {
        "class": "textprocessing.cleaner.URLCleaner"
    },
    {
        "class": "textprocessing.cleaner.WordsWithNumbersCleaner"
    },
    {
        "class": "textprocessing.cleaner.NonCharCleaner"
    },
    {
        "class": "textprocessing.cleaner.MultipleSpacesCleaner"
    }
]

PREPROCESSING_ALGORITHMS = [
    {
        "class": "textprocessing.cleaner.MailCleaner"
    },
    {
        "class": "textprocessing.cleaner.URLCleaner"
    },
    {
        "class": "textprocessing.cleaner.WordsWithNumbersCleaner"
    },
    {
        "class": "textprocessing.cleaner.WordListCleaner",
        "parameters": {
            "words_path": "C:\\Users\\sasce\\PycharmProjects\\SarcasmOnReddit\\data\\stopwords.txt"
        }
    },
    {
        "class": "textprocessing.cleaner.NonCharCleaner"
    },
    {
        "class": "textprocessing.cleaner.MultipleSpacesCleaner"
    }
]
