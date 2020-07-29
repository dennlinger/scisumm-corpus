import os

import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import convenience

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
         
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class column_selector():
    def __init__(self, column=None):
            self.column = column

    def transform(self, input_df, **transform_params):
        return list(input_df[self.column])

    def fit(self, X, y=None, **fit_params):
        return self


class section_title_selector():
    def transform(self, input_df, **transform_params):
        result = np.array(input_df["section_titles"])
        result = result.reshape(len(result), 1)
        return result

    def fit(self, X, y=None, **fit_params):
        return self


class sid_selector():
    def transform(self, input_df, **transform_params):
        result = np.array(input_df["sid"])
        result = result.reshape(len(result), 1)
        return result

    def fit(self, X, y=None, **fit_params):
        return self


if __name__ == "__main__":
    training_reference_sentences, training_citance_sentences, training_titles, training_sid, training_multitargets, test_reference_sentences, test_citance_sentences, test_titles, test_sid, test_multitargets = convenience.get_training_and_test_data_multilabel(
        training_ratio=0.8, shuffle=True, balance_dataset=False, balance_number=200)

    training_data = pd.DataFrame([(t1, t2, t3, t4) for t1, t2, t3, t4 in zip(training_reference_sentences, training_citance_sentences, training_sid, training_titles)], columns=["ref_sentences", "cit_sentences", "sid", "section_titles"])
    test_data = pd.DataFrame([(t1, t2, t3, t4) for t1, t2, t3, t4 in zip(test_reference_sentences, test_citance_sentences, test_sid, test_titles)], columns=["ref_sentences", "cit_sentences", "sid", "section_titles"])

    pipeline = Pipeline([
        ("features", FeatureUnion([
            ("ref_word_features",
            Pipeline([
                ("selector", column_selector("ref_sentences")),
                ("vect", CountVectorizer(ngram_range=(1,2), analyzer="word", tokenizer=LemmaTokenizer())),
                ("tfidf", TfidfTransformer()),
            ])),

            # ("cit_word_features",
            # Pipeline([
            #     ("selector", column_selector("cit_sentences")),
            #     ("vect", CountVectorizer(ngram_range=(1,2), analyzer="word", tokenizer=LemmaTokenizer())),
            #     ("tfidf", TfidfTransformer()),
            # ])),

            # ("section_title_features",
            # Pipeline([
            #     ("selector", sid_selector()),
            # ])
            # )
        ])),
        
        ("clf", OneVsRestClassifier(RandomForestClassifier(n_estimators=10, class_weight="balanced"))),
    ])

    pipeline.fit(training_data, training_multitargets)

    predictions = pipeline.predict(test_data)

    print("Random Forest Multilabel")
    print(multilabel_confusion_matrix(test_multitargets, predictions))
    print(metrics.classification_report(test_multitargets, predictions, digits=3))
