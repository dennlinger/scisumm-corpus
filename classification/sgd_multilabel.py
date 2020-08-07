import os

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import FunctionTransformer
import joblib
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import convenience


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
         
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in stopwords.words('english')]


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
        training_ratio=0.8, shuffle=True, balance_dataset=False, balance_number=100)

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

            ("cit_word_features",
            Pipeline([
                ("selector", column_selector("cit_sentences")),
                ("vect", CountVectorizer(ngram_range=(1,2), analyzer="word", tokenizer=LemmaTokenizer())),
                ("tfidf", TfidfTransformer()),
            ])),

            ("section_title_features",
            Pipeline([
                ("selector", sid_selector()),
            ])
            )
        ])),
        
        ("clf", OneVsRestClassifier(SGDClassifier(loss="perceptron", penalty="elasticnet",
                                    alpha=1e-3, max_iter=1000, tol=None, l1_ratio=0.4))),
    ])

    pipeline.fit(training_data, training_multitargets)

    predictions = pipeline.predict(test_data)

    print("SGD Multilabel")
    print(multilabel_confusion_matrix(test_multitargets, predictions))
    print(metrics.classification_report(test_multitargets, predictions, digits=3))

    predictions = pipeline.predict(test_data)

    # print("SGD Multilabel")
    # print(multilabel_confusion_matrix(test_multitargets, predictions))
    # print(metrics.classification_report(test_multitargets, predictions, digits=3))

    joblib.dump(pipeline, "sgd_multilabel2.model")

    # new_model = joblib.load("sgd_multilabel.model")

    # predictions = new_model.predict(test_data)

    # print("SGD Multilabel")
    # print(multilabel_confusion_matrix(test_multitargets, predictions))
    # print(metrics.classification_report(test_multitargets, predictions, digits=3))


