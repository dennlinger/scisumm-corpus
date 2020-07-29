import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix

import convenience


if __name__ == "__main__":
    training_sentences, training_targets, training_multitargets, test_sentences, test_targets, test_multitargets = convenience.get_training_and_test_data_multilabel(
        training_ratio=0.7, shuffle=False, balance_dataset=False, balance_number=200)

    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english", ngram_range=(3,8), analyzer="char_wb")),
        ('tfidf', TfidfTransformer()),
        ('clf', DecisionTreeClassifier(criterion="gini", class_weight="balanced")),
    ])

    pipeline.fit(training_sentences, training_multitargets)

    predictions = pipeline.predict(test_sentences)

    print(multilabel_confusion_matrix(test_multitargets, predictions))
    print(metrics.classification_report(test_multitargets, predictions, digits=3))

  