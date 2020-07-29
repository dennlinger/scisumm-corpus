import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

import convenience 


if __name__ == "__main__":
    training_sentences, training_targets, test_sentences, test_targets = convenience.get_training_and_test_data_unilabel(
        training_ratio=0.7, shuffle=False)

    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english", ngram_range=(1,2), analyzer="word")),
        ('tfidf', TfidfTransformer()),
        ('clf', DecisionTreeClassifier(criterion="gini", class_weight="balanced")),
    ])

    pipeline.fit(training_sentences, training_targets)

    predictions = pipeline.predict(test_sentences)

    print(metrics.confusion_matrix(test_targets, predictions))
    print(metrics.classification_report(test_targets, predictions, digits=3))

  