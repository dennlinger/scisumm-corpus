import os

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics

import convenience 


if __name__ == "__main__":
    training_sentences, training_targets, test_sentences, test_targets = convenience.get_training_and_test_data_unilabel(
        training_ratio=0.8, shuffle=False)

    pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english", ngram_range=(1,2), analyzer="word")),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='elasticnet',
                          alpha=1e-3, max_iter=10, tol=None)),
    ])

    pipeline.fit(training_sentences, training_targets)

    predictions = pipeline.predict(test_sentences)

    print(metrics.confusion_matrix(test_targets, predictions))
    print(metrics.classification_report(test_targets, predictions, digits=3))

  