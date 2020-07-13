import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

import convenience


if __name__ == "__main__":
    training_sentences, training_targets, test_sentences, test_targets = convenience.get_training_and_test_data()

    pipelines = dict()
    
    for cat in convenience.categories:
        pipelines[cat] = Pipeline([
                        ('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', OneVsRestClassifier(DecisionTreeClassifier())),
                        ])

    for cat in convenience.categories:
        pipelines[cat].fit(training_sentences, training_targets[cat])

    predictions = dict()
    
    for cat in convenience.categories:
        cat_predictions = pipelines[cat].predict(test_sentences)

        predictions[cat] = [0] * len(cat_predictions)

        for i, pred in enumerate(cat_predictions):
            predictions[cat][i] = pred

    # print(category)
    # print(metrics.confusion_matrix(test_targets[category], predicted))
    # print(metrics.classification_report(test_targets[category], predicted, digits=3))

    results = []
    results_multi = []

    for i in range(len(test_sentences)):
        temp_results = [0] * 5
        num_labels = 0

        for j, cat in enumerate(convenience.categories):
            if test_targets[cat][i]:
                num_labels += 1
            if predictions[cat][i] == test_targets[cat][i]:
                temp_results[j] = 1
            if all(temp_results):
                results.append(1)
            else:
                results.append(0)
            if num_labels > 1:
                if all(temp_results):
                    results_multi.append(1)
                else:
                    results_multi.append(0)

    print("Accuracy all:", results.count(1) / len(results))
    print("Accuracy only for sentences with multiple labels:", results_multi.count(1) / len(results_multi))