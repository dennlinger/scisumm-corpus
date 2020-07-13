import spacy
import os
from typing import List
from lxml import etree
import random
from collections import Counter
import re


def get_citances_for_file(file_id: str, citances_json: List) -> List:
    base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + file_id

    try:
        annotations_file = os.path.join(base_path, "annotation", file_id + ".ann.txt")
        with open(annotations_file) as f:
            annotations = f.readlines()
    except FileNotFoundError:
        annotations_file = os.path.join(base_path, "annotation", file_id + ".annv3.txt")
        with open(annotations_file) as f:
            annotations = f.readlines()

    citances = []
    for line in annotations:
        if line.strip():
            citances.append(line.strip("\n |").split(" | "))

    for citance in citances:
        citance_dict = {}

        for el in citance:
            # Only split at first colon, since text may contain more.
            k, v = el.split(":", maxsplit=1)
            k = k.strip(" ")
            v = v.strip(" ")
            if k in ("Citation Marker Offset", "Citation Offset", "Reference Offset"):
                citance_dict[k] = eval(v)

            # Merge Discourse Facets to consistent naming
            elif k == "Discourse Facet":
                if v.strip(" ")[0] == "[":
                     temp_facets = eval(v)
                else:
                    temp_facets = [v]

                temp_facets = [facet.lower().replace(" ", "_").replace("result_", "results_") for facet in temp_facets]
                citance_dict[k] = temp_facets

            else:
                citance_dict[k] = v

        citances_json.append(citance_dict)

    return citances_json


def get_clean_text(text: str) -> str:
    """
    Preprocessing for query parameters.
    :param text:
    :return:
    """
    # Remove Lastname et al. \ Keep group to potentially keep their name only.
    clean_text = re.sub(r"\(?([A-Za-z]+) et al.(, \(?[0-9]{4}\)?)?", "", text)

    # Remove "Lastname and Lastname (<year>)"
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ and [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)

    # TODO: Evaluate if replacing it with "translated" characters would be better?
    # Remove HTML special characters
    clean_text = re.sub(r"\&[a-z]{4};", "", clean_text)

    # Clean up any left over duplicate spaces
    clean_text = re.sub(r"\s+", " ", clean_text)

    # Remove any non-ascii character from the query, according to
    # https://stackoverflow.com/a/18430817/3607203
    clean_text = clean_text.encode("ascii", errors="ignore").decode()

    # Don't mask the special characters here, since solr will take care of it.

    return clean_text


if __name__ == "__main__":
    citances = []
    for filename in sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/")):
        citances = get_citances_for_file(filename, citances)

    data = []

    for citance in citances:
        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Citing Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Citance_XML", xml_filename)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()

        citation_text = []
        if type(citance["Citation Offset"]) == str:
            el = root.xpath(".//S[@sid='" + citance["Citation Offset"] + "']")
            citation_text.append(get_clean_text(el[0].text))
        else:
            for offset in citance["Citation Offset"]:
                el = root.xpath(".//S[@sid='" + offset + "']")
                citation_text.append(get_clean_text(el[0].text))
        citation_text = " ".join(citation_text)

        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Reference Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()

        reference_text = []
        if type(citance["Reference Offset"]) == str:
            el = root.xpath(".//S[@sid='" + citance["Reference Offset"] + "']")
            reference_text.append(get_clean_text(el[0].text))
        else:
            for offset in citance["Reference Offset"]:
                el = root.xpath(".//S[@sid='" + offset + "']")
                reference_text.append(get_clean_text(el[0].text))
        reference_text = " ".join(reference_text)

        data.append((citation_text, reference_text, citance["Discourse Facet"]))

    cites = set([d[0] for d in data])
    refs  = set([d[1] for d in data])

    print(len(data), len(cites), len(refs))
        
    lengths = []
    for dat in data:
        lengths.append(len(dat[2]))

    print(Counter(lengths).most_common())
    
    categories = [  "aim_citation", 
                    "hypothesis_citation", 
                    "implication_citation", 
                    "method_citation", 
                    "results_citation"  ]

    cat_id = dict()
    cat_id["aim_citation"] = 1
    cat_id["hypothesis_citation"] = 2
    cat_id["implication_citation"] = 3
    cat_id["method_citation"] = 4
    cat_id["results_citation"] = 5

random.shuffle(data)

training_data = data[:round(len(data)*0.7)]
test_data = data[round(len(data)*0.7):]

train_sentences = []
train_targets = []
for dat in training_data:
    # train_sentences.append(dat[0] + " " + dat[1])
    train_sentences.append(dat[1])
    train_targets.append(cat_id[dat[2][0]])

test_sentences = []
test_targets = []
for dat in test_data:
    # test_sentences.append(dat[0] + " " + dat[1])
    test_sentences.append(dat[1])
    test_targets.append(cat_id[dat[2][0]])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='elasticnet',
                          alpha=1e-3, max_iter=10, tol=None)),
    ])

text_clf.fit(train_sentences, train_targets)

predicted = text_clf.predict(test_sentences)

np.mean(predicted == test_targets)

print(metrics.confusion_matrix(test_targets, predicted))

print(metrics.classification_report(test_targets, predicted, digits=3))


# MULTI LABEL

training_data = data[:round(len(data)*0.7)]
test_data = data[round(len(data)*0.7):]

train_sentences = []
train_targets = dict()
for cat in categories:
    train_targets[cat] = [0] * len(training_data)
for i, dat in enumerate(training_data):
    # train_sentences.append(dat[0] + " " + dat[1])
    train_sentences.append(dat[1])
    for cat in dat[2]:
        train_targets[cat][i] = 1

test_sentences = []
test_targets = dict()
test_results = []
for cat in categories:
    test_targets[cat] = [0] * len(test_data)
for i, dat in enumerate(test_data):
    # test_sentences.append(dat[0] + " " + dat[1])
    test_sentences.append(dat[1])
    for cat in dat[2]:
        test_targets[cat][i] = 1


text_multilabel = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='elasticnet',
                                alpha=1e-3, max_iter=10, tol=None))),
    ])

test_results = [dict() for i in range(len(test_data))]

for category in categories:
    text_multilabel.fit(train_sentences, train_targets[category])

    predicted = text_multilabel.predict(test_sentences)

    for i, p in enumerate(predicted):
        test_results[i][category] = p

    # print(category)
    # print(metrics.confusion_matrix(test_targets[category], predicted))
    # print(metrics.classification_report(test_targets[category], predicted, digits=3))

results = []
results_multi = []
for i in range(len(test_results)):
    temp_results = [0] * 5
    num_labels = 0
    for j, cat in enumerate(categories):
        if test_targets[cat][i]:
            num_labels += 1
        if test_results[i][cat] == test_targets[cat][i]:
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

print("Accuracy:", results.count(1) / len(results))
print("Accuracy:", results_multi.count(1) / len(results_multi))


################################
# group by reference sentences
################################

data = dict()
facets = []
for citance in citances:
        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Citing Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Citance_XML", xml_filename)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()

        citation_text = []
        if type(citance["Citation Offset"]) == str:
            el = root.xpath(".//S[@sid='" + citance["Citation Offset"] + "']")
            citation_text.append(el[0].text)
            # citation_text.append(get_clean_text(el[0].text))
        else:
            for offset in citance["Citation Offset"]:
                el = root.xpath(".//S[@sid='" + offset + "']")
                citation_text.append(el[0].text)
                # citation_text.append(get_clean_text(el[0].text))
        citation_text = " ".join(citation_text)

        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Reference Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()

        reference_text = []
        if type(citance["Reference Offset"]) == str:
            el = root.xpath(".//S[@sid='" + citance["Reference Offset"] + "']")
            reference_text.append(el[0].text)
            # reference_text.append(get_clean_text(el[0].text))
        else:
            for offset in citance["Reference Offset"]:
                el = root.xpath(".//S[@sid='" + offset + "']")
                reference_text.append(el[0].text)
                # reference_text.append(get_clean_text(el[0].text))
        reference_text = " ".join(reference_text)

        data[reference_text] = (citation_text, citance["Discourse Facet"])
        facets.append(citance["Discourse Facet"])

training_size = round(len(data) * 0.7)
test_size = len(data) - training_size
# training_keys = random.sample(data.keys(), training_size)
training_keys = list(data.keys())[:training_size]

training_data = dict()
test_data = []
for cat in categories:
    training_data[cat] = []
for key in data.keys():
    if key in training_keys:
        for cat in data[key][1]:
            training_data[cat].append((key, data[key][0], data[key][1]))
    else:
        test_data.append((key, data[key][0], data[key][1]))

for cat in categories:
    # random.shuffle(training_data[cat])
    training_data[cat] = training_data[cat][:100]

training_data = [item for sublist in training_data.values() for item in sublist]

train_sentences = []
train_targets = dict()
for cat in categories:
    train_targets[cat] = [0] * len(training_data)
for i, dat in enumerate(training_data):
    train_sentences.append(dat[0] + " " + dat[1])
    # train_sentences.append(dat[1])
    for cat in dat[2]:
        train_targets[cat][i] = 1

test_sentences = []
test_targets = dict()
test_results = []
for cat in categories:
    test_targets[cat] = [0] * len(test_data)
for i, dat in enumerate(test_data):
    test_sentences.append(dat[0] + " " + dat[1])
    # test_sentences.append(dat[1])
    for cat in dat[2]:
        test_targets[cat][i] = 1


text_multilabel = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    # ('clf', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='elasticnet',
                                # alpha=1e-3, max_iter=10, tol=None))),
    # ('clf', OneVsRestClassifier(SVC(kernel="linear", degree=2)))
    ("clf", DecisionTreeClassifier())
    ])

test_results = [dict() for i in range(len(test_data))]

for category in categories:
    text_multilabel.fit(train_sentences, train_targets[category])

    predicted = text_multilabel.predict(test_sentences)

    for i, p in enumerate(predicted):
        test_results[i][category] = p

    # print(category)
    # print(metrics.confusion_matrix(test_targets[category], predicted))
    # print(metrics.classification_report(test_targets[category], predicted, digits=3))

results = []
results_multi = []
for i in range(len(test_results)):
    temp_results = [0] * 5
    num_labels = 0
    for j, cat in enumerate(categories):
        if test_targets[cat][i]:
            num_labels += 1
        if test_results[i][cat] == test_targets[cat][i]:
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

print("Accuracy:", results.count(1) / len(results))
print("Accuracy:", results_multi.count(1) / len(results_multi))