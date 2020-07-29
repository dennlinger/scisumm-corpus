import os

import pandas as pd
import joblib
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import xml.etree.ElementTree as ET


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


def read_annotation_files(base_path: str, sub_folder_structure=True):
    files = os.listdir(base_path)

    annotations = dict()

    if sub_folder_structure:
        for file in files:
            with open(os.path.join(base_path, file, "annotation", file + ".csv"), "r") as f:
                data = pd.read_csv(f, index_col=0)
                annotations[file + ".csv"] = data
    else:
        for file in files:
            with open(os.path.join(base_path, file), "r") as f:
                data = pd.read_csv(f, index_col=0)
                annotations[file] = data

    return annotations


def load_model(name: str):
    return joblib.load(name)


if __name__ == "__main__":
    id2cat = dict()
    id2cat[0] = "'aim_citation'"
    id2cat[1] = "'hypothesis_citation'"
    id2cat[2] = "'implication_citation'"
    id2cat[3] = "'method_citation'"
    id2cat[4] = "'results_citation'"

    # Test annotations
    # base_path = "../data/Test-Set-2018"
    base_path = "../runs/intersection_2_field/Task1"
    write_path = "../runs/intersection_2_field/Task1b"
    annotations = read_annotation_files(base_path, sub_folder_structure=False)

    # pre-trained model
    model = load_model("sgd_multilabel.model")

    s = list()

    # Get result for each annotation
    for key, annotation in annotations.items():
        # Get reference sentences
        reference_sentences = []

        for ref in annotation["Reference Text"]:
            reference_sentences.append("<r>" + ref + "</r>")

        for j in range(len(reference_sentences)):
            xml = ET.fromstring(reference_sentences[j])
            reference_sentences[j] = " ".join([child.text for child in xml])

        annotation["ref_sentences"] = reference_sentences

        # Rename column for citation sentences as expected by the model
        annotation = annotation.rename({"Citation Text Clean": "cit_sentences"}, axis="columns")

        # Predict using the model
        predictions = model.predict(annotation)

        discourse_facets = []
        for pred in predictions:
            categories = []
            for j in range(len(pred)):
                if pred[j]:
                    categories.append(id2cat[j])
            discourse_facets.append("[" + ",".join(categories) + "]")

        annotation["Discourse Facet"] = discourse_facets

        for d in discourse_facets:
            s.append(d)

        # clean up csv
        annotation = annotation.drop("ref_sentences", axis="columns")
        annotation = annotation.rename({"cit_sentences": "Citation Text Clean"}, axis="columns")

        annotations[key] = annotation

    for key, annotation in annotations.items():
        annotation.to_csv(os.path.join(write_path, key))

