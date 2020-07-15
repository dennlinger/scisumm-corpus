from collections import OrderedDict
from functools import lru_cache
from bs4 import BeautifulSoup
from typing import Dict
from lxml import etree
from tqdm import tqdm
import numpy as np
import pickle
import spacy
import os
from scipy.spatial.distance import cosine


def get_citances_for_file(file_id: str, citances_json: Dict) -> Dict:
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

        citances_json[citance_dict["Citance Number"]] = citance_dict

    return citances_json

@lru_cache(1)
def get_spacy():
    return spacy.load("en_core_web_lg")


def create_embeddings(filename):
    nlp = get_spacy()

    # Write reference doc first
    base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + filename
    ref_xml = os.path.join(base_path, "Reference_XML", filename + ".xml")
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))

    sentences = tree.xpath(".//S")

    reference_sentences = OrderedDict()

    for sentence in sentences:
        doc = nlp(sentence.text)
        reference_sentences[sentence.attrib["sid"]] = doc.vector

    with open(os.path.join("./word2vec", filename + "_reference.pkl"), "wb") as f:
        pickle.dump(reference_sentences, f)

    citances = get_citances_for_file(filename, dict())
    citance_sentences = OrderedDict()
    for citance in citances.values():
        all_query_sentences = citance["Citation Text"]
        soup = BeautifulSoup(all_query_sentences, "html.parser")
        texts = soup.findAll()
        cleaned = ""

        # Add text together from different citation sentences.
        # Results show that a big single query performs better.
        for el in texts:
            cleaned += el.text

        doc = nlp(cleaned)
        citance_sentences[citance["Citance Number"]] = doc.vector

    with open(os.path.join("./word2vec", filename + "_citance.pkl"), "wb") as f:
        pickle.dump(citance_sentences, f)


if __name__ == "__main__":
    top_k = 10
    tp = 0
    exact_tp = 0
    overall = 0

    np.random.seed(50)

    # Check that subfolder exists for pickled files.
    os.makedirs("./word2vec", exist_ok=True)

    for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/"))):

        if not os.path.isfile(os.path.join("./word2vec", filename + "_reference.pkl")):
            create_embeddings(filename)

        with open(os.path.join("./word2vec", filename + "_reference.pkl"), "rb") as f:
            ref = pickle.load(f)
        with open(os.path.join("./word2vec", filename + "_citance.pkl"), "rb") as f:
            cite = pickle.load(f)

        citances = get_citances_for_file(filename, dict())

        for i, (k, vec) in enumerate(cite.items()):
            truth = citances[k]["Reference Offset"]
            distances = []
            for sid, sentence in ref.items():
                distances.append((sid, cosine(sentence, vec)))

            distances.sort(key=lambda row: row[1])

            results = [el[0] for el in distances[:top_k]]

            # Total number of samples is equal to annotations in truth.
            overall += len(truth)
            tp += len(set(results).intersection(truth))
            exact_tp += len(set(list(results)[:len(truth)]).intersection(truth))

    print(f"Recall @ {top_k} for the whole dataset: {tp / overall:.4f}")
    print(f"Precision for the whole dataset: {exact_tp / overall:.4f}")

