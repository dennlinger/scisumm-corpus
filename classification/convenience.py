from typing import List
import os
import re
from lxml import etree
import random
import collections


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

id2cat = dict()
id2cat[1] = "aim_citation"
id2cat[2] = "hypothesis_citation"
id2cat[3] = "implication_citation"
id2cat[4] = "method_citation"
id2cat[5] = "results_citation"

 
def get_citances_for_file(file_id: str, citances_json: List) -> List:
    base_path = "../data/Training-Set-2019/Task1/From-Training-Set-2018/" + file_id

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


def get_all_citances():
    citances = []
    for filename in sorted(os.listdir("../data/Training-Set-2019/Task1/From-Training-Set-2018/")):
        citances = get_citances_for_file(filename, citances)
    return citances


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

    # Replace any left special characters with escaping
    clean_text = re.sub(r"([\+\-(&&)\|\|!\(\)\{\}\[\]\^\"\~\*\?:\\\/])", r"\\\1", clean_text)

    return clean_text


def get_citation_text(citance, clean_text=False):
    base_path = "../data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
    # replace potential wrong file extension
    xml_filename = citance["Citing Article"].split(".")[0] + ".xml"
    ref_xml = os.path.join(base_path, "Citance_XML", xml_filename)
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
    root = tree.getroot()

    citation_text = []
    if type(citance["Citation Offset"]) == str:
        el = root.xpath(".//S[@sid='" + citance["Citation Offset"] + "']")
        if clean_text:
            citation_text.append(get_clean_text(el[0].text))
        else:
            citation_text.append(el[0].text)
    else:
        for offset in citance["Citation Offset"]:
            el = root.xpath(".//S[@sid='" + offset + "']")
            if clean_text:
                citation_text.append(get_clean_text(el[0].text))
            else:
                citation_text.append(el[0].text)
    citation_text = " ".join(citation_text)

    return citation_text


def get_reference_text(citance, clean_text=False):
    base_path = "../data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
    # replace potential wrong file extension
    xml_filename = citance["Reference Article"].split(".")[0] + ".xml"
    ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
    root = tree.getroot()

    reference_text = []
    if type(citance["Reference Offset"]) == str:
        el = root.xpath(".//S[@sid='" + citance["Reference Offset"] + "']")
        if clean_text:
            reference_text.append(get_clean_text(el[0].text))
        else:
            reference_text.append(el[0].text)
    else:
        for offset in citance["Reference Offset"]:
            el = root.xpath(".//S[@sid='" + offset + "']")
            if clean_text:
                reference_text.append(get_clean_text(el[0].text))
            else:
                reference_text.append(el[0].text)
    reference_text = " ".join(reference_text)

    return reference_text


def get_training_and_test_data(training_ratio=0.7, clean_text=False, shuffle=False, balance_dataset=False, balance_number=40):
    """
    For all annotations get the reference and citation sentences as well as the label (discourse facet) 
    """
    annotations = []
    for filename in sorted(os.listdir("../data/Training-Set-2019/Task1/From-Training-Set-2018/")):
        annotations.append(get_citances_for_file(filename, []))

    if shuffle:
        random.shuffle(annotations)

    training_size = round(len(annotations) * training_ratio)

    training_annotations = annotations[:training_size]
    test_annotations = annotations[training_size:]

    training_sentences = []
    training_targets = dict()
    for cat in categories:
        training_targets[cat] = []

    for annotation in training_annotations:
        for citance in annotation:
            reference_text = get_reference_text(citance, clean_text)
            citation_text = get_citation_text(citance, clean_text)
            facets = list(citance["Discourse Facet"])

            training_sentences.append(reference_text + " " + citation_text)

            for cat in categories:
                if cat in facets:
                    training_targets[cat].append(1)
                else:
                    training_targets[cat].append(0)

    test_sentences = []
    test_targets = dict()
    for cat in categories:
        test_targets[cat] = []

    for annotation in test_annotations:
        for citance in annotation:
            reference_text = get_reference_text(citance, clean_text)
            citation_text = get_citation_text(citance, clean_text)
            facets = list(citance["Discourse Facet"])

            test_sentences.append(reference_text + " " + citation_text)

            for cat in categories:
                if cat in facets:
                    test_targets[cat].append(1)
                else:
                    test_targets[cat].append(0)

    return training_sentences, training_targets, test_sentences, test_targets
