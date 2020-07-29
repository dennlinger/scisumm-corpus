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

# Section Classes are: abstract, introduction, acknowledgements, related work, methods, conclusion, results, unknown
section_mapping = collections.defaultdict(lambda: "unknown")
section_mapping["abstract"] = "abstract"
section_mapping["introduction"] = "introduction"
section_mapping["acknowledgments"] = "acknowledgements"
section_mapping["related work"] = "related work"
section_mapping["conclusion"] = "conclusion"
section_mapping["experiments"] = "results"
section_mapping["conclusions"] = "conclusion"
section_mapping["acknowledgements"] = "acknowledgements"
section_mapping["evaluation"] = "results"
section_mapping["discussion"] = "conclusion"
section_mapping["results"] = "results"
section_mapping["acknowledgement"] = "acknowledgements"
section_mapping["conclusions and future work"] = "conclusion"
section_mapping["conclusion and future work"] = "conclusion"
section_mapping["background"] = "related work"
section_mapping["experimental results"] = "results"
section_mapping["S"] = "unknown"
section_mapping["previous work"] = "related work"
section_mapping["experimental evaluation"] = "results"
section_mapping["experimental setup"] = "results"
section_mapping["method"] = "methods"
section_mapping["experiments and results"] = "results"
section_mapping["motivation"] = "introduction"
section_mapping["experiment"] = "results"
section_mapping["acknowledgment"] = "acknowledgements"
section_mapping["model"] = "methods"
section_mapping["summary"] = "conclusion"
section_mapping["system description"] = "methods"
section_mapping["analysis"] = "results"
section_mapping["experimentation"] = "results"
section_mapping["decoding"] = "methods"
section_mapping["features"] = "methods"
section_mapping["conclusions and future  work"] = "conclusion"
section_mapping["methodology"] = "methods"
section_mapping["discussion and conclusion"] = "conclusion"
section_mapping["future work"] = "conclusion"
section_mapping["training"] = "methods"
section_mapping["related research"] = "related work"
section_mapping["fertility distribution parameters"] = "methods"
# overview refers in documents either to introduction or related work..
section_mapping["overview"] = "unknown"

section_numbering = {
    "unknown"           : 0,
    "abstract"          : 1,
    "introduction"      : 2,
    "related work"      : 3,
    "methods"           : 4,
    "results"           : 5,
    "conclusion"        : 6,
    "acknowledgements"  : 7,
}


def get_section_numbering(citation_title):
    return section_numbering[section_mapping[citation_title.strip(" .").lower()]]

 
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


sentence_count = collections.defaultdict(lambda: None)


def get_citation_text(citance, clean_text=False):
    global sentence_count
    base_path = "../data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
    # replace potential wrong file extension
    xml_filename = citance["Citing Article"].split(".")[0] + ".xml"
    ref_xml = os.path.join(base_path, "Citance_XML", xml_filename)
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
    root = tree.getroot()

    file_sentence_count = sentence_count[xml_filename]
    if not file_sentence_count:
        sentence_count[xml_filename] = max([int(el.attrib["sid"]) for el in root.xpath(".//S")])
        file_sentence_count = sentence_count[xml_filename]


    citation_text = []
    citation_titles = []
    if type(citance["Citation Offset"]) == str:
        el = root.xpath(".//S[@sid='" + citance["Citation Offset"] + "']")
        sid = int(citance["Citation Offset"])
        if clean_text:
            citation_text.append(get_clean_text(el[0].text))
        else:
            citation_text.append(el[0].text)

        # get section title
        parent = el[0].getparent()
        try:
            citation_titles.append(parent.attrib["title"])
        except KeyError:
            citation_titles.append(parent.tag)

    else:
        sid = 0
        for offset in citance["Citation Offset"]:
            el = root.xpath(".//S[@sid='" + offset + "']")
            sid += int(offset)
            if clean_text:
                citation_text.append(get_clean_text(el[0].text))
            else:
                citation_text.append(el[0].text)

            # get section title
            parent = el[0].getparent()
            try:
                citation_titles.append(parent.attrib["title"])
            except KeyError:
                citation_titles.append(parent.tag)
        sid /= len(citance["Citation Offset"])

    citation_text = " ".join(citation_text)

    relative_sid_pos = sid / file_sentence_count

    return citation_text, citation_titles[0], relative_sid_pos


def get_reference_text(citance, clean_text=False):
    base_path = "../data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
    # replace potential wrong file extension
    xml_filename = citance["Reference Article"].split(".")[0] + ".xml"
    ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
    root = tree.getroot()

    reference_text = []
    reference_titles = []
    if type(citance["Reference Offset"]) == str:
        el = root.xpath(".//S[@sid='" + citance["Reference Offset"] + "']")
        if clean_text:
            reference_text.append(get_clean_text(el[0].text))
        else:
            reference_text.append(el[0].text)

        # get section title
        parent = el[0].getparent()
        try:
            reference_titles.append(parent.attrib["title"])
        except KeyError:
            reference_titles.append(parent.tag)

    else:
        for offset in citance["Reference Offset"]:
            el = root.xpath(".//S[@sid='" + offset + "']")
            if clean_text:
                reference_text.append(get_clean_text(el[0].text))
            else:
                reference_text.append(el[0].text)

            # get section title
            parent = el[0].getparent()
            try:
                reference_titles.append(parent.attrib["title"])
            except KeyError:
                reference_titles.append(parent.tag)

    reference_text = " ".join(reference_text)

    return reference_text


def get_training_and_test_data_multilabel(training_ratio=0.7, clean_text=False, shuffle=False, balance_dataset=False, balance_number=40):
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

    training_reference_sentences = []
    training_citance_sentences = []
    training_titles = []
    training_sid = []
    training_multitargets = []

    for annotation in training_annotations:
        for citance in annotation:
            reference_text = get_reference_text(citance, clean_text)
            citation_text, citation_title, cit_relative_sid_pos = get_citation_text(citance, clean_text)
            facets = list(citance["Discourse Facet"])

            training_titles.append(get_section_numbering(citation_title))
            training_reference_sentences.append(reference_text)
            training_citance_sentences.append(citation_text)
            training_sid.append(cit_relative_sid_pos)

            multitarget = []
            for cat in categories:
                if cat in facets:
                    multitarget.append(1)
                else:
                    multitarget.append(0)
            training_multitargets.append(multitarget)

    if balance_dataset:
        counts = collections.defaultdict(int)
        temp_training_ref_sentences = []
        temp_training_cit_sentences = []
        temp_training_sid = []
        temp_training_multitargets = []

        if shuffle:
            shuffle_list = list(zip(training_reference_sentences, training_citance_sentences, training_sid, training_multitargets))
            random.shuffle(shuffle_list)
            training_reference_sentences, training_citance_sentences, training_sid, training_multitargets = zip(*shuffle_list)
        
        for i in range(len(training_reference_sentences)):
            check_balance = True
            for j in range(5):
                # if a sentence is labeled with a class that already has enough samples, skip
                if training_multitargets[i][j] == 1:
                    if counts[j] > balance_number:
                        check_balance = False
                    else:
                        counts[j] += 1
            if not check_balance:
                continue

            temp_training_ref_sentences.append(training_reference_sentences[i])
            temp_training_cit_sentences.append(training_citance_sentences[i])
            temp_training_sid.append(training_sid[i])
            temp_training_multitargets.append(training_multitargets[i])

        training_reference_sentences = temp_training_ref_sentences
        training_citance_sentences = temp_training_cit_sentences
        training_sid = temp_training_sid
        training_multitargets = temp_training_multitargets


    test_reference_sentences = []
    test_citance_sentences = []
    test_titles = []
    test_sid = []
    test_multitargets = []

    for annotation in test_annotations:
        for citance in annotation:
            reference_text = get_reference_text(citance, clean_text)
            citation_text, citation_title, cit_relative_sid_pos  = get_citation_text(citance, clean_text)
            facets = list(citance["Discourse Facet"])

            test_titles.append(section_numbering[section_mapping[citation_title]])
            test_reference_sentences.append(reference_text)
            test_citance_sentences.append(citation_text)
            test_sid.append(cit_relative_sid_pos)

            multitarget = []
            for cat in categories:
                if cat in facets:
                    multitarget.append(1)
                else:
                    multitarget.append(0)
            test_multitargets.append(multitarget)

    return training_reference_sentences, training_citance_sentences, training_titles, training_sid, training_multitargets, test_reference_sentences, test_citance_sentences, test_titles, test_sid, test_multitargets


def get_training_and_test_data_unilabel(training_ratio=0.7, clean_text=False, shuffle=False):
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
    training_targets = []

    for annotation in training_annotations:
        for citance in annotation:
            reference_text = get_reference_text(citance, clean_text)
            citation_text, citation_title = get_citation_text(citance, clean_text)
            facets = list(citance["Discourse Facet"])

            training_sentences.append(section_mapping[citation_title] + " " + reference_text + " " + citation_text)

            training_targets.append(cat_id[facets[0]])

    test_sentences = []
    test_targets = []

    for annotation in test_annotations:
        for citance in annotation:
            reference_text = get_reference_text(citance, clean_text)
            citation_text, citation_title = get_citation_text(citance, clean_text)
            facets = list(citance["Discourse Facet"])

            test_sentences.append(section_mapping[citation_title] + " " + reference_text + " " + citation_text)
                
            test_targets.append(cat_id[facets[0]])
            

    return training_sentences, training_targets, test_sentences, test_targets