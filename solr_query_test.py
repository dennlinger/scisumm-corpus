from collections import defaultdict
from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm
import numpy as np
import pysolr
import re
import os
from html import unescape

from exploration import get_citances_for_file


def write_results(query, results, fn, truth, folder="./data/Training-Set-2019/Task1/From-Training-Set-2018/"):
    """
    Writes out the results from a query into a file for reranking training
    """
    base_path = folder + fn
    # replace potential wrong file extension
    xml_filename = fn + ".xml"
    ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
    relevant = ""
    irrelevant = ""
    query = query.replace("\n", "").replace("\t", "")
    counter = 0
    for result in results:
        if counter == 10:
            break
        elif counter > 10:
            raise ValueError("Problem encountered in other writing")
        ref = tree.xpath(".//S[@sid='" + result + "']")
        if ref:
            ref = ref[0].text.replace("\n", "").replace("\t", "")
            # Formatting has to be relevant first, then irrelevant.
            if result in truth:
                relevant += query + "\t" + ref + "\t" + "1\t" + str(result) + "\n"
            else:
                irrelevant += query + "\t" + ref + "\t" + "0\t" + str(result) + "\n"
            counter += 1
        else:
            raise ValueError(f"No results for {result} in {fn}.")

    with open("satya_input_search_only.tsv", "a") as f:
        f.write(relevant)
        f.write(irrelevant)


def write_with_truth_results(query, results, fn, truth, folder="./data/Training-Set-2019/Task1/From-Training-Set-2018/"):
    """
    Instead of the above function, always includes *all* true positive samples as well.
    """
    base_path = folder + fn
    # replace potential wrong file extension
    xml_filename = fn + ".xml"
    ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
    relevant = ""
    irrelevant = ""
    counter = 0
    # Do minimal necessary level of processing
    query = query.replace("\n", "").replace("\t", "")
    # Manually write out relevant ones
    for result in truth:
        counter += 1
        ref = tree.xpath(".//S[@sid='" + result + "']")
        ref = ref[0].text.replace("\n", "").replace("\t", "")
        relevant += query + "\t" + ref + "\t" + "1\t" + str(result) + "\n"
    for result in results:
        # Make sure to cancel after a certain number of results has been reached.
        if counter == 10:
            break
        if counter > 10:
            raise ValueError("Problem encountered!")
        ref = tree.xpath(".//S[@sid='" + result + "']")
        if ref:
            ref = ref[0].text.replace("\n", "").replace("\t", "")
            # Formatting has to be relevant first, then irrelevant.
            if result in truth:
                continue
            else:
                irrelevant += query + "\t" + ref + "\t" + "0\t" + str(result) + "\n"
                counter += 1
        else:
            raise ValueError(f"No results for {result} in {fn}.")

    with open("satya_input_with_truth.tsv", "a") as f:
        f.write(relevant)
        f.write(irrelevant)


def get_top_k_by_weight(d, k):
    """
    Dictionary with value: weight pairs, get the top k values from it by weight.
    :param d:
    :return:
    """
    sorted_dict = sorted(d.items(), key=lambda item: item[1], reverse=True)
    keys = [el[0] for el in sorted_dict[:k]]
    return keys


def get_intersection(res1, res2):
    intermediate_res = {}
    for doc in res1.docs:
        intermediate_res[doc['id']] = doc['score']

    res = {}
    for doc in res2.docs:
        if doc['id'] in intermediate_res.keys():
            res[doc['id']] = intermediate_res[doc['id']] + doc['score']

    keys = get_top_k_by_weight(res, len(res))

    return keys


def get_clean_text(text: str) -> str:
    """
    Preprocessing for query string.
    :param text: Raw input text
    :return: Formulated (and escaped) query
    """

    ####### Citations #########
    # Remove Lastname et al. \ Keep group to potentially keep their name only.
    clean_text = re.sub(r"\(?([A-Za-z]+) et al.(,? \(?[0-9]{4}\)?)?", "", text)

    # Remove "Lastname and Lastname (<year>)","Lastname & Lastname (<year>), "Lastname, Lastname and Lastname (<year>),
    # (Lastname and others <year>)"
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+,? [A-Z][A-Za-z\-]+,? and [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ and [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ &amp[;]* [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ and others, \(?[0-9]{4}\)?", "", clean_text)

    # Remove " (Lastname, <year>)"
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)

    # Remove "Lastname and Lastname(year)"
    clean_text = re.sub(r"[A-Z][A-Za-z\-]+,? and [A-Z][A-Za-z\-]+,?\(?[0-9]{4}\)?", "", clean_text)

    # Remove " [number]" for citations
    clean_text = re.sub(r"\[[0-9]+\]?", "", clean_text)

    ####### Math #########
    clean_text=clean_text.replace("(e.g.", "example")  # otherwise they will identified as functions
    # Starting with O() is a complexity
    clean_text = re.sub(r"O[\s]*\([^\)]+\)", "<COMPLEXITY>", clean_text)
    # Starting with P or Pr () its a probablity
    clean_text = re.sub(r"[\s=][Pp][\s]*[r]*[\s]*\([^\)]+\)", " <PROBABILITY>", clean_text)
    # [numbers] vector
    clean_text = re.sub(r"\[[^\]]+[,][^\]]+\]", "<VECTOR>", clean_text)
    # character[=/]() a function
    clean_text = re.sub(r"[A-Za-z]*[\s]*[=/\d]*[\s]*\([^\)]+[=+][^\)]+\)", "<FUNCTION>", clean_text)

    # Remove any non-ascii character from the query, according to
    # https://stackoverflow.com/a/18430817/3607203
    clean_text = clean_text.encode("ascii", errors="ignore").decode()

    # unescape html characters
    clean_text = unescape(clean_text)

    # Remove Figure and Tables
    clean_text = re.sub(r"Figure [\d]", "", clean_text)
    clean_text = re.sub(r"Tabel [\d]", "", clean_text)

    # remove punctuation
    clean_text = re.sub(r"[;,#]+", "", clean_text)
    clean_text = re.sub(r"[\s]\W[\s]", " ", clean_text)

    # Remove empty brackets
    clean_text = re.sub(r"(\[\]|\(\)|\{\})", "", clean_text)

    # Clean up any left over duplicate spaces
    clean_text = re.sub(r"\s+", " ", clean_text)

    # Replace any left special characters with escaping, necessary for default query parser
    # clean_text = re.sub(r"([\+\-&{2}\|{2}\!\(\)\{\}\[\]\^\"\~\*\?:\\\/])", r"\\\1", clean_text)
    return clean_text


if __name__ == "__main__":
    top_k = 10
    tp = 0
    exact_tp = 0
    overall = 0
    num_samples = 0
    intersect_tp = 0
    retrieved = 0

    np.random.seed(50)

    tp_random = 0
    exact_tp_random = 0

    invalid_queries = 0
    # This is the validation set used by the UoM team (winner 2019)
    # valid = ["C00-2123", "C04-1089", "I05-5011", "J96-3004", "N06-2049", "P05-1004", "P05-1053", "P98-1046"]
    # Enable either ScisummNet or regular training data
    # folder = "./data/Training-Set-2019/Task1/From-Training-Set-2018/"
    folder = "./data/Training-Set-2019/Task1/From-ScisummNet-2019/"
    # for filename in tqdm(valid):  # Used for the validation runs with the UoM data
    for filename in tqdm(sorted(os.listdir(folder))):
        solr = pysolr.Solr('http://localhost:8983/solr/' + filename + '/', always_commit=True)

        solr.ping()

        citances = get_citances_for_file(filename, list(), folder)

        for i, citance in enumerate(citances):
            results = defaultdict(float)
            all_query_sentences = citance["Citation Text"]
            soup = BeautifulSoup(all_query_sentences, "html.parser")
            truth = set(citance["Reference Offset"])
            texts = soup.findAll()
            cleaned = ""
            satya_input_query = ""
            # Add text together from different citation sentences.
            # Results show that a big single query performs better.
            for el in texts:
                print(el)
                cleaned += get_clean_text(el.text)
                satya_input_query += el.text
                print(cleaned+"\n")

            # For empty queries, skip results...
            if not cleaned:
                invalid_queries += 1
                continue
            # This is mostly relevant if scores from multiple queries would be combined.
            res1 = solr.search(cleaned, df="text", fl="id, score", rows=top_k,
                              bf="position_boost", defType="edismax")
            for doc in res1.docs:
                results[doc["id"]] += doc["score"]

            res2 = solr.search(cleaned, df="text2", fl="id, score", rows=top_k,
                              bf="position_boost", defType="edismax")
            for doc in res2.docs:
                results[doc["id"]] += doc["score"]

            intersection = get_intersection(res1, res2)


            # res3 = solr.search(cleaned, df="text3", fl="id, score", rows=top_k,)
            #                   # bf="position_boost", defType="edismax")
            # for doc in res3.docs:
            #     results[doc["id"]] += doc["score"]
            #
            # res4 = solr.search(cleaned, df="text4", fl="id, score", rows=top_k,
            #                   bf="position_boost", defType="edismax")
            # for doc in res4.docs:
            #     results[doc["id"]] += doc["score"]
            #
            # res5 = solr.search(cleaned, df="text5", fl="id, score", rows=top_k,
            #                   bf="position_boost", defType="edismax")
            # for doc in res5.docs:
            #     results[doc["id"]] += doc["score"]

            # TODO: Only write with new results!!

            res = get_top_k_by_weight(results, top_k)
            if len(res) >= 10:
                write_results(satya_input_query, res, filename, truth, folder)
                write_with_truth_results(satya_input_query, res, filename, truth, folder)

            # Total number of samples is equal to annotations in truth.
            overall += len(truth)
            num_samples += 1
            # Calculate coverage of current top k results
            top_k_results = get_top_k_by_weight(results, top_k)
            tp += len(set(top_k_results).intersection(truth))
            intersect_tp += len(set(intersection).intersection(truth))
            retrieved += len(intersection)
            exact_tp += len(set(top_k_results[:len(truth)]).intersection(truth))

            # # Random baseline part.
            # res_random = solr.search("*:*", setRows=0)
            # randoms = np.random.choice(res_random.hits, top_k, replace=False)
            # tp_random += len(set(randoms).intersection(truth))
            # exact_tp_random += len(set(list(randoms)[:len(truth)]).intersection(truth))

    print(f"Recall @ {top_k} for the whole dataset: {tp / overall:.4f}")
    print(f"True positives assuming perfect knowledge about number of relevant: {exact_tp / overall:.4f}")
    # print(f"Random Baseline Recall @ {top_k}: {tp_random / overall:.4f}")

    print(f"Relevant documents: {overall}")
    print(f"Retrieved documents: {top_k * num_samples}")
    print(f"---------------------------------------")
    precision = tp / (top_k * num_samples)
    print(f"Precision: {precision:.4f}")
    recall = tp / overall
    print(f"Recall: {recall:.4f}")
    print(f"F1 overlap: {2 * (precision * recall) / (precision + recall):.4f}")

    print(f"---------------------------------------")
    precision2 = intersect_tp / retrieved
    print(f"Precision for intersection: {precision2:.4f}")
    recall2 = intersect_tp / overall
    print(f"Recall: {recall2:.4f}")
    print(f"F1 overlap: {2 * (precision2 * recall2) / (precision2 + recall2):.4f}")