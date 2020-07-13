from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm
import pysolr
import re
import os
from html import unescape

from exploration import get_citances_for_file


def get_clean_text(text: str) -> str:
    """
    Preprocessing for query parameters.
    :param text:
    :return:
    """

    ####### Citations #########
    # Remove Lastname et al. \ Keep group to potentially keep their name only.
    clean_text = re.sub(r"\(?([A-Za-z]+) et al.(, \(?[0-9]{4}\)?)?", "", text)

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

    # remove punctuation
    clean_text = re.sub(r"[;,#]+", "", clean_text)
    clean_text = re.sub(r"[\s]\*[\s]", " ", clean_text)

    # Clean up any left over duplicate spaces
    clean_text = re.sub(r"\s+", " ", clean_text)

    # Replace any left special characters with escaping
    clean_text = re.sub(r"([\+\-(&&)\|\|!\(\)\{\}\[\]\^\"\~\*\?:\\\/])", r"\\\1", clean_text)

    return clean_text


if __name__ == "__main__":
    top_k = 5
    tp = 0
    exact_tp = 0
    overall = 0
    for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/"))):
        solr = pysolr.Solr('http://localhost:8983/solr/' + filename + '/', always_commit=True)

        solr.ping()

        citances = get_citances_for_file(filename, list())

        for citance in citances:
            results = defaultdict(float)
            all_query_sentences = citance["Citation Text"]
            soup = BeautifulSoup(all_query_sentences, "html.parser")
            truth = set(citance["Reference Offset"])
            texts = soup.findAll()
            cleaned = ""
            for el in texts:
                print(el)
                cleaned = get_clean_text(el.text)
                print(cleaned+"\n")

            res = solr.search(cleaned, fl="id, score", rows=top_k)
            # print(res.docs, len(truth))
            for doc in res.docs:
                results[doc["id"]] += doc["score"]

            # print(results)
            # print(truth)
            # Total number of samples is equal to annotations in truth.
            overall += len(truth)
            # Calculate coverage of current top k results
            tp += len(set(results.keys()).intersection(truth))
            exact_tp += len(set(list(results.keys())[:len(truth)]).intersection(truth))

    print(f"Recall @ {top_k} for the whole dataset: {tp / overall:.4f}")
    print(f"Precision for the whole dataset: {exact_tp / overall:.4f}")

