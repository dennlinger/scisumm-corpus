from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm
import pysolr
import re
import os

from exploration import get_citances_for_file


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
            clean_text = ""
            for el in texts:
                clean_text += re.sub(r"([\+\-(&&)\|\|!\(\)\{\}\[\]\^\"\~\*\?:\\\/])", r"\\\1", el.text)
                # print(clean_text)

            res = solr.search(clean_text, fl="id, score", rows=top_k)
            print(res.docs, len(truth))
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

