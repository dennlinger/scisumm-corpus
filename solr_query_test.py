from collections import defaultdict
from bs4 import BeautifulSoup
import pysolr
import re

from exploration import get_citances_for_file


if __name__ == "__main__":
    solr = pysolr.Solr('http://localhost:8983/solr/scisumm/', always_commit=True)

    solr.ping()

    citances = get_citances_for_file("C00-2123", list())

    for citance in citances:
        results = defaultdict(float)
        all_query_sentences = citance["Citation Text"]
        soup = BeautifulSoup(all_query_sentences, "html.parser")
        truth = set(citance["Reference Offset"])
        texts = soup.findAll()
        for el in texts:
            clean_text = re.sub(r"([\+\-(&&)\|\|!\(\)\{\}\[\]\^\"\~\*\?:\\\/])", r"\\\1", el.text)
            # print(clean_text)
            res = solr.search(clean_text, fl="id, score")
            for doc in res.docs:
                results[doc["id"]] += doc["score"]

        print(results)
        print(truth)