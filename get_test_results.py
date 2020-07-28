from collections import defaultdict
from lxml import etree
from tqdm import tqdm
import pandas as pd
import pysolr
import os

from solr_query_test import get_intersection, get_top_k_by_weight, get_clean_text


def get_citances_from_csv(filename, citances, folder):
    base_path = os.path.join(folder, filename)
    citances = pd.read_csv(os.path.join(base_path, "annotation", filename + ".csv"), sep=",", quotechar='"')

    return citances


def write_results(query, results, fn, folder):
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
            irrelevant += query + "\t" + ref + "\t" + "0\t" + str(result) + "\n"
            counter += 1
        else:
            raise ValueError(f"No results for {result} in {fn}.")

    with open("test_set.tsv", "a") as f:
        f.write(irrelevant)


if __name__ == "__main__":
    top_k = 4
    folder = "./data/Test-Set-2018/"
    duplicates = 0

    with open("results_intersection_2_fields.tsv", "w") as f:
        f.write("query\tdoc_id\tlabel\n")

    with open("test_set.tsv", "w") as f:
        pass

    for filename in tqdm(sorted(os.listdir(folder))):
        citances = get_citances_from_csv(filename, list(), folder)

        solr = pysolr.Solr('http://localhost:8983/solr/' + filename + '/', always_commit=True)

        solr.ping()

        query_set = set()
        for row in citances["Citation Text Clean"]:
            if row in query_set:
                print(f"Duplicate detected in file {filename}!\n{row}")
                duplicates += 1
                continue
            else:
                query_set.add(row)

            if "\t" in row or "\n" in row:
                raise ValueError("Found tab or newline")
            cleaned = get_clean_text(row)

            results = defaultdict(float)
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

            res = get_top_k_by_weight(results, top_k)
            # Hand over empty set since we don't know the answers.
            # write_results(row, res, filename, folder)

            with open("results_intersection_2_fields.tsv", "a") as f:
                dummy = str([1] * len(intersection))  # Necessary to merge with Satya's results
                if intersection:
                    f.write(row + "\t" + str(intersection) + "\t" + dummy + "\n")
                # If no "consensus" was reached, return the top-1 result.
                else:
                    f.write(row + "\t" + str([res[0]]) + "\t" + '[1]' + "\n")

    print(f"{duplicates} duplicates detected.")
