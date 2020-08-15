from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from html import unescape

from exploration import get_citances_for_file

if __name__ == "__main__":
    valid = ["C00-2123", "C04-1089", "I05-5011", "J96-3004", "N06-2049", "P05-1004", "P05-1053", "P98-1046"]
    folder = "./data/Training-Set-2019/Task1/From-Training-Set-2018/"

    result_dict = {}
    data = pd.read_csv("doc_ids_dev_ground_truth_3_new.csv", delimiter="\t")

    tp = 0
    overall_truth = 0
    overall_predicted = 0
    count_errors = 0
    count_citances = 0

    for i, row in data.iterrows():
        # unescape html characters
        clean_text = BeautifulSoup(row["query"]).text
        if clean_text.startswith("Levin"):
            clean_text = clean_text.replace("Levin&aposs", "Levin's")
        elif clean_text.startswith("In explormg these quest1ons"):
            clean_text = "In explormg these quest1ons"

        doc_ids = [str(el) for el in eval(row["doc_id"])]
        labels = eval(row["label"])
        similarities = eval(row["similiarity"])
        # Use for thresholding
        # relevant_doc_ids = set()
        # for i in range(len(doc_ids)):
        #     if similarities[i] <= 0.3:
        #         relevant_doc_ids.add(doc_ids[i])

        # Fixed return of the highest 4 similarity scores "Always 4"
        sorted_doc_ids = [x for _, x in sorted(zip(similarities, doc_ids), reverse=True)]
        relevant_doc_ids = set(sorted_doc_ids[:4])
        result_dict[clean_text] = relevant_doc_ids

    for filename in tqdm(valid):  # Used for the validation runs with the UoM data

        citances = get_citances_for_file(filename, list(), folder)

        for i, citance in enumerate(citances, start=1):
            count_citances += 1
            all_query_sentences = citance["Citation Text"]
            soup = BeautifulSoup(all_query_sentences, "html.parser")
            truth = set(citance["Reference Offset"])

            texts = soup.findAll()
            satya_input_query = ""
            # Add text together from different citation sentences.
            # Results show that a big single query performs better.
            for el in texts:
                satya_input_query += el.text
            key = satya_input_query.strip(" \t\n")
            if key.startswith("In explormg these quest1ons"):
                key = "In explormg these quest1ons"
            try:
                predicted = result_dict[key]
                tp += len(truth.intersection(predicted))
                print(predicted)
                print(truth)
                print("-------------------------------------")
                overall_predicted += len(predicted)
                overall_truth += len(truth)
            except KeyError:
                count_errors += 1

    print(f"Found {count_errors} KeyErrors, with a total of {len(result_dict)} queries.")
    print(tp)

    prec = tp / overall_predicted
    print(f"Precision: {prec:.4f}")
    rec = tp / overall_truth
    print(f"Recall: {rec:.4f}")
    f1 = 2* (prec * rec) / (prec + rec)
    print(f"F1: {f1:.4f}")