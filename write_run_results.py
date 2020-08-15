from bs4 import BeautifulSoup
from lxml import etree
from tqdm import tqdm
import pandas as pd
import linecache
import os


if __name__ == "__main__":
    # run_name = "intersection_3_field"
    # result_data = "./results_intersection_3_fields.tsv"

    # run_name = "intersection_2_field"
    # result_data = "./results_intersection_2_fields.tsv"

    # run_name = "negative_only_2_field"
    # result_data = "./doc_ids_test_search_only_2_new.csv"

    run_name = "negative_only_3_field"
    result_data = "./doc_ids_test_search_only_3_new.csv"

    # run_name = "with_truth_2_field"
    # result_data = "./doc_ids_test_ground_truth_2_new.csv"

    # run_name = "with_truth_3_field"
    # result_data = "./doc_ids_test_ground_truth_3_new.csv"

    test_path = "./data/Test-Set-2018"

    base_path = os.path.join("./runs/", run_name, "Task1")
    os.makedirs(base_path, exist_ok=True)

    # with open(result_data) as f:
    #     lines = f.readlines()
    #
    # result_dict = {}
    # for line in lines:
    #     curr_split = line.split("\t")
    #     result_dict[curr_split[0]] = curr_split[1]

    result_dict = {}
    data = pd.read_csv(result_data, delimiter="\t")

    for i, row in data.iterrows():
        # unescape html characters
        clean_text = row["query"]

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
        relevant_doc_ids = sorted_doc_ids[:4]
        result_dict[clean_text] = relevant_doc_ids

    exc = 0
    for filename in tqdm(sorted(os.listdir(test_path))):
        annotations = os.path.join(test_path, filename, "annotation", filename + ".csv")
        xml_file = os.path.join(test_path, filename, "Reference_XML", filename + ".xml")
        # Get tree to export sentences
        tree = etree.parse(xml_file, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        # load incomplete data file
        temp_result_data = pd.read_csv(annotations, quotechar='"', sep=",")
        temp_result_data["Reference Offset"] = temp_result_data["Reference Offset"].astype(object)
        temp_result_data["Reference Text"] = temp_result_data["Reference Offset"].astype(object)
        for i, row in temp_result_data.iterrows():
            # all_sids = eval(result_dict[row["Citation Text Clean"]])
            all_sids = result_dict[row["Citation Text Clean"]]
            # Map format to expected output
            temp_result_data.at[i, "Reference Offset"] = ",".join(["'" + str(el) + "'" for el in all_sids])
            ref_text = ""  # Multiple references are just appended one after another.
            for sid in all_sids:
                sourceline = tree.xpath(".//S[@sid='" + str(sid) + "']")[0].sourceline
                line = linecache.getline(xml_file, sourceline)
                ref_text += line.strip(" \t\n")

            temp_result_data.at[i, "Reference Text"] = ref_text

        temp_result_data.to_csv(os.path.join(base_path, filename + ".csv"), index=False)


