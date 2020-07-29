from lxml import etree
from tqdm import tqdm
import pandas as pd
import linecache
import os


if __name__ == "__main__":
    run_name = "intersection_2_field"
    result_data = "./results_intersection_2_fields.tsv"

    # run_name = "negative_only_2_field"
    # result_data = "./doc_ids_search_only.csv"

    test_path = "./data/Test-Set-2018"

    base_path = os.path.join ("./runs/", run_name, "Task1")
    os.makedirs(base_path, exist_ok=True)

    with open(result_data) as f:
        lines = f.readlines()

    result_dict = {}
    for line in lines:
        curr_split = line.split("\t")
        result_dict[curr_split[0]] = curr_split[1]

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
            all_sids = eval(result_dict[row["Citation Text Clean"]])
            # Map format to expected output
            temp_result_data.at[i, "Reference Offset"] = ",".join(["'" + el + "'" for el in all_sids])
            ref_text = ""  # Multiple references are just appended one after another.
            for sid in all_sids:
                sourceline = tree.xpath(".//S[@sid='" + sid + "']")[0].sourceline
                line = linecache.getline(xml_file, sourceline)
                ref_text += line.strip(" \t\n")

            temp_result_data.at[i, "Reference Text"] = ref_text

        temp_result_data.to_csv(os.path.join(base_path, filename + ".csv"), index=False)


