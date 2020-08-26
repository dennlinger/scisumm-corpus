
from solr_query_test import get_citances_for_file
from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm


def write_new_files(lines, queries_to_remove, fn, fn_skipped):
    new_lines = []
    skipped_lines = []
    for line in lines:
        query = line.split("\t")[0]
        if query in queries_to_remove:
            skipped_lines.append(line)
        else:
            new_lines.append(line)

    with open(fn, "w") as f:
        f.write("".join(new_lines))

    with open(fn_skipped, "w") as f:
        f.write("".join(skipped_lines))





if __name__ == "__main__":
    # Relevant files to overwrite
    with open("satya_input_search_only.tsv") as f:
        lines_2_search = f.readlines()
    with open("satya_input_search_only_3.tsv") as f:
        lines_3_search = f.readlines()
    with open("satya_input_with_truth.tsv") as f:
        lines_2_truth = f.readlines()
    with open("satya_input_with_truth_3.tsv") as f:
        lines_3_truth = f.readlines()

    # This is the validation set used by the UoM team (winner 2019)
    valid = sorted(["C00-2123", "C04-1089", "I05-5011", "J96-3004", "N06-2049", "P05-1004", "P05-1053", "P98-1046"])

    folder = "./data/Training-Set-2019/Task1/From-Training-Set-2018/"
    queries_to_remove = []
    for filename in tqdm(valid):  # Used for the validation runs with the UoM data
        citances = get_citances_for_file(filename, list(), folder)

        for i, citance in enumerate(citances, start=1):
            all_query_sentences = citance["Citation Text"]
            soup = BeautifulSoup(all_query_sentences, "html.parser")
            texts = soup.findAll()
            satya_input_query = ""
            # Add text together from different citation sentences.
            # Results show that a big single query performs better.
            for el in texts:
                satya_input_query += el.text
                # print(cleaned+"\n")
            queries_to_remove.append(satya_input_query)

    print(len(queries_to_remove))
    write_new_files(lines_2_search, queries_to_remove, "satya_input_search_only.tsv", "2_search_test.tsv")
    write_new_files(lines_3_search, queries_to_remove, "satya_input_search_only_3.tsv", "3_search_test.tsv")
    write_new_files(lines_2_truth, queries_to_remove, "satya_input_with_truth.tsv", "2_truth_test.tsv")
    write_new_files(lines_3_truth, queries_to_remove, "satya_input_with_truth_3.tsv", "3_truth_test.tsv")

