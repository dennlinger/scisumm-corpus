from tqdm import tqdm
import pandas as pd
import os


if __name__ == "__main__":
    path_intersection = "runs/intersection_2_field/Task1"
    path_bert = "runs/with_truth_2_field/Task1"

    overlap = 0
    total_i = 0
    total_b = 0
    for filename in tqdm(sorted(os.listdir(path_intersection))):
        int_data = pd.read_csv(os.path.join(path_intersection, filename))
        bert_data = pd.read_csv(os.path.join(path_bert, filename))

        int_offsets = []
        for i, row in int_data.iterrows():
            int_offsets.append(eval("[" + row["Reference Offset"] + "]"))

        bert_offsets = []
        for i, row in bert_data.iterrows():
            preds = eval("[" + row["Reference Offset"] + "]")
            bert_offsets.append(preds)

        for i, b in zip(int_offsets, bert_offsets):
            total_i += len(i)
            total_b += len(b)

            overlap += len(set(i).intersection(set(b)))

    print(f"Overlap: {overlap}, out of {total_b} BERT predictions and {total_i} intersection")
