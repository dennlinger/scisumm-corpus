from bs4 import BeautifulSoup
from tqdm import tqdm
import os

from exploration import get_citances_for_file

if __name__ == "__main__":
    for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/"))):
    # for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-ScisummNet-2019/"))):
        res = []
        # print(filename)
        citances = get_citances_for_file(filename, res)
        for citance in citances:
            # print(citance['Citance Number'])
            soup = BeautifulSoup(citance["Citation Text"], "html.parser")
            cit_offset = citance["Citation Offset"]
            marker_offset = citance["Citation Marker Offset"]
            texts = soup.findAll()

            # if len(texts) != len(cit_offset):
            #     print(f"{filename}: {citance['Citance Number']}")
            #     print(f"Mismatch of lengths for sentences ({len(texts)}) and citations ({len(cit_offset)}).")
            #
            # if marker_offset[0] not in cit_offset:
            #     print(f"{filename}: {citance['Citance Number']}")
            #     print(f"Mismatch of citations ({cit_offset}) and citation markers ({marker_offset}).")

            ref_offset = citance["Reference Offset"]
            soup = BeautifulSoup(citance["Reference Text"], "html.parser")
            texts = soup.findAll()
            if len(texts) != len(ref_offset):
                print(f"{filename}: {citance['Citance Number']}")
                print(f"Mismatch of reference text ({len(texts)}) and markers ({len(ref_offset)}).")

            for text in texts:
                if text.attrs['sid'] not in ref_offset:
                    print(f"{filename}: {citance['Citance Number']}")
                    print(f"Mismatch of reference text ({texts}) and markers ({ref_offset}).")

