"""
First analysis
"""

from collections import Counter
from typing import List
from lxml import etree
import os


def get_citances_for_file(file_id: str, citances_json: List) -> List:
    base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + file_id
    # ref_xml = os.path.join(base_path, "Reference_XML", "C00-2123.xml")
    # tree = etree.parse(ref_xml)
    # root = tree.getroot()
    # print(etree.tostring(root))

    try:
        annotations_file = os.path.join(base_path, "annotation", file_id + ".ann.txt")
        with open(annotations_file) as f:
            annotations = f.readlines()
    except FileNotFoundError:
        annotations_file = os.path.join(base_path, "annotation", file_id + ".annv3.txt")
        with open(annotations_file) as f:
            annotations = f.readlines()

    citances = []
    for line in annotations:
        if line.strip():
            citances.append(line.strip("\n |").split(" | "))

    for citance in citances:
        citance_dict = {}

        for el in citance:
            # Only split at first colon, since text may contain more.
            k, v = el.split(":", maxsplit=1)
            k = k.strip(" ")
            v = v.strip(" ")
            if k in ("Citation Marker Offset", "Citation Offset", "Reference Offset"):
                citance_dict[k] = eval(v)
            elif k == "Discourse Facet":
                if v.strip(" ")[0] == "[":
                    citance_dict[k] = eval(v)
                else:
                    citance_dict[k] = [v]
            else:
                citance_dict[k] = v

        citances_json.append(citance_dict)

    return citances_json


if __name__ == "__main__":

    citances = []
    for filename in sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/")):
        citances = get_citances_for_file(filename, citances)

    facets = []
    for citance in citances:
        # print(citance)
        # try:
        #     print(citance["Reference Article"], citance["Citance Number"])
        # except KeyError:
        #     print("ERROR", citance["Reference Article"], citance["Citation Number"])
        if len(citance["Discourse Facet"]) >= 2:
            print(citance)
        for el in citance["Discourse Facet"]:
            facets.append(el)

    print(Counter(facets).most_common())





