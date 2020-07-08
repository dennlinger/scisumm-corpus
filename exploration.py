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

            # Merge Discourse Facets to consistent naming
            elif k == "Discourse Facet":
                if v.strip(" ")[0] == "[":
                     temp_facets = eval(v)
                else:
                    temp_facets = [v]

                temp_facets = [facet.lower().replace(" ", "_").replace("result_", "results_") for facet in temp_facets]
                citance_dict[k] = temp_facets

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

    parent_name = []
    for citance in citances:
        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Citing Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Citance_XML", xml_filename)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()
        if len(citance["Citation Offset"]) > 1:
            print(ref_xml)
            print(len(citance["Citation Offset"]), int(citance["Citation Offset"][-1]) - int(citance["Citation Offset"][0]))

        el = root.xpath(".//S[@sid='" + citance["Citation Offset"][0] + "']")

        parent = el[0].getparent()
        try:
            # parent_name.append(parent.attrib["title"].strip(". ").lower())
            for facet in citance["Discourse Facet"]:
                parent_name.append((parent.attrib["number"], facet))
        except KeyError:
            for facet in citance["Discourse Facet"]:
                parent_name.append((parent.tag, facet))

    print(Counter(parent_name).most_common())

    # Do the same but for the reference texts.
    parent_name = []
    for citance in citances:
        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Reference Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
        print(ref_xml)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()

        prev_offset = -2
        for offset in citance["Reference Offset"]:
            # Don't count subsequent offsets, instead go to the next one.
            if int(offset) - prev_offset == 1:
                prev_offset = int(offset)
                continue
            el = root.xpath(".//S[@sid='" + offset + "']")

            parent = el[0].getparent()
            try:
                parent_name.append(parent.attrib["title"].strip(". ").lower())
            except KeyError:
                parent_name.append(parent.tag)

            prev_offset = int(offset)
            # try:
            #     # parent_name.append(parent.attrib["title"].strip(". ").lower())
            #     for facet in citance["Discourse Facet"]:
            #         parent_name.append((parent.attrib["number"], facet))
            # except KeyError:
            #     for facet in citance["Discourse Facet"]:
            #         parent_name.append((parent.tag, facet))

    print(Counter(parent_name).most_common())


