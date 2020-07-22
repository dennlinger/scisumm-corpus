"""
First analysis
"""

from collections import Counter
from typing import List
from lxml import etree
import os


def get_citances_for_file(file_id: str, citances_json: List,
                          folder: str="./data/Training-Set-2019/Task1/From-Training-Set-2018/") -> List:
    base_path = folder + file_id

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

    for i, citance in enumerate(citances):
        citance_dict = {}

        # print(i+1)

        for el in citance:
            # print(el)
            # Only split at first colon, since text may contain more.
            try:
                k, v = el.split(":", maxsplit=1)
            except ValueError:
                continue
            k = k.strip(" ")
            v = v.strip(" ")
            if k in ("Citation Marker Offset", "Citation Offset", "Reference Offset"):
                try:
                    citance_dict[k] = eval(v)
                except NameError:
                    citance_dict[k] = "undefined"

            # Merge Discourse Facets to consistent naming
            elif k == "Discourse Facet":
                if v.strip(" ")[0] == "[":
                    try:
                        temp_facets = eval(v)
                    except NameError:
                        temp_facets = "undefined"
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
        # if len(citance["Discourse Facet"]) >= 2:
        #     print(citance)
        for el in citance["Discourse Facet"]:
            facets.append(el)

    # print(Counter(facets).most_common())

    parent_name = []
    for citance in citances:
        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Citing Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Citance_XML", xml_filename)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()
        # if len(citance["Citation Offset"]) > 1:
        #     print(ref_xml)
        #     print(len(citance["Citation Offset"]), int(citance["Citation Offset"][-1]) - int(citance["Citation Offset"][0]))

        el = root.xpath(".//S[@sid='" + citance["Citation Offset"][0] + "']")

        parent = el[0].getparent()
        try:
            # parent_name.append(parent.attrib["title"].strip(". ").lower())
            for facet in citance["Discourse Facet"]:
                parent_name.append((parent.attrib["number"], facet))
        except KeyError:
            for facet in citance["Discourse Facet"]:
                parent_name.append((parent.tag, facet))

    # print(Counter(parent_name).most_common())

    # Do the same but for the reference texts.
    parent_name = []
    for citance in citances:
        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Reference Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
        # print(ref_xml)
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
                # parent_name.append(parent.attrib["title"].strip(". ").lower())
                parent_name.append(parent.attrib["number"])
            except KeyError:
                parent_name.append(parent.tag)

            prev_offset = int(offset)
            try:
                # parent_name.append(parent.attrib["title"].strip(". ").lower())
                for facet in citance["Discourse Facet"]:
                    parent_name.append((parent.attrib["number"], facet))
            except KeyError:
                for facet in citance["Discourse Facet"]:
                    parent_name.append((parent.tag, facet))

    # print(Counter(parent_name).most_common())


    section_names = []
    for citance in citances:
        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + citance["Reference Article"].split(".")[0]
        # replace potential wrong file extension
        xml_filename = citance["Citing Article"].split(".")[0] + ".xml"
        ref_xml = os.path.join(base_path, "Citance_XML", xml_filename)
        tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
        root = tree.getroot()

        children = root.xpath("/PAPER/*")

        for child in children:
            try:
                section_names.append(child.attrib["title"].strip(". ").lower())
                if child.attrib["title"].strip(". ").lower() == "overview":
                    print(citance["Reference Article"].split(".")[0], xml_filename)
            except KeyError:
                section_names.append(child.tag.lower())

    # print(Counter(section_names).most_common())

    # Classes are: abstract, introduction, acknowledgements, related work, methods, conclusion, results, unknown
    section_mapping = dict()
    section_mapping["abstract"] = "abstract"
    section_mapping["introduction"] = "introduction"
    section_mapping["acknowledgments"] = "acknowledgements"
    section_mapping["related work"] = "related work"
    section_mapping["conclusion"] = "conclusion"
    section_mapping["experiments"] = "results"
    section_mapping["conclusions"] = "conclusion"
    section_mapping["acknowledgements"] = "acknowledgements"
    section_mapping["evaluation"] = "results"
    section_mapping["discussion"] = "conclusion"
    section_mapping["results"] = "results"
    section_mapping["acknowledgement"] = "acknowledgements"
    section_mapping["conclusions and future work"] = "conclusion"
    section_mapping["conclusion and future work"] = "conclusion"
    section_mapping["background"] = "related work"
    section_mapping["experimental results"] = "results"
    section_mapping["S"] = "unknown"
    section_mapping["previous work"] = "related work"
    section_mapping["experimental evaluation"] = "results"
    section_mapping["experimental setup"] = "results"
    section_mapping["method"] = "methods"
    section_mapping["experiments and results"] = "results"
    section_mapping["motivation"] = "introduction"
    section_mapping["experiment"] = "results"
    section_mapping["acknowledgment"] = "acknowledgements"
    section_mapping["model"] = "methods"
    section_mapping["summary"] = "conclusion"
    section_mapping["system description"] = "methods"
    section_mapping["analysis"] = "results"
    section_mapping["experimentation"] = "results"
    section_mapping["decoding"] = "methods"
    section_mapping["features"] = "methods"
    section_mapping["conclusions and future  work"] = "conclusion"
    section_mapping["methodology"] = "methods"
    section_mapping["discussion and conclusion"] = "conclusion"
    section_mapping["future work"] = "conclusion"
    section_mapping["training"] = "methods"
    section_mapping["related research"] = "related work"
    section_mapping["fertility distribution parameters"] = "methods"
    # overview refers in documents either to introduction or related work..
    section_mapping["overview"] = "unknown"

    
