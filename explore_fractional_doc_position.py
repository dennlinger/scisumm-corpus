import matplotlib.pyplot as plt
from lxml import etree
from tqdm import tqdm
import numpy as np
import os

from exploration import get_citances_for_file

if __name__ == "__main__":
    positions = []

    for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/"))):
        citances = get_citances_for_file(filename, list())

        for citance in citances:
            base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + \
                        citance["Reference Article"].split(".")[0]
            # replace potential wrong file extension
            xml_filename = citance["Reference Article"].split(".")[0] + ".xml"
            ref_xml = os.path.join(base_path, "Reference_XML", xml_filename)
            tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))

            sentences = tree.xpath(".//S")

            truth = citance["Reference Offset"]
            for pos in truth:
                positions.append(int(pos) / len(sentences))

    y, x = np.histogram(positions, bins=4, range=(0, 1))
    x = x[:-1] + 0.05
    plt.bar(x, y, width=0.1)
    plt.show()
