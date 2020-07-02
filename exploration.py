"""
First analysis
"""

from lxml import etree
import os

if __name__ == "__main__":
    base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/C00-2123/"
    ref_xml = os.path.join(base_path, "Reference_XML", "C00-2123.xml")
    tree = etree.parse(ref_xml)
    root = tree.getroot()
    print(etree.tostring(root))

    annotations_file = os.path.join(base_path, "annotation", "C00-2123.ann.txt")
    with open(annotations_file) as f:
        annotations = f.readlines()

    
    for line in annotations:



