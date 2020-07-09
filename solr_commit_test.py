from lxml import etree
import pysolr
import os
import re


if __name__ == "__main__":
    solr = pysolr.Solr('http://localhost:8983/solr/scisumm/', always_commit=True)

    solr.ping()

    # solr.add([
    #     {
    #         "id": "1",
    #         "ssid": "1",
    #         "reference_section": "Abstract",
    #         "text": "This is a first test."
    #     },
    # ])

    base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + "C00-2123"
    ref_xml = os.path.join(base_path, "Reference_XML", "C00-2123.xml")
    tree = etree.parse(ref_xml, parser=etree.XMLParser(encoding='ISO-8859-1', recover=True))
    root = tree.getroot()

    sentences = tree.xpath(".//S")

    for sentence in sentences:
        try:
            ssid = sentence.attrib["ssid"]
        except KeyError:
            ssid = "0"  # Only happens for title.

        # Reference sections for PAPER (title) and ABSTRACT are different
        try:
            reference_section = sentence.getparent().attrib["title"]
        except KeyError:
            reference_section = sentence.getparent().tag

        # clean_text = re.sub(r"\+-(&&)\|\|!\(\)\{\}\[\]\^\"~\*\?:\\/", "", sentence.text)
        clean_text = sentence.text
        solr.add([
            {
                "id": sentence.attrib["sid"],
                "ssid": ssid,
                "reference_section": reference_section,
                "text": clean_text
            }])
