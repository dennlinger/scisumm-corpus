from lxml import etree
from tqdm import tqdm
import subprocess
import pysolr
import os


if __name__ == "__main__":

    for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/"))):

        subprocess.call(["/home/dennis/solr-8.5.2/bin/solr", "create_core", "-c", filename, "-d",
                         "/home/dennis/solr-8.5.2/server/solr/configsets/scisumm/"])

        solr = pysolr.Solr('http://localhost:8983/solr/' + filename + '/', always_commit=True)

        solr.ping()

        # solr.add([
        #     {
        #         "id": "1",
        #         "ssid": "1",
        #         "reference_section": "Abstract",
        #         "text": "This is a first test."
        #     },
        # ])

        base_path = "./data/Training-Set-2019/Task1/From-Training-Set-2018/" + filename
        ref_xml = os.path.join(base_path, "Reference_XML", filename + ".xml")
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
