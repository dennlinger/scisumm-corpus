from lxml import etree
from tqdm import tqdm
import subprocess
import pysolr
import time
import os
import re

def get_clean_text(text: str) -> str:
    """
    Preprocessing for query parameters.
    :param text:
    :return:
    """
    # Remove Lastname et al. \ Keep group to potentially keep their name only.
    clean_text = re.sub(r"\(?([A-Za-z]+) et al.(, \(?[0-9]{4}\)?)?", "", text)

    # Remove "Lastname and Lastname (<year>)"
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ and [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)

    # TODO: Evaluate if replacing it with "translated" characters would be better?
    # Remove HTML special characters
    clean_text = re.sub(r"\&[a-z]{4};", "", clean_text)

    # Clean up any left over duplicate spaces
    clean_text = re.sub(r"\s+", " ", clean_text)

    # Remove any non-ascii character from the query, according to
    # https://stackoverflow.com/a/18430817/3607203
    clean_text = clean_text.encode("ascii", errors="ignore").decode()

    # Don't mask the special characters here, since solr will take care of it.

    return clean_text


if __name__ == "__main__":

    # Clean indexes
    for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/"))):
        subprocess.call(["/home/dennis/solr-8.5.2/bin/solr", "delete", "-c", filename])
    subprocess.call(["/home/dennis/solr-8.5.2/bin/solr", "restart"])
    time.sleep(5)  # Give time to restart, although above runs already until restart time.

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

        values = []

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
            clean_text = get_clean_text(sentence.text)

            values.append({
                    "id": sentence.attrib["sid"],
                    "ssid": ssid,
                    "reference_section": reference_section,
                    "text": clean_text
                })

        solr.add(values)
