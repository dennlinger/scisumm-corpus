from html import unescape
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

    ####### Citations #########
    # Remove Lastname et al. \ Keep group to potentially keep their name only.
    clean_text = re.sub(r"\(?([A-Za-z]+) et al.(, \(?[0-9]{4}\)?)?", "", text)

    # Remove "Lastname and Lastname (<year>)","Lastname & Lastname (<year>), "Lastname, Lastname and Lastname (<year>),
    # (Lastname and others <year>)"
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+,? [A-Z][A-Za-z\-]+,? and [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ and [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ &amp[;]* [A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+ and others, \(?[0-9]{4}\)?", "", clean_text)

    # Remove " (Lastname, <year>)"
    clean_text = re.sub(r"\(?[A-Z][A-Za-z\-]+,? \(?[0-9]{4}\)?", "", clean_text)

    # Remove "Lastname and Lastname(year)"
    clean_text = re.sub(r"[A-Z][A-Za-z\-]+,? and [A-Z][A-Za-z\-]+,?\(?[0-9]{4}\)?", "", clean_text)

    # Remove " [number]" for citations
    clean_text = re.sub(r"\[[0-9]+\]?", "", clean_text)

    ####### Math #########
    clean_text=clean_text.replace("(e.g.", "example")  # otherwise they will identified as functions
    # Starting with O() is a complexity
    clean_text = re.sub(r"O[\s]*\([^\)]+\)", "<COMPLEXITY>", clean_text)
    # Starting with P or Pr () its a probablity
    clean_text = re.sub(r"[\s=][Pp][\s]*[r]*[\s]*\([^\)]+\)", " <PROBABILITY>", clean_text)
    # [numbers] vector
    clean_text = re.sub(r"\[[^\]]+[,][^\]]+\]", "<VECTOR>", clean_text)
    # character[=/]() a function
    clean_text = re.sub(r"[A-Za-z]*[\s]*[=/\d]*[\s]*\([^\)]+[=+][^\)]+\)", "<FUNCTION>", clean_text)

    # Remove any non-ascii character from the query, according to
    # https://stackoverflow.com/a/18430817/3607203
    clean_text = clean_text.encode("ascii", errors="ignore").decode()

    # unescape html characters
    clean_text = unescape(clean_text)

    # remove punctuation
    clean_text = re.sub(r"[;,#]+", "", clean_text)
    clean_text = re.sub(r"[\s]\W[\s]", " ", clean_text)

    # Clean up any left over duplicate spaces
    clean_text = re.sub(r"\s+", " ", clean_text)

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
        committed = False
        auxil_offset = -1
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
            if sentence.attrib["sid"] == "":
                sid = str(auxil_offset)
                auxil_offset -= 1
            else:
                sid = sentence.attrib["sid"]
            fraction_sid = max(0, int(sid))
            fraction = fraction_sid / len(sentences)
            values.append({
                    "id": sid,
                    "ssid": ssid,
                    "reference_section": reference_section,
                    "text": clean_text,
                    # "text2": clean_text,
                    # "text3": clean_text,
                    # "text4": clean_text,
                    # "text5": clean_text,
                })

            # Give the first 30% of the document a significant boost
            if fraction >= 0.3 and not committed:
                solr.add(values, boost={"text": 1.25})
                values = []
                committed = True  # So it doesn't happen every round from now on.
        # Remaining values are not boosted
        solr.add(values)
