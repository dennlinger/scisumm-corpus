from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from itertools import chain
from tqdm import tqdm
import os

from exploration import get_citances_for_file
from solr_query_test import get_clean_text
import spacy

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_md")
    token_set = set()
    stopword_set = set(stopwords.words("english"))
    for filename in tqdm(sorted(os.listdir("./data/Training-Set-2019/Task1/From-Training-Set-2018/"))):

        citances = get_citances_for_file(filename, list())

        for citance in citances:
            results = defaultdict(float)
            all_query_sentences = citance["Citation Text"]
            soup = BeautifulSoup(all_query_sentences, "html.parser")
            truth = set(citance["Reference Offset"])
            texts = soup.findAll()
            for el in texts:
                cleaned = get_clean_text(el.text)
                # Make sure to iterate over the tags here
                for token in nlp(cleaned):
                    # Only add the following PoS tags
                    clean_token = token.text.strip("[]\\.()'\"")
                    if token.pos_ in ("NOUN", "VERB", "ADJ") and clean_token not in stopword_set and clean_token:
                        # Some more manual cleaning, here we don't want any special characters
                        token_set.add(clean_token)

    for token in token_set:
        synsets = wn.synsets(token)
        lemmas = set(chain.from_iterable([word.lemma_names() for word in synsets]))
        # Remove _ for multi-word synonyms
        lemmas = [el.replace("_", " ") for el in lemmas]
        if len(lemmas) > 1:
            with open("synonyms.txt", "a") as f:
                f.write(token + " => " + ", ".join(lemmas) + "\n")