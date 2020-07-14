from nltk.corpus import stopwords
import requests
import json


if __name__ == "__main__":
    print(len(stopwords.words("english")))
    res = requests.get("http://localhost:8983/solr/scisumm/schema/analysis/stopwords/english")
    print(len(res.json()["wordSet"]["managedList"]))
    words = json.dumps(stopwords.words("english"))
    requests.put("http://localhost:8983/solr/scisumm/schema/analysis/stopwords/english", data=words)

    res = requests.get("http://localhost:8983/solr/scisumm/schema/analysis/stopwords/english")
    print(len(res.json()["wordSet"]["managedList"]))
