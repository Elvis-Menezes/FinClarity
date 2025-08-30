from transformers import pipeline
import re

# load once (faster than loading inside the function every call)
ner = pipeline("token-classification",
               model="dslim/bert-base-NER",
               aggregation_strategy="average")

def ner_tag(query: str):
    ents = ner(query)

    # clean helper (pipeline may return extra spaces sometimes)
    def clean(s): 
        return re.sub(r"\s+", " ", s).strip()

    orgs = [clean(e["word"]) for e in ents if e.get("entity_group") == "ORG"]

    if not orgs:
        print("No ORG entity found.")
        return None

    # print all ORG entities found
    for org in dict.fromkeys(orgs):  # keeps order, removes dups
        print(org)
    return orgs

# Example:
ner_tag("Give me a story about Reliance Industeries")