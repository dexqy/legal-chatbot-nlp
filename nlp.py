
import spacy
import re

nlp = spacy.load("xx_ent_wiki_sm")

def classify_contract_type(text: str) -> str:
    text = text.lower()
    if "employment" in text or "employee" in text:
        return "Employment Contract"
    elif "lease" in text or "rent" in text:
        return "Lease Agreement"
    elif "vendor" in text or "supplier" in text:
        return "Vendor Contract"
    elif "partnership" in text:
        return "Partnership Deed"
    elif "service" in text:
        return "Service Agreement"
    else:
        return "General Contract"


def extract_entities(clause: str) -> dict:
    doc = nlp(clause)
    entities = {
        "PARTIES": [],
        "DATES": [],
        "MONEY": [],
        "JURISDICTION": [],
    }

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            entities["PARTIES"].append(ent.text)
        elif ent.label_ == "DATE":
            entities["DATES"].append(ent.text)
        elif ent.label_ == "MONEY":
            entities["MONEY"].append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            entities["JURISDICTION"].append(ent.text)

    return entities


def classify_clause_type(clause: str) -> str:
    clause_lower = clause.lower()

    if any(word in clause_lower for word in ["shall", "must", "required to"]):
        return "Obligation"
    elif any(word in clause_lower for word in ["may", "entitled to"]):
        return "Right"
    elif any(word in clause_lower for word in ["shall not", "must not", "prohibited"]):
        return "Prohibition"
    else:
        return "Neutral"

def detect_ambiguity(clause: str) -> bool:
    ambiguous_words = ["reasonable", "as necessary", "as soon as possible", "may", "appropriate"]
    return any(word in clause.lower() for word in ambiguous_words)


def assess_risk(clause: str) -> str:
    clause_lower = clause.lower()

    high_risk_terms = ["penalty", "terminate immediately", "indemnify", "liability", "non-compete"]
    medium_risk_terms = ["notice", "renewal", "confidential", "jurisdiction"]

    if any(word in clause_lower for word in high_risk_terms):
        return "High"
    elif any(word in clause_lower for word in medium_risk_terms):
        return "Medium"
    else:
        return "Low"


def analyze_clause(clause: str) -> dict:
    return {
        "entities": extract_entities(clause),
        "clause_type": classify_clause_type(clause),
        "ambiguity": detect_ambiguity(clause),
        "risk_level": assess_risk(clause)
    }
