import os
import re
import time
import spacy
from google import genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_NAME = "xx_ent_wiki_sm"
nlp = spacy.load(MODEL_NAME)
nlp.add_pipe("sentencizer")


def split_into_clauses(text: str):
    text = text.replace("\r", "\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    clauses = []
    current_clause = ""

    for sent in sentences:
        if re.match(r'^(\d+(\.\d+)*|\([a-zA-Z]\)|[IVX]+\.)', sent):
            if current_clause:
                clauses.append(current_clause.strip())
            current_clause = sent
        else:
            current_clause += " " + sent if current_clause else sent

    if current_clause:
        clauses.append(current_clause.strip())

    cleaned = []
    for c in clauses:
        if len(c) >= 25:
            cleaned.append(c.strip())

    return cleaned


def parse_model_output(text: str, expected_count: int):
    parts = text.split("===CLAUSE_START===")
    results = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        results.append(part)

    if len(results) != expected_count:
        return ["[Parsing error: model returned unexpected format]"] * expected_count

    return results


def call_gemini_batch(clauses_batch: list):

    joined_clauses = ""
    for i, clause in enumerate(clauses_batch, start=1):
        joined_clauses += f"Clause {i}:\n{clause}\n\n"

    prompt = f"""
You are a GenAI Legal Assistant for Indian SME contract analysis.

CRITICAL RULES (must follow):
- Analyze ONLY the clauses provided in INPUT CLAUSES
- Do NOT analyze or mention any clause not present
- Do NOT summarize the whole agreement
- Do NOT repeat previous clauses
- Produce output for EXACTLY {len(clauses_batch)} clauses
- If you output more or fewer clauses, the response is invalid

For EACH clause provide:
1. Explanation (2–3 bullet points, max 40 words)
2. Risk Level (Low/Medium/High)
3. Unfavorable Terms (or "None identified")
4. Suggested Alternative (only if risk is Medium or High, else "Not required")
5. Compliance (Indian Law) or "No specific compliance issue found"

OUTPUT FORMAT (STRICT):

===CLAUSE_START===
ClauseNumber: 1
Explanation:
- point
- point
Risk Level: <Low/Medium/High>
Unfavorable Terms:
- point
Suggested Alternative:
- text
Compliance (Indian Law):
- text


(Repeat exactly this format for each clause)

INPUT CLAUSES:
{joined_clauses}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    if hasattr(response, "candidates") and len(response.candidates) > 0:
        text = response.candidates[0].content.parts[0].text
        return parse_model_output(text, len(clauses_batch))
    else:
        return ["[No response from Gemini]"] * len(clauses_batch)


def process_text(text: str):

    clauses = split_into_clauses(text)
    results = []

    batch_size = 5 

    for i in range(0, len(clauses), batch_size):
        batch = clauses[i:i + batch_size]

        try:
            explanations = call_gemini_batch(batch)

            for clause, explanation in zip(batch, explanations):
                results.append({
                    "clause": clause,
                    "explanation": explanation
                })

            time.sleep(2)

        except Exception as e:
            for clause in batch:
                results.append({
                    "clause": clause,
                    "explanation": f"[Error: {e}]"
                })

    return results
