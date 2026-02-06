
import streamlit as st
from dotenv import load_dotenv

from text_preprocessing import file_type_language as ftl
from text_preprocessing import apicall as gem
import nlp as nlp_module

load_dotenv()



st.title("Legal Contract Analysis Bot")

uploaded_file = st.file_uploader(
    "Upload your legal document (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"]
)

if st.button("Analyze Document"):

    if uploaded_file is not None:
        try:
            st.info(" Extracting and normalizing text ")

            raw_text = ftl.extract_text_from_file(uploaded_file)

            st.success(" Text extracted and normalized successfully!")

            contract_type = nlp_module.classify_contract_type(raw_text)
            st.success(f" Detected Contract Type: {contract_type}")

            st.info("rocessing clauses ")

            results = gem.process_text(raw_text)

            st.success("Document processed successfully!")

            for idx, item in enumerate(results, start=1):

                clause_text = item["clause"]
                nlp_data = nlp_module.analyze_clause(clause_text)

                st.markdown(f"### Clause {idx}")

                st.write("**Clause Text:**")
                st.write(clause_text)

                # ---- NLP OUTPUT ----
                entities = nlp_data["entities"]

                st.write("**Extracted Entities:**")
                st.write(f"• Parties: {', '.join(entities['PARTIES']) if entities['PARTIES'] else 'None'}")
                st.write(f"• Dates: {', '.join(entities['DATES']) if entities['DATES'] else 'None'}")
                st.write(f"• Money: {', '.join(entities['MONEY']) if entities['MONEY'] else 'None'}")
                st.write(f"• Jurisdiction: {', '.join(entities['JURISDICTION']) if entities['JURISDICTION'] else 'None'}")

                st.write(f"**Clause Type:** {nlp_data['clause_type']}")
                st.write(f"**Ambiguity Detected:** {'Yes' if nlp_data['ambiguity'] else 'No'}")
                st.write(f"**Risk Level (NLP):** {nlp_data['risk_level']}")

            
                st.write("**Gemini Explanation:**")
                st.write(item.get("explanation", "No explanation generated"))

                st.markdown("---")

        except Exception as e:
            st.error(f" Error processing file: {e}")

    else:
        st.warning(" Please upload a document first.")
