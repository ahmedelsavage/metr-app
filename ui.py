import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="Metr - Contract Analysis", layout="wide")

st.title("Metr - Contract Analysis")

tab1, tab2 = st.tabs(["Analyze Contract", "Search (RAG)"])

with tab1:
    input_type = st.radio("Input method:", ["Text", "File Upload"])
    
    contract_text = ""
    uploaded_file = None
    
    if input_type == "Text":
        contract_text = st.text_area("Contract text:", height=200)
    else:
        uploaded_file = st.file_uploader("Upload file", type=["pdf", "png", "jpg", "jpeg"])
        
    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            files = None
            data = {"language": "ar", "limit_per_clause": 5}
            
            if input_type == "Text" and contract_text.strip():
                data["text"] = contract_text
            elif input_type == "File Upload" and uploaded_file is not None:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            else:
                st.error("Please provide text or upload a file.")
                st.stop()
                
            try:
                resp = requests.post(f"{API_URL}/analyze", data=data, files=files)
                if resp.status_code == 200:
                    result = resp.json()
                    analysis = result.get("analysis", {})
                    
                    st.success("Analysis complete.")
                    
                    st.subheader("Verdict")
                    verdict = analysis.get("verdict", "")
                    if verdict == "safe":
                        st.info("Safe")
                    else:
                        st.error("Risky")
                    
                    st.write(f"**Advice:** {analysis.get('advice', '')}")
                    
                    st.subheader("Summary")
                    st.write(analysis.get("summary", ""))
                    
                    risks = analysis.get("risks", [])
                    if risks:
                        st.subheader("Risks")
                        for r in risks:
                            with st.container():
                                st.warning(f"Clause: {r.get('clause')}\n\nReason: {r.get('reason')}\n\nReference: {r.get('legal_reference', 'N/A')}")
                    
                    with st.expander("RAG Context"):
                        st.json(result.get("legal_context", []))
                else:
                    st.error(f"Server error: {resp.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab2:
    query = st.text_input("Search query:")
    if st.button("Search"):
        with st.spinner("Searching..."):
            try:
                resp = requests.post(f"{API_URL}/search", data={"query": query, "limit": 5})
                if resp.status_code == 200:
                    chunks = resp.json()
                    for c in chunks:
                        st.info(f"{c.get('law_name')} - Article {c.get('article_number', '')}\n\n{c.get('text')}\n\n(Score: {round(c.get('score', 0), 2)})")
                else:
                    st.error("Search error.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
