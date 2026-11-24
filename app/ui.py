import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

METHODS = {
    "BM25 (keyword)": "bm25",
    "Semantic (embeddings)": "semantic",
    "Hybrid (BM25 + semantic)": "hybrid",
    "Hybrid + Re-ranker": "hybrid_rerank",
}

st.set_page_config(page_title="Semantic Product Search", page_icon="🔎", layout="wide")


def call_api(query, method, top_k):
    r = requests.get(
        f"{API_URL}/search",
        params={"q": query, "method": method, "top_k": top_k},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def show_results(data, title):
    st.subheader(title)
    st.caption(f"Response time: {data['latency_ms']} ms")
    for item in data["results"]:
        with st.container(border=True):
            title = item["title"][:110].strip().replace("$", "\\$")
            st.markdown(f"**{item['rank']}. {title}**")
            meta = " · ".join(x for x in [item["brand"], item["color"]] if x)
            if meta:
                st.caption(meta)
            st.caption(f"Score: {item['score']}")
            if item["description"]:
                with st.expander("Description"):
                    st.write(item["description"])


# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
mode = st.sidebar.radio("View", ["Single search", "Compare two methods"])
top_k = st.sidebar.slider("Number of results", 3, 20, 5)

if mode == "Single search":
    method_label = st.sidebar.selectbox("Method", list(METHODS), index=3)
else:
    left_label = st.sidebar.selectbox("Left method", list(METHODS), index=0)
    right_label = st.sidebar.selectbox("Right method", list(METHODS), index=3)

st.sidebar.markdown("---")
st.sidebar.caption("Re-ranking is the most accurate method but also the slowest.")

# ---------------- Main page ----------------
st.title("🔎 E-commerce Semantic Product Search")
st.write("Compare keyword search with semantic search on 40,000 Amazon products.")

with st.form("search_form"):
    query = st.text_input("Search for a product", placeholder="e.g. something to keep my coffee hot")
    submitted = st.form_submit_button("Search")

st.caption("Try: something to keep my coffee hot · gift for a 5 year old boy · laptop bag for 15 inch")

if submitted:
    if not query.strip():
        st.warning("Please type a search query.")
    else:
        try:
            with st.spinner("Searching..."):
                if mode == "Single search":
                    data = call_api(query, METHODS[method_label], top_k)
                    show_results(data, method_label)
                else:
                    col1, col2 = st.columns(2)
                    left = call_api(query, METHODS[left_label], top_k)
                    right = call_api(query, METHODS[right_label], top_k)
                    with col1:
                        show_results(left, left_label)
                    with col2:
                        show_results(right, right_label)
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the API. Start it first with: uvicorn app.api:app")
        except requests.exceptions.RequestException as e:
            st.error(f"API error: {e}")