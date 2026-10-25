import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from main import retrieve_policy, generate_answer



st.title("政策问答助手")

query = st.text_input("请输入您的问题：")
if query:
    with st.spinner("正在检索相关政策..."):
        result = retrieve_policy(query)
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        docs = documents[0] if documents else []
        metas = metadatas[0] if metadatas else []
        if not docs:
            st.warning("抱歉，未检索到相关文档。")
        else:
            answer = generate_answer(query, docs, metas)
            st.write(answer)