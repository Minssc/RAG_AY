#!/usr/bin/env python3
"""
Streamlit RAG 앱 — Drone Info Assistant
- FAISS 인덱스에서 문단을 검색하고 LLM으로 근거 포함 답변 생성
- Ollama (로컬) 또는 OpenAI 사용 가능
"""

import os
import json
import streamlit as st
from typing import List, Dict, Optional

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# -------- Config --------
INDEX_DIR = "vector_index"        # FAISS 저장 위치 (build_vectorstore.py 에서 만든 폴더)
JSON_PATH = "drone_docs.json"     # 원본 JSON (선택적)
USE_OLLAMA_DEFAULT = True         # 기본 LLM (변경 가능)

# -------- Helpers --------
def load_vectorstore(index_dir: str) -> Optional[FAISS]:
    if not os.path.exists(index_dir):
        return None
    try:
        embeddings = OllamaEmbeddings(model="llama3.1:8b")
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.error(f"Vectorstore 로드 실패: {e}")
        return None

def get_llm(use_ollama: bool, temperature: float = 0.2):
    """LLM 인스턴스 반환 (Ollama or OpenAI)"""
    if use_ollama:
        return ChatOllama(model="exaone3.5:7.8b", temperature=temperature)

def get_embeddings(use_ollama: bool):
    if use_ollama:
        return OllamaEmbeddings(model="llama3.1:8b")

def docs_to_display_text(docs: List[Document]) -> str:
    """검색된 문서(들)를 읽기 좋은 문자열로 변환 (출처 포함)"""
    parts = []
    for d in docs:
        meta = d.metadata or {}
        title = meta.get("title", meta.get("source", "unknown"))
        chunk_id = meta.get("chunk_id", None)
        src = f"{title}" + (f" (chunk {chunk_id})" if chunk_id is not None else "")
        text = d.page_content if hasattr(d, "page_content") else str(d)
        parts.append(f"---\n📘 {src}\n{text}")
    return "\n\n".join(parts)

# -------- Streamlit UI --------
st.set_page_config(page_title="🛸 Drone Info Assistant (RAG)", layout="wide", page_icon="🛸")
st.title("🛸 Drone Info Assistant — RAG 기반 문서 검색 & 답변")

# Sidebar: engine options & admin
with st.sidebar:
    st.header("설정")
    use_ollama = st.checkbox("Use Ollama (로컬 LLM & Embeddings)", value=USE_OLLAMA_DEFAULT)
    temperature = st.slider("LLM 온도", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    k_retrieve = st.number_input("검색할 문서 수 (k)", value=4, min_value=1, max_value=10, step=1)
    st.markdown("---")
    st.write("인덱스 관리")
    if st.button("인덱스 로드"):
        st.session_state["vectorstore"] = load_vectorstore(INDEX_DIR)
        if st.session_state["vectorstore"]:
            st.success("✅ 인덱스 로드 완료")
        else:
            st.warning("인덱스를 찾을 수 없습니다. `build_vectorstore.py`로 생성하세요.")
    if st.button("인덱스 상태 확인"):
        vs = load_vectorstore(INDEX_DIR)
        if vs:
            st.write("Vectorstore info:", type(vs), getattr(vs, "index", "no-index"))
            st.success("인덱스가 존재합니다.")
        else:
            st.error("인덱스를 찾을 수 없습니다.")

# Load (or lazy load) vectorstore
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = load_vectorstore(INDEX_DIR)

if st.session_state["vectorstore"] is None:
    st.warning("FAISS 인덱스가 로드되지 않았습니다. 사이드바에서 '인덱스 로드' 를 클릭하거나 먼저 build_vectorstore.py 로 인덱스를 만드세요.")
    st.stop()

vectorstore: FAISS = st.session_state["vectorstore"]

# Category filter (derived from metadata if available)
all_categories = sorted(list({d.metadata.get("category","기타") for d in vectorstore.docstore._dict.values()})) if hasattr(vectorstore, "docstore") else ["법률","매뉴얼","기타"]
selected_categories = st.multiselect("검색할 카테고리 필터 (비워두면 전체 검색)", options=all_categories, default=[])

# Chat history init
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Query input
st.markdown("### 질문 입력")
query = st.text_input("드론 관련 질문을 입력하세요 (예: 비행 허가 절차는 어떻게 되나요?)")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🔍 검색 및 답변", use_container_width=True):
        if not query.strip():
            st.warning("질문을 입력하세요.")
        else:
            # Create retriever with possible filter
            retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieve})
            # If category filters present, construct simple metadata filter if vectorstore supports it
            if selected_categories:
                def filtered_retriever(q, k=k_retrieve):
                    docs = retriever.vectorstore.similarity_search(query, k=5)
                    # client-side filter fallback: filter by metadata category and slice k
                    filtered = [d for d in docs if d.metadata.get("category") in selected_categories]
                    return filtered[:k]
                # use filtered_retriever for retrieval
                retrieved_docs = filtered_retriever(query)
            else:
                retrieved_docs = retriever.vectorstore.similarity_search(query, k=5)
                if not retrieved_docs:
                    st.warning("⚠️ 드론 관련 문서를 찾을 수 없습니다. 질문을 다시 입력해주세요.")
                    st.stop()

            # Display retrieved docs (collapsed)
            with st.expander(f"🔎 검색된 {len(retrieved_docs)}개 문서 보기"):
                for d in retrieved_docs:
                    title = d.metadata.get("title", d.metadata.get("source", "unknown"))
                    chunk = d.metadata.get("chunk_id", "")
                    st.markdown(f"**{title}** (chunk: {chunk})")
                    st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))

            # Prepare LLM
            llm = get_llm(use_ollama=use_ollama, temperature=temperature)
            # Build prompt: include retrieved docs as context + explicit instruction to cite sources
            context_text = docs_to_display_text(retrieved_docs)
            system_template = (
                "당신은 드론 관련 법규와 기술 문서에 대한 전문 분석가입니다. "
                "다음에 제시된 문서들을 **유일한 정보 출처**로 사용해야 합니다. "
                "문서 외부의 정보나 URL, 출처, 규정, 수치를 생성하거나 추측하지 마세요.\n\n"
                "=== 참고 문서 시작 ===\n"
                "{context}\n"
                "=== 참고 문서 끝 ===\n\n"
                "엄격한 지침:\n"
                "1. 위 문서의 내용만을 바탕으로 답변하세요.\n"
                "2. 문서에 포함되지 않은 URL, 기관명, 수치, 규정을 절대로 생성하지 마세요.\n"
                "3. 문서에 'url' 또는 'source' 메타데이터가 없는 경우, 그 항목은 생략하세요.\n"
                "4. '참고문헌' 섹션에는 실제 문서 메타데이터에 존재하는 title, url만 표시하세요.\n"
                "5. 문서 내용이 불충분할 경우, '해당 문서에서는 관련 정보를 찾을 수 없습니다.' 라고 명시하세요.\n"
                "질문이 드론과 무관할 경우, 반드시 다음처럼 답하세요: 이 시스템은 드론 관련 정보만 제공합니다.\n"
                "6. 사용자의 질문 언어를 감지하고, 반드시 동일한 언어로 답변하세요."
            )
            user_template = "사용자 질문: {query}\n\n요약하고 단계별로 설명하세요."
            system_prompt = system_template.format(context=context_text)
            user_prompt = user_template.format(query=query)

            # Compose messages according to model interface
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                resp = llm.invoke(messages) if hasattr(llm, "invoke") else llm(messages)
                answer_text = getattr(resp, "content", str(resp))
            except Exception as e:
                st.error(f"LLM 호출 실패: {e}")
                answer_text = f"오류로 인해 답변을 생성할 수 없습니다: {e}"

            # Save history
            st.session_state["chat_history"].append({"q": query, "a": answer_text})
            st.rerun()

with col2:
    if st.button("🧾 대화 기록 초기화", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# Show chat history
st.markdown("### 대화 기록")
for i, turn in enumerate(reversed(st.session_state["chat_history"]), 1):
    st.markdown(f"**Q{i}.** {turn['q']}")
    st.markdown(f"**A{i}.**")
    st.write(turn["a"])
    st.markdown("---")

# Optional: allow uploading new doc and indexing in-memory
st.markdown("### 문서 업로드 및 인덱스에 추가 (선택)")
uploaded_files = st.file_uploader("PDF / 텍스트 파일 업로드 (여러개 가능)", accept_multiple_files=True)
if uploaded_files:
    if st.button("➕ 업로드 문서 인덱싱 (메모리)"):
        embeddings = get_embeddings(use_ollama=use_ollama)
        new_docs = []
        import fitz, io
        for f in uploaded_files:
            filename = f.name
            if filename.lower().endswith(".pdf"):
                # extract text quickly via PyMuPDF
                pdf_bytes = f.read()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                txt = ""
                for p in pdf_doc:
                    txt += p.get_text("text") + "\n"
            else:
                txt = f.getvalue().decode("utf-8")
            # split naive (could use RecursiveCharacterTextSplitter if available)
            # We'll create a single doc per file for simplicity
            new_docs.append(Document(page_content=txt, metadata={"title": filename, "category": "업로드"}))

        try:
            # add_documents available in many FAISS wrappers
            vectorstore.add_documents(new_docs, embeddings=embeddings)
            st.success("업로드 문서를 인메모리로 인덱싱했습니다. (앱 재시작 전 유지)")
        except Exception as e:
            st.error(f"인덱스 추가 실패: {e}")

st.markdown("### 사용 팁")
st.write(
    "- 검색 시 원하는 카테고리를 선택하면 관련 문서만 우선 검색합니다.\n"
    "- 답변에 출처가 명시되지 않으면 해당 질문에 대한 근거 문서가 부족하다는 뜻입니다.\n"
)

