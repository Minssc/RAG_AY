#!/usr/bin/env python3
"""
Streamlit RAG 앱 — Drone Info Assistant
- FAISS 인덱스에서 문단을 검색하고 LLM으로 근거 포함 답변 생성
- Ollama (로컬) 또는 OpenAI 사용 가능
"""

import os
from pathlib import Path
import re
import glob
import faiss
import streamlit as st
from typing import List, Dict, Optional
from langdetect import detect, LangDetectException

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.docstore import InMemoryDocstore

# -------- Config --------
PDF_DIR = "drone_pdfs"
INDEX_DIR = "vector_index"
MODEL_NAME = "llama3.1:8b"
USE_OLLAMA_DEFAULT = True 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def classify_category(filename: str) -> str:
    """파일명 기반 간단한 카테고리 분류"""
    name = filename.lower()
    if "law" in name or "법" in name:
        return "법률"
    elif "manual" in name or "guide" in name or "controller" in name:
        return "매뉴얼"
    else:
        return "기타"
    
def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-{2,}', ' ', text)
    text = re.sub(r'Page\s*\d+(/\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[•●◆▶→※■□▣◈◇★☆]', ' ', text)
    return text.strip()

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang.startswith("ko"): return "ko"
        if lang.startswith("en"): return "en"
        return "other"
    except LangDetectException:
        return "unknown"

def load_all_data(pdf_dir: str, txt_dir: str = 'data') -> List[Document]:
    """txt_dir: for MD, RST"""
    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))

    docs: List[Document] = []
    for p in sorted(pdf_paths):
        try:
            loader = PyPDFLoader(p)
            loaded = loader.load()
            category = classify_category(os.path.basename(p))
            for d in loaded:
                d.page_content = clean_text(d.page_content)
                d.metadata["source"] = os.path.basename(p)
                d.metadata["category"] = category 
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"PDF 로드 실패: {os.path.basename(p)} -> {e}")

    for path in Path(txt_dir).rglob('*'):
        if path.suffix.lower() not in ['.md', '.rst']:
            continue
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        docs.append(Document(
            page_content=content,
            metadata = {
                'source': os.path.basename(path),
                'category': '매뉴얼',
            }
        ))
        
    return docs

def build_vectorstore(pdf_dir: str = PDF_DIR, index_dir: str = INDEX_DIR) -> Optional[FAISS]:
    docs = load_all_data(pdf_dir)
    if not docs:
        st.error("데이터가 없습니다. drone_pdfs/data 폴더를 확인하세요.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True)
    splits = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    emb_dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(emb_dim)
    vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    for i in range(0, len(splits), 100):
        vectorstore.add_documents(splits[i:i+100])
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    return vectorstore

@st.cache_resource(show_spinner=False)
def load_or_build_index(index_dir: str = INDEX_DIR):
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
    if os.path.exists(index_dir):
        try:
            return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass
    return build_vectorstore()

def get_llm_stream(stream_mode=True):
    return ChatOllama(model=MODEL_NAME, temperature=0.3, stream=stream_mode)

def get_embeddings(use_ollama: bool):
    return OllamaEmbeddings(model=MODEL_NAME)

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
    stream_enabled = st.sidebar.checkbox("스트리밍 응답", value=True)
    st.markdown("---")
    st.write("인덱스 관리")
    if st.button("인덱스 로드"):
        st.session_state["vectorstore"] = load_or_build_index(INDEX_DIR)
        if st.session_state["vectorstore"]:
            st.success("✅ 인덱스 로드 완료")
        else:
            st.warning("인덱스를 찾을 수 없습니다. `build_vectorstore.py`로 생성하세요.")
    if st.button("인덱스 상태 확인"):
        vs = load_or_build_index(INDEX_DIR)
        if vs:
            st.write("Vectorstore info:", type(vs), getattr(vs, "index", "no-index"))
            st.success("인덱스가 존재합니다.")
        else:
            st.error("인덱스를 찾을 수 없습니다.")
    rebuild = st.sidebar.button("인덱스 재빌드")
    if rebuild:
        import shutil
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
        st.session_state["vectorstore"] = build_vectorstore()
        st.sidebar.success("인덱스 재생성 완료")

# Load (or lazy load) vectorstore
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = load_or_build_index(INDEX_DIR)

if st.session_state["vectorstore"] is None:
    st.warning("FAISS 인덱스가 로드되지 않았습니다. 사이드바에서 '인덱스 로드' 를 클릭하거나 새로 인덱스를 만드세요.")
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
lang = detect_language(query)

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
                    docs = retriever.vectorstore.similarity_search(query, k=k_retrieve)
                    # client-side filter fallback: filter by metadata category and slice k
                    filtered = [d for d in docs if d.metadata.get("category") in selected_categories]
                    return filtered[:k]
                # use filtered_retriever for retrieval
                retrieved_docs = filtered_retriever(query)
            else:
                retrieved_docs = retriever.vectorstore.similarity_search(query, k=k_retrieve)
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
            llm = get_llm_stream(stream_enabled)
            # Build prompt: include retrieved docs as context + explicit instruction to cite sources
            context_text = docs_to_display_text(retrieved_docs)
            system_template = (
                f"**반드시** { '한국어로' if lang=='ko' else '영어로' } 답변하세요.\n\n"
                "문서 외부의 정보나 URL, 출처, 규정, 수치를 생성하거나 추측하지 마세요.\n\n"
                "다음에 제시된 문서들을 **유일한 정보 출처**로 사용해야 합니다. "
                "당신은 드론 관련 법규와 기술 문서에 대한 전문 분석가입니다. "
                "엄격한 지침:\n"
                "1. 아래 참고 문서의 내용만을 바탕으로 답변하세요.\n"
                "2. 문서에 포함되지 않은 URL, 기관명, 수치, 규정을 절대로 생성하지 마세요.\n"
                "3. 문서에 'url' 또는 'source' 메타데이터가 없는 경우, 그 항목은 생략하세요.\n"
                "4. '참고문헌' 섹션에는 실제 문서 메타데이터에 존재하는 title, url만 표시하세요.\n"
                "5. 문서 내용이 불충분할 경우, '해당 문서에서는 관련 정보를 찾을 수 없습니다.' 라고 명시하세요.\n"
                "질문이 드론과 무관할 경우, 반드시 다음처럼 답하세요: 이 시스템은 드론 관련 정보만 제공합니다.\n"
                "=== 참고 문서 시작 ===\n\n"
                "{context}\n"
                "=== 참고 문서 끝 ===\n\n"
            )
            user_template = f"({ '한국어로' if lang=='ko' else 'in English' }) 사용자 질문: {query}\n\n요약하고 단계별로 설명하세요."
            system_prompt = system_template.format(context=context_text)
            user_prompt = user_template.format(query=query)

            # Compose messages according to model interface
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            st.markdown("### ✨ 실시간 답변")
            response_area = st.empty()
            partial = ""
            try:
                with st.spinner("AI가 답변 중입니다..."):
                    for chunk in llm.stream(messages):
                        token = getattr(chunk, "content", "")
                        partial += token
                        response_area.markdown(partial)
                answer_text = partial
            except Exception as e:
                st.error(f"LLM 호출 실패: {e}")
                answer_text = f"오류로 인해 답변을 생성할 수 없습니다: {e}"

            # Save history
            st.session_state["chat_history"].insert(0, {"q": query, "a": answer_text})
            st.rerun()

with col2:
    if st.button("🧾 대화 기록 초기화", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

# Show chat history
st.markdown("### 대화 기록")
if "chat_history" in st.session_state:
    for i, chat in enumerate(st.session_state["chat_history"], start=1):
        st.markdown(f"**{i}️⃣ 질문:** {chat['q']}")
        st.markdown(f"💬 {chat['a']}")
        st.markdown("---")

# Optional: allow uploading new doc and indexing in-memory
st.markdown("### 문서 업로드 및 인덱스에 추가 (선택)")
uploaded_files = st.file_uploader("PDF / 텍스트 파일 업로드 (여러개 가능)", accept_multiple_files=True)
if uploaded_files:
    if st.button("➕ 업로드 문서 인덱싱 (메모리)"):
        embeddings = get_embeddings(use_ollama=USE_OLLAMA_DEFAULT)
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

