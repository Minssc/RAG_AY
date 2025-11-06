#!/usr/bin/env python3
"""
🚁 Drone Info Assistant — RAG 기반 Streamlit 앱
- 폴더 내 PDF를 자동 로드하여 벡터 인덱스 구축
- Ollama 또는 OpenAI 기반 LLM 지원 가능
"""

import os
import glob
from pathlib import Path
import faiss
import streamlit as st
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# =====================================
# 🔧 기본 설정
# =====================================
PDF_DIR = "drone_pdfs"      # PDF 폴더명 (donre_pdfs → 오타 수정)
TEXT_DIR = "data"
INDEX_DIR = "faiss_index"   # 인덱스 저장 폴더
USE_OLLAMA_DEFAULT = True   # 기본 설정값

st.set_page_config(page_title="🛸 Drone Info Assistant", page_icon="🛸", layout="wide")
st.title("🛸 Drone Info Assistant — RAG 기반 문서 검색 & 답변")


# =====================================
# 📄 DATA 로드
# =====================================
def load_all_data(pdf_dir: str, text_dir: str = "data"):
    text_paths = []
    for file_path in Path(text_dir).rglob('*'):
        if file_path.suffix.lower() not in ['.md', '.rst']:
            continue
        text_paths.append(file_path.as_posix())

    pdf_paths = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_paths:
        st.error(f"❌ PDF 폴더({pdf_dir})에 파일이 없습니다.")
        return [] # skip text folder checking

    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        try:
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = os.path.basename(pdf_path)
            all_docs.extend(docs)
            st.info(f"📘 {os.path.basename(pdf_path)} 로드 완료 ({len(docs)} 페이지)")
        except Exception as e:
            st.warning(f"⚠️ {pdf_path} 로드 실패: {e}")

    for text_path in text_paths:
        with open(text_path, 'r') as f:
            content = f.read()
        all_docs.append(Document(
            page_content=content,
            metadata={
                "source": os.path.basename(text_path),
                "path": text_path,
                "type": "text"
            }           
        ))

    st.info(f"📘 data 폴더 내 md,rst 파일 로드 완료: ({len(text_paths)}) 파일")
        
    return all_docs


# =====================================
# 🧠 벡터스토어 구축
# =====================================
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    all_docs = load_all_data(PDF_DIR)
    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    st.success(f"총 {len(splits)}개 문서 청크 생성 완료!")

    class InstructEmbeddings(OllamaEmbeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            instructed_texts = [f"Represent this sentence for retrieval: {t}" for t in texts]
            return super().embed_documents(instructed_texts)

        def embed_query(self, text: str) -> List[float]:
            instructed_text = f"Represent this sentence for retrieval: {text}"
            return super().embed_documents([instructed_text])[0]

    embeddings = InstructEmbeddings(model="llama3.1:8b")
    embedding_dim = len(embeddings.embed_query("테스트"))
    index = faiss.IndexFlatL2(embedding_dim)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    batch_size = 100
    for i in range(0, len(splits), batch_size):
        vectorstore.add_documents(splits[i:i + batch_size])

    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)
    st.success(f"✅ FAISS 인덱스 생성 완료 및 {INDEX_DIR}/ 에 저장됨")
    return vectorstore


# =====================================
# 💾 인덱스 로드
# =====================================
def load_vectorstore(index_dir: str):
    if not os.path.exists(index_dir):
        return None

    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    try:
        vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.error(f"인덱스 로드 실패: {e}")
        return None


# =====================================
# 🧩 LLM 생성기 (Ollama / Exaone 선택 가능)
# =====================================
def get_llm(use_ollama=True, temperature=0.3):
    if use_ollama:
        return ChatOllama(model="exaone3.5:7.8b", temperature=temperature)


# =====================================
# 📋 도우미 함수
# =====================================
def docs_to_display_text(docs: List[Document]) -> str:
    """검색된 문서들을 프롬프트용 문자열로 변환"""
    parts = []
    for d in docs:
        title = d.metadata.get("title", d.metadata.get("source", "unknown"))
        chunk = d.metadata.get("chunk_id", "")
        parts.append(f"[{title} - chunk {chunk}]\n{d.page_content}")
    return "\n\n".join(parts)


# =====================================
# 🧭 사이드바 설정
# =====================================
with st.sidebar:
    st.header("⚙️ 설정")
    use_ollama = st.checkbox("Use Ollama (로컬 LLM)", value=USE_OLLAMA_DEFAULT)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05)
    k_retrieve = st.number_input("검색할 문서 수 (k)", 1, 10, 4)

    st.markdown("---")
    st.subheader("📦 인덱스 관리")

    if st.button("인덱스 새로 구축"):
        st.session_state["vectorstore"] = build_vectorstore()

    if st.button("인덱스 로드"):
        st.session_state["vectorstore"] = load_vectorstore(INDEX_DIR)
        if st.session_state["vectorstore"]:
            st.success("✅ 인덱스 로드 완료")
        else:
            st.warning("인덱스를 찾을 수 없습니다.")

# =====================================
# 🔍 질의응답
# =====================================
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = load_vectorstore(INDEX_DIR)

vectorstore = st.session_state.get("vectorstore")
if vectorstore is None:
    st.warning("❗ 인덱스가 없습니다. 사이드바에서 새로 구축하거나 로드하세요.")
    st.stop()

query = st.text_input("드론 관련 질문을 입력하세요 (예: 비행 허가 절차는 어떻게 되나요?)")

if st.button("🔎 검색 및 답변"):
    if not query.strip():
        st.warning("질문을 입력하세요.")
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": k_retrieve})
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        st.warning("관련 문서를 찾을 수 없습니다.")
        st.stop()

    with st.expander(f"🔍 검색된 문서 ({len(retrieved_docs)}개)"):
        for d in retrieved_docs:
            st.markdown(f"**{d.metadata.get('source','unknown')}**")
            st.write(d.page_content[:500] + "...")

    context_text = docs_to_display_text(retrieved_docs)

    system_prompt = (
        "당신은 드론 관련 법규 및 기술 문서 분석 전문가입니다. "
        "제공된 문서 외의 정보나 URL을 생성하지 마세요.\n\n"
        f"=== 참고 문서 ===\n{context_text}\n=== 끝 ==="
        "엄격한 지침:\n"
        "1. 위 문서의 내용만을 바탕으로 답변하세요.\n"
        "2. 문서에 포함되지 않은 URL, 기관명, 수치, 규정을 절대로 생성하지 마세요.\n"
        "3. 문서에 'url' 또는 'source' 메타데이터가 없는 경우, 그 항목은 생략하세요.\n"
        "4. '참고문헌' 섹션에는 실제 문서 메타데이터에 존재하는 title, url만 표시하세요.\n"
        "5. 문서 내용이 불충분할 경우, '해당 문서에서는 관련 정보를 찾을 수 없습니다.' 라고 명시하세요.\n"
        "질문이 드론과 무관할 경우, 반드시 다음처럼 답하세요: 이 시스템은 드론 관련 정보만 제공합니다.\n"
        "6. 사용자의 질문 언어를 감지하고, 반드시 동일한 언어로 답변하세요."
    )

    user_prompt = f"질문: {query}\n\n문서 내용만 기반으로 실무적 답변을 작성하세요."

    llm = get_llm(use_ollama=use_ollama, temperature=temperature)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    with st.spinner("AI가 답변 중입니다..."):
        resp = llm.invoke(messages)
        answer = getattr(resp, "content", str(resp))

    st.markdown("## ✈️ 답변")
    st.write(answer)
