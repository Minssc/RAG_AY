#!/usr/bin/env python3
"""
📊 JSON → FAISS 인덱스 생성기
드론 문서 JSON을 벡터스토어로 변환합니다.
"""

import os
import json
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ==============
# 설정
# ==============
JSON_PATH = "drone_docs.json"
INDEX_DIR = "vector_index"
USE_OLLAMA = True  # 로컬 LLM을 쓸 경우 True, OpenAI API를 쓸 경우 False

# ==============
# 변환 로직
# ==============
def build_vectorstore():
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"{JSON_PATH} 파일이 존재하지 않습니다.")

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data["documents"]:
        doc = Document(
            page_content=item["content"],
            metadata={
                "title": item.get("title"),
                "category": item.get("category"),
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
            },
        )
        docs.append(doc)

    print(f"📄 총 {len(docs)}개의 문서를 로드했습니다.")

    if USE_OLLAMA:
        embeddings = OllamaEmbeddings(model="llama3.1:8b")
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print(f"✅ 벡터스토어 저장 완료: {INDEX_DIR}/")

if __name__ == "__main__":
    build_vectorstore()
