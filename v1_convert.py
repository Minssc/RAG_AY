#!/usr/bin/env python3
"""
📘 Drone PDF → JSON 변환기
드론 법률/매뉴얼/가이드 문서를 RAG용 JSON 포맷으로 변환
"""

import os
from pathlib import Path
import re
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str) -> str:
    """PDF에서 추출된 텍스트를 정제"""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("‐", "-").strip()
    return text


def classify_category(filename: str) -> str:
    """파일명 기반 간단한 카테고리 분류"""
    name = filename.lower()
    if "law" in name or "법" in name:
        return "법률"
    elif "manual" in name or "guide" in name or "controller" in name:
        return "매뉴얼"
    else:
        return "기타"


def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF에서 텍스트 추출 (PyMuPDF)"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        text += page_text + "\n"
    return clean_text(text)

def extract_text_from_file(md_path: str) -> str:
    """md파일에서 텍스트 추출"""
    with open(md_path, "r") as f:
        text = f.read()
    return clean_text(text)


def split_into_chunks(text: str, chunk_size=1000, overlap=150):
    """LangChain TextSplitter로 문단 분리"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)


def convert_data_to_json(pdf_folder="drone_pdfs", text_folder='data', output_path="drone_docs.json"):
    """PDF 및 md,rst파일 전체 변환"""
    all_docs = []

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    text_files = []
    for file_path in Path(text_folder).rglob('*'):
        if file_path.suffix.lower() not in ['.md', '.rst']:
            continue
        text_files.append(file_path.as_posix())

    if not pdf_files:
        print("⚠️ PDF 파일이 없습니다.")
        return

    for filename in tqdm(pdf_files, desc="PDF 변환 중"):
        pdf_path = os.path.join(pdf_folder, filename)
        category = classify_category(filename)
        text = extract_text_from_pdf(pdf_path)

        chunks = split_into_chunks(text)

        for idx, chunk in enumerate(chunks):
            all_docs.append({
                "title": filename.replace(".pdf", ""),
                "category": category,
                "chunk_id": idx,
                "content": chunk,
                "source": pdf_path
            })

    for file_path in tqdm(text_files, desc="md, rst 변환 중"):
        category = "매뉴얼" ### 
        text = extract_text_from_file(file_path)

        chunks = split_into_chunks(text)

        for idx, chunk in enumerate(chunks):
            all_docs.append({
                "title": filename.replace(".md", ""),
                "category": category,
                "chunk_id": idx,
                "content": chunk,
                "source": file_path
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"documents": all_docs}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 변환 완료: {output_path}")
    print(f"총 {len(all_docs)}개의 문단이 저장되었습니다.")

if __name__ == "__main__":
    os.makedirs("drone_pdfs", exist_ok=True)
    convert_data_to_json("drone_pdfs", 'data', "drone_docs.json")


# #!/usr/bin/env python3
# """
# 📘 PDF → JSON 변환기 (의미 기반 Chunk 분할)
# 드론 관련 법률 및 기술 매뉴얼 문서를 RAG 입력용 JSON으로 변환
# """

# import os
# import json
# import re
# import fitz  # PyMuPDF
# from typing import List, Dict

# # =============================
# # 🔧 CONFIG
# # =============================
# PDF_DIR = "drone_pdfs"           # 변환할 PDF 폴더
# OUTPUT_JSON = "drone_docs.json"
# CHUNK_SIZE = 1200          # chunk 최대 길이
# OVERLAP = 150              # 문맥 overlap (문장 단위)
# MIN_TEXT_LEN = 100         # 너무 짧은 페이지 필터링

# # =============================
# # 🧩 HELPER FUNCTIONS
# # =============================
# def extract_text_from_pdf(path: str) -> str:
#     """PDF 파일에서 텍스트 전체 추출"""
#     doc = fitz.open(path)
#     text = ""
#     for page in doc:
#         text += page.get_text("text") + "\n"
#     doc.close()
#     return text


# def smart_chunk_text(text: str, max_len: int = 1200, overlap: int = 150) -> List[str]:
#     """
#     의미 기반 텍스트 청크 분할
#     - 헤딩(‘제1조’, 숫자. , # 제목) 기준 1차 분리
#     - 너무 길면 문장 단위로 2차 분리
#     """
#     # 1️⃣ 헤딩/섹션 단위로 1차 분할
#     sections = re.split(r'\n(?=(제\s?\d+\s?조|[0-9]+\.\s|#{1,3}\s))', text)
#     chunks = []

#     for section in sections:
#         section = section.strip()
#         if len(section) < MIN_TEXT_LEN:
#             continue

#         # 2️⃣ 긴 섹션은 문장 단위로 분리
#         while len(section) > max_len:
#             split_idx = section[:max_len].rfind(".")
#             if split_idx == -1:
#                 split_idx = max_len
#             chunks.append(section[:split_idx])
#             section = section[split_idx - overlap:]
#         chunks.append(section)
#     return chunks


# # =============================
# # 🧠 MAIN CONVERSION PIPELINE
# # =============================
# def convert_pdfs_to_json(pdf_dir: str, output_path: str):
#     docs = []
#     total_chunks = 0

#     pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
#     for fname in pdf_files:
#         pdf_path = os.path.join(pdf_dir, fname)
#         print(f"📄 변환 중: {fname}")

#         try:
#             # ✅ 페이지 단위로 처리 (스트리밍 방식)
#             doc = fitz.open(pdf_path)
#             for page_num, page in enumerate(doc):
#                 text = page.get_text("text")
#                 if len(text.strip()) < MIN_TEXT_LEN:
#                     continue
#                 chunks = smart_chunk_text(text, max_len=CHUNK_SIZE, overlap=OVERLAP)
#                 for i, chunk in enumerate(chunks):
#                     docs.append({
#                         "source": fname,
#                         "page": page_num + 1,
#                         "chunk_id": f"{page_num}-{i}",
#                         "content": chunk,
#                         "title": infer_title(chunk),
#                     })
#                 total_chunks += len(chunks)
#             doc.close()

#             # ✅ 중간 저장 (10개 파일마다 flush)
#             if len(docs) > 1000:
#                 with open(output_path, "w", encoding="utf-8") as f:
#                     json.dump(docs, f, ensure_ascii=False, indent=2)
#                 docs = []  # 메모리 비움
#                 print("💾 중간 저장 완료. 메모리 정리.")

#         except Exception as e:
#             print(f"⚠️ {fname} 변환 실패: {e}")

#     # 마지막 flush
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(docs, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ 변환 완료 ({total_chunks}개 청크 생성)")

# def infer_title(chunk: str) -> str:
#     """
#     Chunk의 첫 부분에서 제목 또는 섹션명 추정
#     ex) '제3조 드론 운항 제한' → '드론 운항 제한'
#     """
#     lines = chunk.strip().split("\n")
#     first_line = lines[0][:100] if lines else ""
#     title = re.sub(r"^(제\s?\d+\s?조|[0-9]+\.\s|#\s*)", "", first_line).strip()
#     return title if title else "Untitled Section"


# # =============================
# # 🚀 ENTRY POINT
# # =============================
# if __name__ == "__main__":
#     convert_pdfs_to_json(PDF_DIR, OUTPUT_JSON)
