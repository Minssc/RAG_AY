# 🚁 팀 애용 RAG 프로젝트

> **FAISS + LangChain + RAG** 를 활용한 **드론 정보 챗봇 프로젝트**

---

## 🧩 Requirements

```bash
pip install faiss-cpu langchain langchain-core langchain-ollama langchain-community streamlit pymupdf tqdm pypdf langdetect cryptography
```

---

## 📖 프로젝트 소개

**팀 애용 RAG 프로젝트**는 드론 관련 법령, 가이드라인, 기술 문서를 기반으로  
**질의응답이 가능한 RAG(Retrieval-Augmented Generation) 챗봇**을 구축한 프로젝트입니다.  

사용자는 Streamlit 웹 인터페이스를 통해 문서를 업로드하거나  
사전 구축된 데이터베이스(FAISS 벡터 스토어)에 질의할 수 있습니다.  
이를 통해 드론 운용, 안전 규정, SDK 문서 등 다양한 정보를 **자연어로 쉽게 탐색**할 수 있습니다.

---

## 👥 팀원 소개

| 이름 | GitHub | 역할 |
|:------:|:------:|:------:|
| **김대용** | [@Dae-Yong-Kim](https://github.com/Dae-Yong-Kim) | 데이터 수집 |
| **김민성** | [@Minssc](https://github.com/Minssc) | README 작성 |
| **여정인** | [@saikey-gintama](https://github.com/saikey-gintama) | 데이터 수집 |
| **윤영진** | [@YunYoungJin](https://github.com/YunYoungJin) | 코드 작성 |
| **정지훈** | [@JJhun26](https://github.com/JJhun26) | 발표 자료 (PPT) |

---

## ⚙️ 기술 스택

| 분류 | 기술 | 설명 |
|:------|:------|:------|
| **임베딩 저장소** | **FAISS** | 대규모 벡터 데이터를 효율적으로 검색하기 위한 고성능 벡터 데이터베이스 |
| **프레임워크** | **LangChain** | LLM(대형 언어모델) 기반 파이프라인 구성 및 문서 검색-응답 통합 |
| **검색 + 생성 구조** | **RAG (Retrieval-Augmented Generation)** | 문서 검색 결과를 기반으로 LLM이 신뢰성 있는 응답을 생성 |
| **LLM 백엔드** | **Ollama** | 로컬에서 실행 가능한 오픈소스 LLM 서버 |
| **UI** | **Streamlit** | 웹 기반 대화 인터페이스 구성 |
| **문서 파서** | **PyMuPDF / pypdf** | PDF 문서 내용을 텍스트로 변환 및 처리 |
| **보조 도구** | **tqdm** | 데이터 처리 과정 진행률 시각화 |

---

## 💡 주요 기능

- 📂 **문서 업로드** — 사용자가 직접 PDF, Markdown 등 문서를 업로드 가능  
- 🔍 **FAISS 기반 검색** — 문서 내용을 벡터로 변환해 유사도 기반 검색  
- 🤖 **LLM 질의응답 (RAG 구조)** — 검색된 문서를 기반으로 신뢰성 있는 AI 답변 생성  
- 🌐 **스트리밍 응답 지원** — AI 답변을 실시간으로 출력  
- 🗂️ **다국어 지원** — 한국어/영어 질의 모두 가능  
- 🔎 **필터링/검색 옵션** — 카테고리별 검색 가능
- 🧠 **로컬 LLM 통합** — 인터넷 연결 없이 Ollama 로컬 모델을 통한 질의응답  

---

## 🚀 설치 및 실행 방법

1. **필요 패키지 설치**
   ```bash
   pip install faiss-cpu langchain langchain-core langchain-ollama langchain-community streamlit pymupdf tqdm pypdf
   ```

2. **Ollama 서버 실행**
   ```bash
   ollama serve(start-ollama.sh)
   ```

3. **Streamlit 앱 실행**
   ```bash
   streamlit run v1.py
   ```
   또는
   ```bash
   streamlit run v2.py
   ```

4. **웹 접속**
   ```
   http://localhost:8501
   ```

---

## 🧭 한계점 및 개선 방향

| 한계점 | 개선 방향 |
|:--------|:-----------|
| **임베딩 데이터의 품질 한계** | 텍스트 전처리 및 문서 파서 개선 (예: PDF 이미지 해석) |
| **임베딩 속도 느림** | XPU 가속 가능한 임베딩 프레임워크 사용 |
| **응답 지연** | 스트리밍 처리 및 비동기 검색 구조 도입 |
| **데이터 업데이트 어려움** | 자동 크롤러/문서 동기화 기능 추가 |
| **LLM 응답 신뢰성 부족** | 출처 인용 및 답변 근거 표시 기능 추가 |
| **UI 단조로움** | Streamlit 커스텀 컴포넌트 활용한 직관적 인터페이스 개선 |

---

## 🧱 추가 구현 기능

- ⚡**스트리밍 응답: AI 답변을 실시간으로 출력**
- 📎**파일 업로드: 사용자가 직접 문서 업로드 가능**
- 🌏**다국어 지원: 영어/한국어 등 다국어 응답**
- 🗂️**필터링/검색 옵션: 카테고리별, 난이도별 등 필터 기능**

---

## 📚 데이터 출처

### 🏢 [대한드론진흥협회](https://kodpa.or.kr)
- [📄 일상 속 드론 운용 가이드라인.pdf](https://kodpa.or.kr/article/%ED%98%91%ED%9A%8C-%EC%95%8C%EB%A6%BC/2/2365#none)

---

### ⚖️ [국가법령정보센터](https://www.law.go.kr/main.html)
- [항공안전법 (법률 제20981호, 2025.08.28)](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%ED%95%AD%EA%B3%B5%EC%95%88%EC%A0%84%EB%B2%95/(20250828,20981,20250527))  
- [드론 활용의 촉진 및 기반조성에 관한 법률 (법률 제21065호, 2025.10.01)](https://www.law.go.kr/%EB%B2%95%EB%A0%B9/%EB%93%9C%EB%A1%A0%ED%99%9C%EC%9A%A9%EC%9D%98%EC%B4%89%EC%A7%84%EB%B0%8F%EA%B8%B0%EB%B0%98%EC%A1%B0%EC%84%B1%EC%97%90%EA%B4%80%ED%95%9C%EB%B2%95%EB%A5%A0)  
- [무인비행장치 특별비행을 위한 안전기준 및 승인절차에 관한 기준 (국토교통부고시 제2023-343호, 2023.06.30)](https://law.go.kr/LSW/admRulLsInfoP.do?admRulSeq=2100000225206)

---

### 🛫 [국토교통부](https://www.molit.go.kr/portal.do#mltm)
- [📘 드론제도 가이드북 (2017)](https://www.molit.go.kr/LCMS/DWN.jsp?fold=/phc0408/&fileName=%EB%93%9C%EB%A1%A0%EC%A0%9C%EB%8F%84+%EA%B0%80%EC%9D%B4%EB%93%9C%EB%B6%81+2017-%EB%B3%B5%EC%82%AC.pdf)

---

### 🛰️ [드론 원스톱 민원서비스](https://drone.onestop.go.kr/)
- [📑 2022.12.01 항공촬영 지침서 개정안 전문.hwp](https://drone.onestop.go.kr/board/notice/read?id=108)

---

### ⚙️ [Ryze Robotics (Tello)](https://www.ryzerobotics.com/kr)
- [Tello SDK Documentation EN_1.3.pdf](https://dl-cdn.ryzerobotics.com/downloads/tello/20180910/Tello%20SDK%20Documentation%20EN_1.3.pdf)  
- [Tello User Manual V1.0_KR.pdf](https://dl-cdn.ryzerobotics.com/downloads/Tello/201806mul/Tello%20User%20Manual%20V1.0_KR.pdf)  
- [Tello SDK 2.0 User Guide.pdf](https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf)

---

### 🧭 [Pixhawk](https://pixhawk.org)
- [Pixhawk Autopilot Reference Standard.pdf](https://discuss.px4.io/uploads/short-url/hDRYO5hxdNfbmNhWvnSVrpAayuB.pdf)

---

### 🔌 [Raritan](https://www.raritan.com/)
- [PDU_G4_UG_D1_4.2.0.pdf](https://cdn1.raritan.com/download/pdu-g4/4.2.0/PDU_G4_UG_D1_4.2.0.pdf)

---

### 🛠️ [QGroundControl](https://qgroundcontrol.com/)
- [📚 공식 문서 (GitHub)](https://github.com/mavlink/qgroundcontrol/tree/master/docs)

---

### 🛠️ *[Betaflight](https://www.betaflight.com/)**
- [📚 공식 문서 (GitHub)](https://github.com/betaflight/betaflight.com/tree/master/docs)

---

### 🛠️ *[Ardupilot](https://ardupilot.org/ardupilot/)**
- [📚 공식 문서 (GitHub)](https://github.com/ArduPilot/ardupilot_wiki/tree/master/copter/source/docs)


\* *임베딩 시간 단축을 위해 제외*
