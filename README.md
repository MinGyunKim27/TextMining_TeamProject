# Amazon Fashion Review NLP Pipeline (BERT + GPT-2)

Amazon Fashion 리뷰 데이터셋을 기반으로 리뷰 문장을 **카테고리(배송/사이즈/색상/퀄리티)** 및 **감성(긍정/부정)** 기준으로 분류하고,  
HuggingFace Transformers 기반 GPT-2 Fine-tuning을 통해 **조건부 리뷰 텍스트 생성 모델**을 구축한 프로젝트입니다.

본 프로젝트는 데이터 전처리 → 자동 라벨링(BERT) → 학습 데이터셋 구축 → 생성 모델 학습(GPT-2) → 생성 결과 검증까지  
NLP 파이프라인을 직접 설계하고 구현하는 것을 목표로 진행했습니다.

---

## 🔥 Project Overview

- **Dataset**: Amazon Fashion Review Dataset (UCSD Amazon Review Data)
- **Goal**: 리뷰 텍스트를 구조화(카테고리/감성)하고, 조건에 맞는 리뷰 문장을 생성하는 모델 구축
- **Output**: 카테고리/감성별 리뷰 생성 결과 비교 및 실험 수행

---

## 🧩 Pipeline

본 프로젝트는 아래와 같은 흐름으로 진행되었습니다.

1. **Raw 리뷰 데이터 수집 및 EDA**
2. **텍스트 전처리 및 문장 단위 분리**
3. **BERT 기반 분류 모델 학습**
4. **전체 리뷰 데이터 자동 라벨링 (Category + Sentiment)**
5. **카테고리/감성별 학습 데이터셋 구축 (총 8개 txt/csv)**
6. **GPT-2 Fine-tuning을 통한 조건부 리뷰 생성 모델 학습**
7. **Top-k / Top-p Sampling 기반 생성 결과 비교 실험**

---

## 📊 Dataset

### Raw Data
- Amazon Fashion 리뷰 데이터 약 **165,000건**
- 주요 컬럼:
  - overall (별점)
  - verified (구매 인증 여부)
  - year
  - reviewText

### Labeling Strategy
- 리뷰 문장을 아래 4개 카테고리로 분류
  - 배송(Delivery)
  - 사이즈(Size)
  - 색상(Color)
  - 퀄리티(Quality)

- 감성(Sentiment)은 긍정/부정 기준으로 분류
  - Positive
  - Negative

최종적으로 카테고리 × 감성 기준으로 총 **8개 데이터셋**을 구축했습니다.

---

## 🧪 Preprocessing

- 정규표현식 기반 텍스트 정제 (특수문자/불필요 패턴 제거)
- 문장 단위 분리 후 학습 가능한 형태로 변환
- 카테고리/감성 라벨 기준으로 데이터셋 분할 및 txt 파일 생성

---

## 🤖 Models

### 1) BERT 기반 문장 분류 모델
- 목적: 전체 리뷰 데이터에 대해 카테고리/감성 라벨을 자동 부여하기 위한 분류 모델 구축
- 결과: 자동 라벨링 기반 데이터셋 구축에 활용

### 2) GPT-2 Fine-tuning 기반 리뷰 생성 모델
- HuggingFace Transformers 기반 GPT-2 모델을 Fine-tuning
- Trainer API를 활용하여 학습 수행
- 카테고리/감성별 텍스트 생성 결과를 비교 실험

---

## ✨ Text Generation

학습된 GPT-2 모델을 기반으로 다음과 같은 설정을 사용해 리뷰 텍스트를 생성했습니다.

- Sampling Strategy:
  - **Top-k sampling**
  - **Top-p sampling (nucleus sampling)**

생성된 결과를 샘플링하여 카테고리별 문장 생성 품질을 비교했습니다.

---

## 📌 Example Outputs

Below are sample generated sentences from the fine-tuned GPT-2 model:

### Delivery (Positive)
- "The package arrived earlier than expected and was well packed."

### Delivery (Negative)
- "The delivery was delayed and the packaging was damaged."

### Size (Positive)
- "Fits perfectly and feels comfortable to wear."

### Size (Negative)
- "The size was much smaller than expected."

*(The generated sentences are sampled results and may vary depending on sampling parameters.)*

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- PyTorch
- HuggingFace Transformers
- BERT
- GPT-2
- Google Colab

---

## 📌 Key Takeaways
- 텍스트 데이터 전처리 및 구조화 경험

- BERT 기반 분류 모델 학습 및 자동 라벨링 수행

- HuggingFace Trainer 기반 GPT-2 Fine-tuning 경험

- Top-k / Top-p Sampling 기반 생성 결과 비교 실험 수행

- NLP 모델 학습 파이프라인을 직접 설계하며 End-to-End 흐름을 경험

## 🔗 Data Source
Amazon Review Data (UCSD)
https://jmcauley.ucsd.edu/data/amazon/

## 📌 Notes
본 프로젝트는 학부 텍스트마이닝 수업 팀 프로젝트로 진행되었으며,
학습 및 실험 목적의 연구/실습 기반 프로젝트입니다.
