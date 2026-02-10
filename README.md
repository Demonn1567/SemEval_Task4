# 📚 SemEval 2025 Task 4: Multilingual Characterization of Narratives

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%2F%20CUDA-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 🌟 Project Overview

This repository contains the source code and methodology for our submission to **SemEval 2025 Task 4**. The goal of this shared task is to develop systems capable of understanding and comparing complex narratives, moving beyond simple keyword matching to grasp **abstract themes**, **narrative arcs**, and **plot outcomes**.

Our winning approach leverages a **Hybrid Ensemble Architecture** that synergizes the long-context understanding of modern Rerankers with the semantic precision of Sentence Transformers.

---

## 🎯 Task Description

The competition is divided into two distinct tracks focused on narrative similarity:

### 🏛️ Track A: Pairwise Similarity (The Judge)
* **Input:** A `Main Story` and two candidates (`Story A`, `Story B`).
* **Objective:** Determine which candidate story is strictly "closer" to the Main Story in terms of narrative themes and outcomes.
* **Output:** A binary classification (`True` if A is closer, `False` if B is closer).

### 🔍 Track B: Retrieval (The Search Engine)
* **Input:** A single `Main Story`.
* **Objective:** Generate a dense vector representation (embedding) of the story.
* **Constraint:** The embedding must map the story to a vector space where it is closest to other stories sharing the same abstract themes and structural outcomes.

---

## 🚀 Methodology: The "Team of Experts"

We moved beyond standard single-model baselines by implementing an **Ensemble Strategy**. Our hypothesis was that no single model can capture both the "vibe" (semantic similarity) and the "plot" (narrative structure) simultaneously.

### 1️⃣ The Deep Reader (Context Expert)
* **Model:** `jinaai/jina-reranker-v2-base-multilingual`
* **Role:** Acts as the "Reader." It supports a massive **8192-token context window**, allowing it to read stories from start to finish.
* **Why it matters:** Most models (like BERT) cut off after 512 tokens, missing the ending. Jina ensures the *outcome* of the story is factored into the similarity score.

### 2️⃣ The Semantic Judge (Vibe Expert)
* **Model:** `cross-encoder/stsb-roberta-large` (or `BAAI/bge-reranker-v2-m3`)
* **Role:** Focuses on **Semantic Textual Similarity (STS)**. It excels at detecting subtle tonal and thematic overlaps between text segments.
* **Mechanism:** It acts as a "sanity check" for the Deep Reader, ensuring that the chosen story feels stylistically similar to the query.

### 3️⃣ The Instruction Tuned Retriever (Track B)
* **Model:** `dunzhang/stella_en_400M_v5` & `jinaai/jina-embeddings-v3`
* **Strategy:** We utilized **Instruction Tuning** with a custom prompt:
    > *"Retrieve a story that shares the same abstract theme, narrative flow, and final outcome."*
* **Configuration:** Inference forced to a **4096-token context** to capture the full narrative arc.

---

## 🛠️ Models Used

| Track | Model Name | Role | Context Window |
| :--- | :--- | :--- | :--- |
| **A** | `jinaai/jina-reranker-v2` | Long-Context Reranking | 8192 Tokens |
| **A** | `stsb-roberta-large` | Semantic Similarity | 512 Tokens |
| **A** | `BAAI/bge-reranker-v2-m3` | Multilingual Judge | 8192 Tokens |
| **B** | `jinaai/jina-embeddings-v3` | Narrative Embedding | 8192 Tokens |
| **B** | `dunzhang/stella_en_400M_v5` | Instruction Following | 8192 Tokens |

---

## 💻 Installation & Setup

### Prerequisites
* **Python 3.9+**
* **PyTorch** (MPS enabled for Mac / CUDA for NVIDIA)

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/semeval-task4-narratives.git](https://github.com/yourusername/semeval-task4-narratives.git)
cd semeval-task4-narratives