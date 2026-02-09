SemEval 2025 Task 4: Multilingual Characterization of Narratives
🌟 Project Overview
This repository contains the source code and methodology for our submission to SemEval 2025 Task 4. The goal of this shared task is to develop systems capable of understanding and comparing complex narratives, going beyond simple keyword matching to grasp abstract themes, narrative arcs, and plot outcomes.

Our system achieves this by leveraging a Hybrid Ensemble Architecture that combines the long-context understanding of modern Rerankers with the semantic precision of Sentence Transformers.

🎯 Task Description
The competition is divided into two distinct tracks, both focused on narrative similarity:

Track A: Pairwise Similarity (The Judge)
Input: A Main Story and two candidates (Story A, Story B).

Objective: Determine which candidate story is strictly "closer" to the Main Story in terms of narrative themes and outcomes.

Output: A binary classification (True if A is closer, False if B is closer).

Track B: Retrieval (The Search Engine)
Input: A single Main Story.

Objective: Generate a dense vector representation (embedding) of the story.

Constraint: The embedding must map the story to a vector space where it is closest to other stories sharing the same abstract themes and structural outcomes.

🚀 Methodology: The "Team of Experts"
We moved beyond standard single-model baselines by implementing an Ensemble Strategy. Our hypothesis was that no single model can capture both the "vibe" (semantic similarity) and the "plot" (narrative structure) simultaneously.

1. The Deep Reader (Context Expert)
Model: jinaai/jina-reranker-v2-base-multilingual

Role: This model serves as the "Reader." It supports a massive 8192-token context window, allowing it to read stories from start to finish.

Why it matters: Most models (like BERT) cut off after 512 tokens, missing the ending. Jina ensures the outcome of the story is factored into the similarity score.

2. The Semantic Judge (Vibe Expert)
Model: cross-encoder/stsb-roberta-large (or BAAI/bge-reranker-v2-m3 in advanced runs)

Role: This model focuses on Semantic Textual Similarity (STS). It excels at detecting subtle tonal and thematic overlaps between text segments.

Mechanism: It acts as a "sanity check" for the Deep Reader, ensuring that the chosen story feels stylistically similar to the query.

3. The Retrieval Engine (Track B)
Model: jinaai/jina-embeddings-v3

Strategy: We utilized Instruction Tuning. Instead of feeding raw text, we prepended a specific prompt:

"Retrieve a story that shares the same abstract theme, narrative flow, and final outcome."

Configuration: We forced a 4096-token context window during inference to ensure the embedding represents the entire narrative arc.

🛠️ Models Used
💻 Installation & Setup
Prerequisites
Python 3.9+

PyTorch (MPS enabled for Mac / CUDA for NVIDIA)

1. Clone the Repository
2. Install Dependencies
3. Download Models (Optional but Recommended)
To avoid runtime downloads, you can pre-fetch the models using HuggingFace CLI:

🏃‍♂️ Usage
Running the Best Performing System (Ensemble)
To reproduce our highest score (0.63), run the ensemble script. This script automatically:

Loads the "Deep Reader" and "Semantic Judge" models.

Normalizes their scores to a 0-1 scale.

Computes a weighted average vote (60% Context / 40% Semantics).

Generates the submission.zip file.

Running the "God Mode" (High Precision)
For maximum accuracy using BGE-M3 and Stella v5 with custom config patching (to bypass xformers issues on Mac):

📊 Results
🍎 Apple Silicon (M-Series) Optimization
This project is fully optimized for Apple M1/M2/M3/M4 chips.

MPS Acceleration: All scripts detect torch.backends.mps.is_available() and automatically switch to the mps device.

Memory Safety: Inference runs with batch_size=1 by default to prevent RAM overflow on unified memory systems.

FP16 Inference: Models are loaded in half-precision (torch.float16) where supported to double the speed.