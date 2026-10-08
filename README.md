# Potential Talents Candidate Ranking Model

![Python](https://img.shields.io/badge/Python-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blueviolet)

## Project Overview

Recruitment teams spend significant time manually reviewing candidate profiles to decide who is most relevant to a role.

This project develops an **NLP and LLM-based candidate ranking system** that progressively explores different methods for matching candidate job titles against a recruitment search, and finishes with a hybrid ranking model and a FAISS-powered search layer that re-ranks candidates when a recruiter **stars** one.

Rather than relying on exact keyword matching alone, the system compares:

- TF-IDF
- GloVe
- Word2Vec
- FastText
- BERT
- Sentence Transformers
- Llama 3.2
- Qwen 2.5 3B
- GPT-OSS (via the Groq API)

and combines the strongest approaches to produce a more balanced ranking.

The project was completed as part of my **Apziva AI residency**.

---

# Business Problem

Recruitment teams need to quickly identify candidates who are likely to be relevant to a particular role.

A simple keyword search can miss candidates who use different terminology, while a more sophisticated model can introduce its own ranking and calibration issues.

The objective of this project is to answer:
> **How can NLP and modern language models be used to rank potential candidates according to their relevance to a recruitment search?**

The workflow also has to reflect how recruiters really work: after reviewing the ranked list, a recruiter might choose the 7th candidate rather than the first. That choice should feed back into the ranking, so the system supports **starring** a candidate as the ideal match and re-ranks the list around it.

The system is designed as an **initial screening and prioritisation tool**, rather than an automated hiring decision-maker.

---

# Dataset

The dataset contains **104 candidate profiles** (**52 unique job titles**) and includes:

- Candidate ID
- Job Title
- Location
- Number of Professional Connections
- `fit` field

The primary feature used for semantic ranking is the candidate's **job title**.

Recruitment Search:

```
Aspiring Human Resources
```

The `fit` field contains no labelled values, so this project does not train a conventional supervised classification model. Instead, it investigates semantic similarity and LLM-based relevance scoring, and the final pipeline writes its own ranking score into the `fit` column.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-Learn
- Gensim
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- TF-IDF
- GloVe
- Word2Vec
- FastText
- BERT
- all-MiniLM-L6-v2
- Ollama
- Llama 3.2
- Qwen 2.5 3B
- Groq API
- GPT-OSS
- FAISS
- ChromaDB

---

# Project Workflow

```
Data Collection
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Candidate Job Title Preparation
        │
        ▼
TF-IDF Baseline
        │
        ▼
Word Embeddings
GloVe / Word2Vec / FastText
        │
        ▼
Transformer Models
BERT / Sentence Transformer
        │
        ▼
LLM-Based Candidate Scoring
Llama 3.2 / Qwen 2.5 / GPT-OSS
        │
        ▼
Model Comparison
        │
        ▼
Final Hybrid Ranking
MiniLM + GPT-OSS
        │
        ▼
FAISS RAG Search Layer
        │
        ▼
Candidate Starring and Re-Ranking
        │
        ▼
Business Recommendations
```

---

# Exploratory Data Analysis

The project begins by understanding the candidate dataset through:

- Dataset dimensions and data types
- Missing value and duplicate checks
- Candidate locations
- Professional connection counts
- Job-title frequency analysis
- Word-frequency analysis

Key findings included:

- 104 candidate records across 5 columns, with no duplicates
- Only 52 unique job titles, so many candidates share identical titles
- The job title is the main usable text feature for ranking

---

# Layer 1 – Keyword and Embedding Baselines

The first layer measures how closely each candidate's title matches the recruitment search using cosine similarity.

```
Candidate Job Titles + Recruitment Query
        │
        ▼
Vector Representation
(TF-IDF / GloVe / Word2Vec / FastText / BERT / MiniLM)
        │
        ▼
Cosine Similarity
        │
        ▼
Ranked Candidates
```

### Results

- **TF-IDF** is a fast, interpretable baseline, but titles that say "HR" score exactly 0.0
- **GloVe, Word2Vec and FastText** find the same top group of "Aspiring Human Resources" titles, but do not link "HR" to "Human Resources", so "HR Senior Specialist" can score below unrelated titles
- **FastText** compresses scores into a narrow band (even the lowest 20 titles score 0.35 to 0.61)
- **BERT** starts to handle the "HR" issue, but produces a narrow high-score band
- **Sentence Transformer (`all-MiniLM-L6-v2`)** gives the cleanest ranking of the embedding methods

| Candidate                             | MiniLM Similarity |
| ------------------------------------- | ----------------: |
| Aspiring Human Resources Professional |             0.950 |
| Aspiring Human Resources Specialist   |             0.928 |
| Seeking Human Resources Position      |             0.809 |
| Seeking Human Resources Opportunities |             0.800 |
| Human Resources Professional          |             0.795 |

MiniLM strongly favours titles whose wording is close to the query, which is why the rankings need to be checked rather than trusted from similarity scores alone.

---

# Layer 2 – LLM Candidate Scoring

The second layer asks whether generative models can judge relevance using explicit criteria rather than wording similarity.

Each candidate is scored out of 100 using only the information in the job title:

| Criterion               | Maximum Score |
| ----------------------- | ------------: |
| Role Relevance          |            25 |
| Career Intent           |            25 |
| Professional Experience |            25 |
| Overall Fit             |            25 |

Three models were tested: **Llama 3.2** and **Qwen 2.5 3B** (local, via Ollama) and **GPT-OSS** (`gpt-oss-20b`, hosted via the Groq API).

### Results

- **Llama 3.2** identified established HR professionals, but its top 20 sits in a narrow 66 to 73 band with limited separation
- **Qwen 2.5 3B** gave higher scores overall, but 12 of its top 20 candidates share the same score of 94
- **GPT-OSS** spread candidates out the most: 28 distinct scores, a top 20 running from 94 to about 69, 27 candidates scoring 22 or below (13 exactly 0) and 77 scoring 36 or higher

> A more sophisticated model does not automatically produce a better ranking.

### Method Agreement

With no labels, agreement between methods was measured using Spearman rank correlation:

| Comparison                                                   |     Spearman |
| ------------------------------------------------------------ | -----------: |
| TF-IDF, GloVe, Word2Vec, FastText and MiniLM with each other | 0.84 to 0.94 |
| BERT with those five                                         | 0.61 to 0.68 |
| Llama with the embedding methods                             | 0.04 to 0.16 |
| Qwen with the embedding methods                              | 0.09 to 0.25 |
| GPT-OSS with the embedding methods                           | 0.37 to 0.53 |
| Llama, Qwen and GPT-OSS with each other                      | 0.52 to 0.60 |

High agreement does not mean the embedding methods are right, since they all make the same mistake with "HR" titles. GPT-OSS is the LLM closest to both the embeddings and the other LLMs, which supports combining it with MiniLM.

A separate **fine-tuning experiment** (see below) also explores adapting Llama and Qwen to this candidate-scoring task, extending the project beyond prompt-based scoring.

---

# Final Hybrid Ranking

The two strongest signals sometimes ranked the same candidates very differently:

- **MiniLM** was effective at identifying semantic similarity between titles
- **GPT-OSS** could assess candidates against a structured set of recruitment criteria

The final model combines them:

1. Calculate MiniLM cosine similarity between each title and the query
2. Score each candidate with the GPT-OSS rubric (out of 100)
3. Scale both scores to a common 0–1 range
4. Blend them with equal 50/50 weights
5. Rescale the result and write it to the `fit` column

```
Final Score = 0.5 × MiniLM + 0.5 × GPT-OSS
```

Equal weights were used because there are no labelled outcomes to tune them on.

### Results

| Final rank | Title                                                         | MiniLM (rank) | GPT-OSS (rank) |   fit |
| ---------: | ------------------------------------------------------------- | ------------: | -------------: | ----: |
|          1 | Aspiring Human Resources Specialist (5 candidates)            |    0.928 (2)  |        94 (1)  | 1.000 |
|          2 | Aspiring Human Resources Professional, long title ("Passio…") |    0.749 (9)  |        74 (5)  | 0.780 |
|          3 | Skills-list title ("Human Resources, Conflict Management…")   |   0.568 (26)  |        92 (2)  | 0.769 |
|          4 | Aspiring Human Resources Professional (7 candidates)          |    0.950 (1)  |       49 (26)  | 0.762 |
|          5 | Human Resources Professional                                  |    0.795 (5)  |       65 (15)  | 0.757 |

MiniLM placed *Aspiring Human Resources Professional* first because its wording matches the query, while GPT-OSS gave it 49. The almost identical *Aspiring Human Resources Specialist* received 94. The reverse happened with the skills-list title (26th for MiniLM, 2nd for GPT-OSS). The blend balances both.

LLM scores also vary between models: Llama scored the two aspiring-HR titles at 62 each, while Qwen scored them 73 and 77. A single GPT-OSS score should not be treated as an objective measure of candidate quality, and the final `fit` is a **combined relevance signal, not a probability of hiring success**.

---

# FAISS RAG Search and Candidate Starring

A recruiter might not choose the first candidate in the list. The brief asks for the list to **re-rank every time a candidate is starred**, with a starred candidate treated as the ideal match.

Candidate titles are embedded with `all-MiniLM-L6-v2` and stored in a **FAISS** index (`IndexFlatIP` on normalised vectors, so inner product equals cosine similarity). Only the title is embedded, while `id`, `location` and `connection` are kept as metadata, and identical titles are collapsed into one row listing every candidate ID.

Starring does not edit any candidate. It **moves the search query** towards the starred candidate and searches again:

```
New Query = α × Query Embedding + (1 − α) × Starred Candidate Embedding
```

```
Recruiter Query ──┐
                  ├──► Blended Query Vector ──► FAISS Search
Starred Candidate ┘                                  │
                                                     ▼
                                  Blend with GPT-OSS score (50/50)
                                                     │
                                                     ▼
                           Starred candidate pinned to fit = 1.0
                                                     │
                                                     ▼
                                  Re-ranked list, written to `fit`
```

- `α = 0.5` by default (`α = 1` ignores stars, `α = 0` follows only stars)
- Multiple stars are averaged
- A starred candidate, and anyone sharing the same title, is pinned to `fit = 1.0`, and everyone else is capped at 0.99
- `rank_candidates("keyword", star_ids=[...])` also works for new keywords, and `use_llm=False` runs on embeddings alone with no API calls

### Results

Starring **"Human Resources Professional" (id 74)** for the search "Aspiring Human Resources":

| Title                                                 | Rank before | Rank after | Fit before | Fit after |
| ----------------------------------------------------- | ----------: | ---------: | ---------: | --------: |
| Human Resources Professional (starred)                |           5 |          1 |      0.757 |     1.000 |
| Human Resources, Staffing and Recruiting Professional |           6 |          4 |      0.744 |     0.781 |
| Director Human Resources at EY                        |          15 |          9 |      0.660 |     0.687 |
| Aspiring Human Resources Professional (long title)    |           2 |          5 |      0.780 |     0.774 |
| Aspiring Human Resources Professional                 |           4 |          6 |      0.762 |     0.764 |

Starring pulled established HR titles up and pushed the "aspiring" titles down. Following the brief's example, starring the **7th-ranked candidate** ("Human Resources Coordinator at InterContinental") put it first and moved "Aspiring Human Resources Professional" from 4th to 7th.

### ChromaDB Comparison

The same retrieval and starring logic was repeated on **ChromaDB** using identical MiniLM vectors, with `location` and `connections` as metadata. The notebook checks that FAISS and ChromaDB agree on similarity and top-10 overlap. A vector database adds **filtering during search** (for example, one location or 500+ connections) but does not change fit scores.

### Limitations

- `fit` is a relative score: the top candidate always gets 1.0 (even for "data science", where the best match scored only 0.491 similarity, because the dataset has no data science candidates)
- Starring shifts the query rather than learning from feedback
- With no labels, results are judged by whether the re-ranking looks sensible, not by measured accuracy

---

# Experiment – LoRA / QLoRA Fine-Tuning

As a side experiment, I tested whether small open models could be **fine-tuned to output rubric scores directly**, instead of relying on prompting alone. This is documented in `Fine_Tuning_Colab_Notebook.ipynb` and was run on a Google Colab T4 GPU.

- **Models:** `Llama-3.2-3B-Instruct` and `Qwen2.5-3B-Instruct`
- **Method:** QLoRA (4-bit NF4) with LoRA adapters (r = 64, alpha = 16, dropout = 0.1), trained with TRL's `SFTTrainer` for 1 epoch at a learning rate of 2e-4
- **Data:** an extended dataset of **1,285 candidate titles** with screening scores (separate from the 104-candidate dataset above), split 80/20 into 1,028 training and 257 test rows
- **Prompt:** the same four-criteria rubric used for the LLM scoring layer, with the search query and candidate title as input

### Results

| Model                | Eval Loss | Mean Token Accuracy |
| -------------------- | --------: | ------------------: |
| Llama-3.2-3B (QLoRA) |     0.171 |               97.0% |
| Qwen2.5-3B (QLoRA)   |     0.146 |               97.3% |

- **Llama:** in the examples shown, the base model often replied with a long explanation and no usable number, while the fine-tuned model returns a score directly
- **Qwen:** base and fine-tuned scores were often identical or very similar, so fine-tuning added little visible improvement

### Caveats

This was an exploratory experiment and is **not part of the final ranking model**. Token accuracy is measured over the full training text, including the fixed rubric prompt, so it overstates how well the models score candidates. The candidate rankings were also generated on the full extended dataset, which includes the training rows, and there is no labelled benchmark to measure ranking quality. The results show that the models learned the output format, not that they rank candidates better than GPT-OSS.

---

# Business Impact

Compared to a manual screening process:

| Manual Screening                  | Proposed System                                         |
| --------------------------------- | ------------------------------------------------------- |
| Review all 104 candidates         | Prioritised, ranked candidate list                      |
| Exact keyword matching            | Semantic + LLM rubric ranking                           |
| Misses different terminology      | Recognises related titles ("HR", skills-list titles)    |
| Recruiter choice has no feedback  | Starring re-ranks the list around the chosen candidate  |
| Single method, single view        | Hybrid MiniLM + GPT-OSS signal                          |

The proposed system enables recruiters to:

- Prioritise potentially relevant candidates before manual review
- Surface candidates who use different terminology from the search
- Refine the list instantly by starring the best candidate
- Keep human judgement in the final decision

```
104 Candidates
      │
      ▼
Semantic / LLM Ranking
      │
      ▼
Prioritised Candidate List
      │
      ▼
Recruiter Review ──► Star a candidate ──► List re-ranks
      │
      ▼
Final Hiring Decision
```

---

# Repository Structure

```
.
├── data/
├── PotentialTalentsNotebook.ipynb
├── Fine_Tuning_Colab_Notebook.ipynb
├── final_merged_model/
├── groq_scores.json
├── README.md
└── final_merged_model.zip
```

`PotentialTalentsNotebook.ipynb` contains the full ranking pipeline, from exploratory analysis through the hybrid model and FAISS starring. `Fine_Tuning_Colab_Notebook.ipynb` documents the LoRA / QLoRA fine-tuning experiment, and `final_merged_model/` holds the associated model artifacts. `groq_scores.json` caches the GPT-OSS scores so results can be reproduced without repeating API calls.

---

# Future Improvements

Potential future work includes:

- Candidate skills and employment history
- Matching against full job descriptions
- Supervised learning-to-rank using recruiter-labelled data and accumulated stars
- Precision@K, Recall@K, MRR and NDCG evaluation
- LLM score calibration
- LLM-generated explanations for recommendations
- RAG over full candidate profiles
- Deployment through an API or recruitment dashboard with a star button

---

# Key Skills Demonstrated

- Natural Language Processing
- Semantic Search
- Retrieval-Augmented Generation (RAG)
- Vector Search with FAISS and ChromaDB
- Text Representation
- Word Embeddings
- Transformer Models and Sentence Transformers
- Large Language Models
- LLM APIs (Groq / GPT-OSS)
- Prompt Engineering
- LLM Fine-Tuning (LoRA / QLoRA)
- LLM Evaluation
- Model Comparison
- Human-in-the-Loop Re-Ranking
- AI-Assisted Decision Systems
- End-to-End AI Workflow

---

# For Recruiters & Hiring Managers

This project demonstrates much more than applying a pretrained NLP model.

It showcases the ability to take an open-ended business problem and investigate multiple AI approaches before deciding which techniques are most useful. Rather than assuming a larger or newer language model would produce the best results, the rankings were evaluated directly, issues such as score compression and limited candidate separation were identified, and the strongest methods were combined into a hybrid model.

The project highlights experience with traditional NLP, pretrained embeddings, transformer models, local and hosted LLM scoring, vector search with FAISS and ChromaDB, a LoRA / QLoRA fine-tuning experiment on Llama and Qwen, and translating the results into a recruiter workflow where starring a candidate re-ranks the list.

The focus throughout was not simply to use the most powerful model, but to build a ranking system that is transparent about its limitations and genuinely useful to the people screening candidates.
