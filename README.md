
# 🎵 Melodia - Music Recommender & Explainer (LLM Fine-tuning Project)

This project aims to build a **music recommender / explainer assistant** powered by a fine-tuned LLM.  
The assistant takes a user’s music taste (genres, moods, favorite artists) and recommends songs with **contextual explanations** (tags, history, lyrics meaning, etc.).

---

## 📌 Project Pipeline

### 1. Data Collection

#### 1.1 Metadata (tags, genres, moods, related artists)
- **[Last.fm API](https://www.last.fm/api)** → song/artist tags, related artists.  
- **[Discogs API](https://www.discogs.com/developers/)** → genres, labels, album metadata.  

#### 1.2 Explanations (lyrics, context, reviews)
- **[Genius API](https://docs.genius.com/)** → lyrics + annotations/explanations.  
- **Wikipedia** → artist/album pages for history and cultural impact.  

---

### 2. Dataset Generation (QA Style)

Raw data (tags, metadata, explanations) will be converted into **instruction–response pairs**.  
We will use a **Mistral agent** to generate QA samples automatically.

#### Input data (shortened version)
```json
Song: "Smells Like Teen Spirit"  
Tags: grunge, rock, 90s, alternative  
Description: Anthem of Generation X, Nirvana’s breakout hit in 1991.
```

#### Example 1: Recommender
```json
{
  "conversations": [
    {"from": "user", "value": "I like 90s grunge rock. What song should I listen to?"},
    {"from": "assistant", "value": "You might enjoy 'Smells Like Teen Spirit' by Nirvana. It’s a grunge anthem from the 90s that defined the alternative rock scene."}
  ]
}
```

#### Example 2: Explainer
```json
{
  "conversations": [
    {"from": "user", "value": "Can you explain 'Smells Like Teen Spirit'?"},
    {"from": "assistant", "value": "It was released in 1991 as Nirvana’s breakout hit. Known as an anthem for Generation X, its raw energy and distorted guitars shaped the grunge movement."}
  ]
}
```

The dataset will be stored in **JSONL format**.

---

### 3. Fine-tuning

- **Base model**: [`TinyLlama/TinyLlama-1.1B-intermediate`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)  
- **Training method**:  
  - QLoRA + `bitsandbytes` (4-bit quantization)  
  - PEFT (LoRA adapters)  
  - Gradient checkpointing + CPU offloading (to fit in 4 GB VRAM)  
- **Steps**:  
  1. Phase 1: Instruction tuning (Alpaca-style).  
  2. Phase 2: Specialization on **music QA dataset**.  

---

### 4. Deployment (Optional)

If time allows:  
- Wrap the fine-tuned model in a **chat interface** (React frontend or [Gradio](https://www.gradio.app/) demo).  
- Example demo prompts:  
  - *“I like chill indie with female vocals, what should I listen to?”*  
  - *“Can you explain why ‘Bohemian Rhapsody’ is so unique?”*  
  - *“I love electronic music with strong bass, recommend me 3 songs.”*

---

## 🚀 Why this project?

- **Fun & relatable**: everyone has music preferences.  
- **End-to-end pipeline**: scraping → dataset creation → fine-tuning → demo.  
- **Custom tone**: recommendations + explanations, not just raw metadata.  
- **Great showcase**: demonstrates skills in data engineering, LLM training, and frontend integration.  

---

## 📂 Repo Structure (planned)

```
melodia/
│── config/              # Requirements and variables
│── data/                # Raw + processed datasets (JSONL)
│── scripts/             # Scraping + preprocessing
│── training/            # Fine-tuning scripts (QLoRA)
│── output/              # Final trained model + LoRA adapters
│── frontend/            # Chat interface (optional)
│── README.md            # Project documentation
```
---
*Note: This README was co-written with an LLM for clarity and structure.*
