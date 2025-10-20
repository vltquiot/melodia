# 🎵 Melodia - Music Recommender & Explainer (LLM Fine-tuning Project)

This project builds a **French music recommendation and explanation assistant** powered by a fine-tuned LLM with RAG (Retrieval-Augmented Generation). The assistant recommends songs and provides contextual explanations based on user preferences, enhanced by real-time knowledge retrieval from a music database.

---

## 📌 Project Overview

Melodia combines:
- **Fine-tuned TinyLlama model** - Specialized for French music Q&A
- **RAG system with FAISS** - Retrieves relevant information from artist/track databases
- **QLoRA training** - Memory-efficient 4-bit quantization for training on limited hardware
- **Interactive CLI** - Simple command-line interface for local usage

---

## 🛠️ Implementation Pipeline

### 1. Data Collection

#### 1.1 Track Metadata
- **[Discogs API](https://www.discogs.com/developers/)** - Retrieved metadata for French songs (genres, labels, release dates, etc.)
- Output: `tracks.jsonl` with comprehensive track information

#### 1.2 Knowledge Base for RAG
- **Wikipedia API** - Retrieved contextual information for:
  - Individual tracks → `data/tracks_infos/*.txt`
  - Artists → `data/artists_infos/*.txt`
- Purpose: Provide rich context for the RAG system to enhance model responses

---

### 2. Dataset Generation (QA Format)

The training dataset was generated using the **ChatGPT API** to create question-answer pairs in French from the collected track metadata.

#### Dataset Format
```json
{
  "messages": [
    {"role": "system", "content": "Tu es un assistant musical. Utilise uniquement les informations ci-dessous."},
    {"role": "user", "content": "Qui sont les artistes principaux de la chanson 'L'avenir Est A Nous' ?"},
    {"role": "assistant", "content": "Les artistes principaux de la chanson 'L'avenir Est A Nous' sont..."}
  ]
}
```

**Key characteristics:**
- All conversations in **French**
- Mix of recommendation requests and explanatory questions
- Based on real French music catalog from Discogs
- Output: `qa_tracks_recommender.jsonl` with comprehensive track information

---

### 3. Model Fine-tuning

#### Base Model
- **[TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)**

#### Training Configuration
- **Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit with `bitsandbytes`
- **Framework**: Hugging Face PEFT (Parameter-Efficient Fine-Tuning)
- **LoRA adapters**: Saved in `output/` directory
- **Training focus**: French music recommendations and explanations

#### Training optimizations:
- 4-bit quantization for memory efficiency
- LoRA adapters to reduce trainable parameters
- Gradient checkpointing to fit in limited VRAM (4GB in my case)

---

### 4. RAG System

#### Architecture
1. **Document Processing**
   - Load all `.txt` files from `data/tracks_infos/` and `data/artists_infos/`
   - Split documents into chunks (500 chars with 50 char overlap)

2. **Embedding & Indexing**
   - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
   - Vector store: FAISS (Facebook AI Similarity Search)
   - Batch processing (500 chunks per batch) to handle large datasets efficiently

3. **Retrieval at Inference**
   - User query → Embed query → Search FAISS index
   - Retrieve top-K most relevant context chunks
   - Feed context + query to fine-tuned model
   - Generate enhanced response

#### Scripts
- **`scripts/check_gpu.py`** - Tool to check if the cuda environement is working
- **`scripts/count_lines.py`** - Tool to check the size of a given file (for the size of the jsonl files)
- **`scripts/create_faiss_indexes.py`** - Creates and saves FAISS index (run once or when data changes)
- **`scripts/create_faiss_indexes.py`** - Creates and saves FAISS index (run once or when data changes)
- **`scripts/generate_qa.py`** - Generate the QA dataset with the tracks using ChatGPT API
- **`scripts/music_recommender.py`** - Loads model + index and provides interactive Q&A
- **`scripts/parse_tracks_meta.py`** - Retrieve the french tracks metadata from Discogs API
- **`scripts/parse_wikipedia.py`** - Retrieve the txt files about all tracks and artists using Wikipedia API
- **`scripts/train_sft.py`** - Fine-tune the model and put the results in `output/`

---

## 🎯 Key Features

✅ **Specialized French music assistant** - Trained specifically on French music Q&A  
✅ **RAG-enhanced responses** - Retrieves real information from 19,000+ documents  
✅ **Memory-efficient** - QLoRA allows training on consumer hardware  
✅ **Fast inference** - Pre-built FAISS index for instant retrieval  
✅ **Local execution** - No external API calls during inference  
✅ **Interactive CLI** - Simple command-line interface for testing

---

## 📊 Technical Details

| Component | Technology |
|-----------|------------|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning | QLoRA (4-bit) with PEFT |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Chunking | RecursiveCharacterTextSplitter (500/50) |
| RAG Framework | LangChain |
| Language | French |

---

## 🔄 Potential Improvements

### Data Quality Enhancement
The current RAG knowledge base uses Wikipedia API data, which includes some irrelevant information. Future improvements could include:

1. **Better data filtering** - Implement relevance scoring to filter out low-quality Wikipedia articles
2. **Additional sources** - Integrate Genius API for lyrics and song explanations
3. **Hybrid RAG approach** - Combine pre-built knowledge base with real-time API calls for new/trending songs (see: [Hybrid RAG Enhancement Documentation](./docs/hybrid_rag_approach.md))
4. **Manual curation** - Review and curate high-quality artist/track descriptions
5. **Structured data extraction** - Parse Wikipedia infoboxes for more precise information

### Model Improvements
- Fine-tune on larger dataset with more diverse music genres
- Experiment with larger base models (e.g., Mistral-7B)
- Multi-task training (recommendation + explanation + genre classification)

---

## 🚀 Why This Project?

- **End-to-end ML pipeline** - Data collection → Dataset creation → Fine-tuning → RAG → Inference
- **Practical use case** - Music recommendation is relatable and fun
- **French NLP** - Specialized for French language understanding
- **RAG implementation** - Demonstrates knowledge retrieval and augmented generation
- **Efficient training** - Shows how to fine-tune LLMs on consumer hardware

---

## 🎓 Learning Outcomes

This project demonstrates:
- API integration (Discogs, Wikipedia, ChatGPT)
- Dataset generation with LLMs
- Parameter-efficient fine-tuning (QLoRA)
- RAG system implementation
- Vector databases and similarity search
- Memory optimization for large-scale indexing
- End-to-end ML system design

---

*Note: This README was co-written with an LLM for clarity and structure.*