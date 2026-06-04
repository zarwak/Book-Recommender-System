---
title: Semantic Book Recommender
emoji: 📚
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "5.35.0"
app_file: app.py
pinned: false
---

# 📖 Semantic Book Recommender

An AI-powered Gradio app that gives book recommendations based on your query, preferred category, and emotional tone (e.g., Happy, Suspenseful). Built with LangChain, OpenAI, and Hugging Face Spaces.

## ✨ Features

- Search by concept, not just keywords
- Emotion-based filtering (joy, fear, sadness...)
- Auto-generated thumbnails and captions
- Fully interactive Gradio UI

## 🚀 How to Use

1. Enter a book idea (e.g. _"A story about courage and resilience"_)
2. Choose a genre or category
3. Optionally select an emotional tone
4. Click "Find recommendations"

Enjoy your next great read!


## 🛠️ Technical Stack

- **LLM & Embeddings**: OpenAI embeddings for semantic understanding
- **Vector Database**: Chroma for efficient similarity search
- **Framework**: LangChain for orchestration
- **UI**: Gradio for interactive web interface
- **ML Pipeline**: Sentiment analysis, text classification, and vector search

## 📊 Project Structure

- `app.py`: Main Gradio application with recommendation logic
- **Datasets**:
  - `books.csv`: Original book dataset
  - `books_cleaned.csv`: Cleaned book data
  - `books_with_categories.csv`: Books with genre/category information
  - `books_with_emotions.csv`: Books with emotional metadata
- `tagged_description.txt`: Tagged book descriptions for semantic search
- **Jupyter Notebooks**:
  - `sentiment-analysis.ipynb`: Emotional tone analysis pipeline
  - `text_classification.ipynb`: Text classification experiments
  - `vector-search.ipynb`: Semantic search implementation
  - `code.ipynb`: General ML experiments

## Getting Started

1. **Clone the repository:**
   ```powershell
   git clone <repo-url>
   cd ML
   ```
2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Run notebooks:**
   Open any of the `.ipynb` files in Jupyter Notebook or VS Code to explore the analyses and experiments.

## Requirements

Install the required Python packages using the provided `requirements.txt` file.

## License

This project is for educational and research purposes.
