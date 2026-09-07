cat << 'EOF' > README.md
# 🤖 AI-Powered Hybrid FAQ Chatbot

An intelligent, hybrid conversational chatbot built using **Python**, **Streamlit**, **NLP (NLTK & Scikit-Learn)**, and **Google Gemini AI**. The application prioritizes fast, local FAQ retrieval using TF-IDF vectorization and Cosine Similarity, falling back dynamically to Google's Gemini LLM for unanswered or open-ended user queries.

---

## ✨ Features

* **Hybrid Response Logic:** Local FAQ database search with dynamic Google Gemini AI fallback.
* **NLP Pipeline:** Lowercasing, punctuation removal, tokenization, stopword removal, and lemmatization via `NLTK`.
* **Efficient FAQ Retrieval:** Uses `TfidfVectorizer` and `cosine_similarity` for quick semantic matching.
* **Streamlit Interactive UI:** Stateful chat interface with live streaming response output.
* **Performance Caching:** Optimized with `@st.cache_data` and `@st.cache_resource` to minimize latency.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend / Framework:** Streamlit
* **Natural Language Processing:** NLTK, Scikit-Learn
* **Generative AI:** Google Generative AI SDK (`google-generativeai`)

---

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

## For run 
python -m streamlit run app.py