
# 🎙️ Vedanta Voice

**Vedanta Voice** is an interactive AI chatbot that allows users to ask questions related to Vedantic knowledge (e.g., Bhagavad Gita) and receive contextual, multilingual answers.  
It uses **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Llama 2** (via CTransformers) for retrieval-based question answering, integrated with **Streamlit** for the user interface.

---

## 🧩 Features

- 🧠 **Retrieval-based QA** using FAISS and LangChain  
- 💬 **Conversational chat interface** built with Streamlit  
- 🌍 **Automatic language detection and translation** via Google Translate  
- 📚 **PDF context retrieval** — answers sourced from uploaded PDFs  
- 🔗 **Source document linking** displayed with each response  
- 🗣️ (Optional) **Text-to-speech integration** (commented out, ready for use)

---

## ⚙️ Tech Stack

- **Python 3.9+**
- **Streamlit**
- **LangChain**
- **FAISS**
- **HuggingFace Sentence Transformers**
- **CTransformers (Llama 2)**
- **Googletrans**
- **Langdetect**
- **Streamlit Chat**

---

## 🏗️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd vedanta-voice
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal — typically
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📁 Project Structure

```
vedanta-voice/
├── vectorstore/
│   └── db_faiss/           # FAISS vector database
├── vedanta_voice.py        # Main Streamlit app
├── requirements.txt
└── README.md
```

---

## 🧭 How It Works

1. **FAISS Vector DB** stores embeddings of your PDFs.
2. **User asks a question** → language detected → translated to English.
3. **Retriever** searches relevant chunks from FAISS DB.
4. **Llama 2 model** generates a context-aware response.
5. **Response** is translated back to the user’s language (if needed).
6. **Sources** of the answer are displayed and linked.

---

## 🗣️ Optional Voice Features

Commented code is already included for:

* **Speech generation (WhisperSpeech / pyttsx3)**
* **Audio playback of generated responses**

You can enable these by installing the relevant libraries and uncommenting those sections.

---

## ⚠️ Notes

* Make sure your **FAISS database path** matches `DB_FAISS_PATH = 'vectorstore/db_faiss/'`
* The model `"TheBloke/Llama-2-7B-Chat-GGML"` must be downloaded or accessible locally.
* Google Translate API via `googletrans` may occasionally be rate-limited.

---

```
```
