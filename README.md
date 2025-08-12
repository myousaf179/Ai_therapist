# Ai_therapist
AI Therapist is an empathetic, secure, and easy-to-run Flask application designed for prototyping conversational mental‑health tools. It combines text-based emotion detection, disorder classification (from structured inputs), and neural text summarization (T5) with an optional therapist-style chatbot (which uses your own API key). This README delivers concise, step‑by‑step setup and usage instructions so you can run and evaluate the system locally.

---

## Repo structure 

```
Therapist-Ai/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ templates/
│  ├─ index.html
│  ├─ emotion_prediction.html
│  ├─ text_summarizer.html
│  ├─ disease_prediction.html
│  └─ chatbot.html
├─ static/
├─ models/
│  ├─ disorder_model.pkl
│  └─ emotion_model.pkl
├─ my_model_dir/        <- T5 summarizer files (download & extract, see below)
└─ README.md
```

**Important:** `my_model_dir/` is large and not included in the repository. Download it from Google Drive and extract to the repo root:

[https://drive.google.com/file/d/1Cncos-tcAtXAD8IgR4eKNxc1ASa9rICF/view?usp=sharing](https://drive.google.com/file/d/1Cncos-tcAtXAD8IgR4eKNxc1ASa9rICF/view?usp=sharing)

`my_model_dir` should contain tokenizer + model files such as `pytorch_model.bin`, `config.json`, and `sentencepiece.model` (or other files saved by `save_pretrained`).

---

## Quick setup

1. Clone the repo and enter folder:

```bash
git clone <repo-url>
cd Therapist-Ai
```

2. Create & activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. Copy `.env.example` to `.env` and edit it to add your API key for the chatbot. Example `.env`:

```
API_KEY=your_api_key_here
```

> The chatbot will use the API key from `.env`. Do **not** commit `.env` to Git.

4. Download `my_model_dir` from the Google Drive link above and extract into the project root so `my_model_dir/` exists.

5. Place your models inside `models/`:

- `emotion_model.pkl` — should be a dict: `{'model': clf, 'vectorizer': vec}`
- `disorder_model.pkl` — classifier that accepts the numeric features in the feature order used in `app.py`

6. Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not present or incomplete, install:

```
python-dotenv numpy flask google-genai transformers scikit-learn sentencepiece torch
```

> Note: `torch` may require a platform-specific install command. If `pip install torch` fails, follow the official PyTorch install instructions.

7. Run the app:

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## What each page does

- **Home** — overview and links.
- **Emotion** — paste text → returns predicted emotion + confidence.
- **Summarizer** — paste long text → T5 summary + emotion prediction on the summary.
- **Disease** — fill numeric fields (match `feature_names` in `app.py`) → predicted disorder code & label.
- **Chatbot** — interactive therapist-style chat. The chatbot will call the external API using the key you placed in `.env`.

---

## API endpoints (short)

- `POST /predict_emotion` — JSON `{"text":"..."}` → returns emotion + confidence
- `POST /text_summarizer` — form field `summary_input` → returns summary + emotion
- `POST /predict_disorder` — form POST with numeric inputs → returns disorder prediction (rendered)
- `POST /chat_with_therapist` — JSON `{"message":"...","history":[...]}` → returns chatbot reply
- `POST /process_chat_session` — JSON `{"user_messages":[...]}` → returns summary + emotion

Example curl for chat:

```bash
curl -X POST http://127.0.0.1:5000/chat_with_therapist \
 -H "Content-Type: application/json" \
 -d '{"message":"I feel anxious","history":["I lost sleep"]}'
```

**Important note:** All features (emotion detection, summarizer, disorder prediction, and processing chat sessions) work locally without any external API key. Only the therapist chatbot requires you to provide your own API key in the `.env` file.
