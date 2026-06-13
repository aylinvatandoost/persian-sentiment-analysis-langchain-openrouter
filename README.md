# Persian Sentiment Analysis with OpenRouter + LangChain

This project is a Streamlit app for Persian sentiment analysis. It uses a trained PyTorch sentiment model first, and when the text looks negative, it verifies the final decision with an LLM through OpenRouter and LangChain.

## Main change

The original project used Ollama locally. This version replaces Ollama with OpenRouter + LangChain.

## Files in this repository

```text
app.py
requirements.txt
.env.example
best_sentiment_model.pth
vocab_dict.json
.gitignore
README.md
```

The raw dataset is intentionally not included in this repository.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```env
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODEL=google/gemini-2.0-flash-001
```

Run the app:

```bash
streamlit run app.py
```

## Notes

- Do not upload your real `.env` file to GitHub.
- `best_sentiment_model.pth` is required for running the local PyTorch model.
- `vocab_dict.json` is included as a vocabulary backup.
- The dataset and training notebook are not required for running the final app.
