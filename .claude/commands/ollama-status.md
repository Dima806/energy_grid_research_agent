Check Ollama server status and confirm required models are present.

```bash
curl -s http://127.0.0.1:11434/ && echo "" && ollama list
```

Required models: `qwen2.5:1.5b` (LLM) and `nomic-embed-text` (embeddings). If missing, run `make setup`.
