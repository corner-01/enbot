# EnBot V9 Ollama deployment

V9 is trained from the original Qwen2.5-7B-Instruct base with the corrected V9 dataset. It replaces the previous `enbot-optuna:7b` runtime model.

## Assets

- Q4_K_M GGUF: `D:\enbot-finetune-artifacts\deployment\ollama-v9\enbot-v9-q4_k_m.gguf`
- Ollama model: `enbot-v9:7b`
- Model definition: `D:\telebot\deployment\ollama-v9\Modelfile`

Import the model with:

```powershell
ollama create enbot-v9:7b -f D:\telebot\deployment\ollama-v9\Modelfile
ollama list
```

EnBot selects this model with `OLLAMA_MODEL=enbot-v9:7b`.
