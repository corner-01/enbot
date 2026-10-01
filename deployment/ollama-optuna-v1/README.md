# EnBot Optuna Ollama deployment

The frozen Optuna QLoRA candidate is integrated into EnBot RAG as the local Ollama model `enbot-optuna:7b`.

## D-drive assets

- Frozen adapter: `D:\enbot-finetune-artifacts\optuna-qlora-v1\outputs\full_trial_006`
- Portable Q4_K_M GGUF: `D:\enbot-finetune-artifacts\deployment\ollama-optuna-v1\enbot-optuna-q4_k_m.gguf`
- Ollama model store: `D:\ollama-models`
- Model definition: `D:\telebot\deployment\ollama-optuna-v1\Modelfile`

Import or rebuild the Ollama manifest with:

```powershell
ollama create enbot-optuna:7b -f D:\telebot\deployment\ollama-optuna-v1\Modelfile
ollama list
```

EnBot selects the model with `OLLAMA_MODEL=enbot-optuna:7b`.

For this computer's 16 GB GPU, use `OLLAMA_NUM_GPU=0` and `EMBEDDING_DEVICE=cuda`: the Qwen3-8B embedding model uses the GPU and Ollama runs the 4-bit generation model on CPU. On a server with at least 32 GB VRAM, leave `OLLAMA_NUM_GPU` empty so Ollama can use the GPU automatically.

The merged Safetensors directory and F16 GGUF are conversion intermediates. After the Q4_K_M model is imported and verified, both can be deleted without affecting the frozen adapter, portable Q4 GGUF, or Ollama model.
