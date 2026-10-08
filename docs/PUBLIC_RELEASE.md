# Public repository policy

The public repository contains implementation code, configuration templates,
tests, deployment examples, and aggregate evaluation results.

The following assets must stay outside Git:

- `.env` files, API keys, Feishu credentials, tokens, and private endpoints;
- raw knowledge-base documents and proprietary product Q&A;
- training, validation, test, holdout, and manual-review records;
- per-sample prompts, generations, retrieved passages, and judge explanations;
- base models, embedding models, rerankers, LoRA adapters, and GGUF files;
- Milvus/Redis data, SQLite files, caches, logs, temporary files, and virtual environments.

## Before publishing

1. Create a new empty GitHub repository instead of pushing the CNB repository
   history.
2. Copy the cleaned working tree without `.git` and ignored files.
3. Run `git status --ignored` and a secret scan before the first commit.
4. Run both the main and vision test suites.
5. Confirm that publishing the source code is permitted by the project owner.
6. Add an open-source license only when ownership and licensing are confirmed.

The included `.env.example` files contain names and safe defaults only. Real
credentials must never be copied into them.
