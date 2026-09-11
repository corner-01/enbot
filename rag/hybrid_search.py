import heapq
import math
import os
import re
import threading
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document


MODEL_PATTERN = re.compile(r"hess[-a-z0-9~.]+", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://[^\s\])]+")


def extract_model(text: str) -> str:
    match = MODEL_PATTERN.search(text or "")
    # Punctuation immediately following a model name (for example the dash in
    # "HESS-HY-T-4~20K-M - certificate") is not part of the model itself.
    return match.group(0).upper().rstrip("-_.") if match else ""


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese/English technical text without extra packages."""
    normalized = (text or "").lower()
    # Treat "NO.1", "NO. 1" and "NO。1" as the same technical token.
    normalized = re.sub(r"\bno[.。]?\s*(\d+)", r"no.\1", normalized)
    tokens = re.findall(r"[a-z0-9]+(?:[-_./~][a-z0-9]+)*", normalized)
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
            if len(run) <= 8:
                tokens.append(run)
    return tokens


def qa_weighted_tokens(text: str) -> List[str]:
    """Give the question part of a Q&A document extra lexical weight."""
    all_tokens = tokenize(text)
    answer_match = re.search(r"(?:答案|answer)\s*[:：]", text, flags=re.IGNORECASE)
    if answer_match:
        all_tokens.extend(tokenize(text[: answer_match.start()]))
    return all_tokens


class BM25Index:
    def __init__(self, documents: Sequence[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter = Counter()

        for index, document in enumerate(self.documents):
            term_frequency = Counter(qa_weighted_tokens(document.page_content))
            length = sum(term_frequency.values())
            self.doc_lengths.append(length)
            for term, frequency in term_frequency.items():
                self.postings[term].append((index, frequency))
                document_frequency[term] += 1

        self.document_count = len(self.documents)
        self.average_length = (
            sum(self.doc_lengths) / self.document_count if self.document_count else 1.0
        )
        self.idf = {
            term: math.log(
                1.0 + (self.document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 20) -> List[Document]:
        if not self.documents:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term, query_frequency in Counter(tokenize(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for document_index, term_frequency in self.postings.get(term, []):
                document_length = self.doc_lengths[document_index]
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * document_length / self.average_length
                )
                scores[document_index] += (
                    idf
                    * term_frequency
                    * (self.k1 + 1.0)
                    / denominator
                    * query_frequency
                )

        best = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
        output = []
        for document_index, score in best:
            source = self.documents[document_index]
            metadata = dict(source.metadata)
            metadata["bm25_score"] = float(score)
            output.append(Document(page_content=source.page_content, metadata=metadata))
        return output


class LocalCPUReranker:
    """Lazy local CrossEncoder that never occupies CUDA memory."""

    def __init__(self, model_path: str, batch_size: int = 2, max_length: int = 512):
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._lock = threading.Lock()
        self.disabled_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.model_path and os.path.isdir(self.model_path))

    def _load(self):
        if self._model is not None or self.disabled_reason:
            return self._model
        if not self.available:
            self.disabled_reason = "本地 Reranker 模型目录不存在"
            return None

        with self._lock:
            if self._model is not None or self.disabled_reason:
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_path,
                    device="cpu",
                    local_files_only=True,
                    max_length=self.max_length,
                )
                print(f"[OK] CPU Reranker 加载成功: {self.model_path}")
            except Exception as exc:
                self.disabled_reason = str(exc)
                print(f"[WARN] CPU Reranker 加载失败，降级使用 RRF: {exc}")
        return self._model

    def rank(self, query: str, documents: Sequence[Document]) -> List[Document]:
        model = self._load()
        if model is None or not documents:
            return list(documents)

        pairs = [[query, self._comparison_text(document)] for document in documents]
        try:
            with self._lock:
                scores = model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            ranked = []
            for document, score in zip(documents, scores):
                metadata = dict(document.metadata)
                metadata["reranker_score"] = float(score)
                ranked.append(Document(page_content=document.page_content, metadata=metadata))
            return sorted(
                ranked,
                key=lambda document: document.metadata.get("reranker_score", float("-inf")),
                reverse=True,
            )
        except Exception as exc:
            print(f"[WARN] CPU Reranker 推理失败，降级使用 RRF: {exc}")
            return list(documents)

    @staticmethod
    def _comparison_text(document: Document) -> str:
        """For Q&A records, compare the user query with the stored question."""
        content = document.page_content or ""
        answer_match = re.search(
            r"(?:^|\n)\s*(?:答案|answer)\s*[:：]",
            content,
            flags=re.IGNORECASE,
        )
        if not answer_match:
            return content
        question = content[: answer_match.start()]
        question = re.sub(
            r"^\s*(?:问题|question)\s*[:：]\s*",
            "",
            question,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        return question or content


class HybridSearchEngine:
    def __init__(
        self,
        vectorstore: Any,
        dense_retriever: Any,
        reranker_path: Optional[str] = None,
        dense_k: int = 20,
        lexical_k: int = 20,
        rerank_k: int = 10,
        final_k: int = 3,
        cache_size: int = 128,
    ):
        self.vectorstore = vectorstore
        self.dense_retriever = dense_retriever
        self.dense_k = dense_k
        self.lexical_k = lexical_k
        self.rerank_k = rerank_k
        self.final_k = final_k
        self.cache_size = cache_size
        self._cache: OrderedDict[str, List[Document]] = OrderedDict()
        self._cache_lock = threading.Lock()

        payload = vectorstore.get(include=["documents", "metadatas"])
        documents = []
        for content, metadata in zip(
            payload.get("documents", []), payload.get("metadatas", [])
        ):
            if not content:
                continue
            enriched_metadata = dict(metadata or {})
            enriched_metadata.setdefault("model", extract_model(content))
            documents.append(Document(page_content=content, metadata=enriched_metadata))

        self.bm25 = BM25Index(documents)
        enabled = os.getenv("RERANKER_ENABLED", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        self.reranker = (
            LocalCPUReranker(reranker_path) if enabled and reranker_path else None
        )
        print(f"[OK] 混合检索索引已建立: {len(documents)} 条文档")

    @staticmethod
    def _document_key(document: Document) -> str:
        return document.page_content.strip()

    @staticmethod
    def _expanded_queries(query: str) -> List[str]:
        queries = [query]
        normalized = query.lower()
        model_name = extract_model(query)
        if any(token in normalized for token in ("编号", "no.", "no。", "序号")):
            queries.append(
                f"{model_name} LCD home screen grid data NO.1 Voltage "
                "NO.2 active power NO.3 Frequency NO.4 CT power "
                "NO.5 Current NO.6 apparent power NO.7 Mains charging current"
            )
        return queries

    @staticmethod
    def _is_numbered_grid_query(query: str) -> bool:
        """Detect questions asking for the meaning of numbered grid fields."""
        normalized = (query or "").lower()
        has_numbering = any(
            token in normalized for token in ("编号", "序号", "no.", "no。")
        )
        asks_for_meaning = any(
            token in normalized
            for token in ("是什么意思", "含义", "分别", "哪些", "meaning", "which fields")
        )
        mentions_grid = any(
            token in normalized for token in ("电网", "grid", "mains", "lcd")
        )
        return has_numbering or (mentions_grid and asks_for_meaning)

    @staticmethod
    def _is_numbered_grid_document(document: Document) -> bool:
        """Require real numbered-field evidence instead of a loosely related grid Q&A."""
        normalized = re.sub(
            r"\bno[.。]?\s*(\d+)",
            r"no.\1",
            (document.page_content or "").lower(),
        )
        numbered_fields = set(re.findall(r"\bno\.(\d+)\b", normalized))
        mentions_grid = any(
            token in normalized for token in ("电网", "grid data", "mains data")
        )
        return mentions_grid and len(numbered_fields) >= 3

    @staticmethod
    def _is_asset_request(query: str) -> bool:
        normalized = (query or "").lower()
        return any(
            token in normalized
            for token in ("接线图", "示意图", "图片", "照片", "diagram", "image", "picture")
        )

    @staticmethod
    def _is_url_only_answer(document: Document) -> bool:
        content = document.page_content or ""
        answer_match = re.search(
            r"(?:^|\n)\s*(?:答案|answer)\s*[:：]\s*",
            content,
            flags=re.IGNORECASE,
        )
        answer = content[answer_match.end():] if answer_match else content
        answer_without_urls = URL_PATTERN.sub("", answer)
        answer_without_punctuation = re.sub(
            r"[\s\W_]+", "", answer_without_urls, flags=re.UNICODE
        )
        return bool(URL_PATTERN.search(answer)) and len(answer_without_punctuation) < 5

    @staticmethod
    def _copy_documents(documents: Sequence[Document]) -> List[Document]:
        return [
            Document(page_content=document.page_content, metadata=dict(document.metadata))
            for document in documents
        ]

    def _get_cached(self, query: str) -> Optional[List[Document]]:
        with self._cache_lock:
            cached = self._cache.get(query)
            if cached is None:
                return None
            self._cache.move_to_end(query)
            return self._copy_documents(cached)

    def _set_cached(self, query: str, documents: Sequence[Document]):
        with self._cache_lock:
            self._cache[query] = self._copy_documents(documents)
            self._cache.move_to_end(query)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def search(self, query: str) -> List[Document]:
        cached = self._get_cached(query)
        if cached is not None:
            return cached

        candidates: Dict[str, Document] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)
        for expanded_query in self._expanded_queries(query):
            dense_documents = self.dense_retriever.invoke(expanded_query)
            lexical_documents = self.bm25.search(expanded_query, top_k=self.lexical_k)
            for ranking in (dense_documents, lexical_documents):
                for rank, document in enumerate(ranking, start=1):
                    key = self._document_key(document)
                    if not key:
                        continue
                    candidates.setdefault(key, document)
                    rrf_scores[key] += 1.0 / (60.0 + rank)

        if not candidates:
            return []

        if not self._is_asset_request(query):
            informative_keys = [
                key
                for key, document in candidates.items()
                if not self._is_url_only_answer(document)
            ]
            if informative_keys:
                candidates = {key: candidates[key] for key in informative_keys}

        numbered_grid_query = self._is_numbered_grid_query(query)
        if numbered_grid_query:
            structured_keys = [
                key
                for key, document in candidates.items()
                if self._is_numbered_grid_document(document)
            ]
            if not structured_keys:
                return []
            candidates = {key: candidates[key] for key in structured_keys}

        query_model = extract_model(query)
        if query_model:
            exact_keys = [
                key
                for key, document in candidates.items()
                if query_model == extract_model(document.page_content)
            ]
            if exact_keys:
                candidates = {key: candidates[key] for key in exact_keys}
            elif numbered_grid_query:
                # Numbered screen layouts can differ between product families.
                # Never present another model's layout as an exact-model answer.
                return []

        fused_documents = []
        for key, document in candidates.items():
            metadata = dict(document.metadata)
            metadata.setdefault("model", extract_model(document.page_content))
            metadata["rrf_score"] = float(rrf_scores[key])
            fused_documents.append(
                Document(page_content=document.page_content, metadata=metadata)
            )
        fused_documents.sort(
            key=lambda document: document.metadata["rrf_score"], reverse=True
        )

        rerank_candidates = fused_documents[: self.rerank_k]
        if self.reranker is not None:
            rerank_candidates = self.reranker.rank(query, rerank_candidates)

        # CrossEncoder 为单标签相关性模型时通常返回非负分数。
        # 使用相对阈值剔除明显落后的候选，同时至少保留第一名。
        selected_candidates = rerank_candidates
        if rerank_candidates and "reranker_score" in rerank_candidates[0].metadata:
            best_score = rerank_candidates[0].metadata["reranker_score"]
            if best_score > 0:
                relative_threshold = max(best_score * 0.8, 0.1)
                selected_candidates = [
                    document
                    for document in rerank_candidates
                    if document.metadata.get("reranker_score", float("-inf"))
                    >= relative_threshold
                ]
        result = (selected_candidates or rerank_candidates[:1])[: self.final_k]
        self._set_cached(query, result)
        return self._copy_documents(result)
