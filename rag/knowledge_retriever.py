import os
import threading
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from rag.hybrid_search import HybridSearchEngine

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_knowledge_retriever_cache = None


class LocalSentenceEmbeddings(Embeddings):
    def __init__(self, model_path):
        # 移除 BitsAndBytesConfig，因为 SentenceTransformer 不支持它
        # RAGAS 会读取 embeddings.model 并将其作为模型名称记录，
        # EmbeddingUsageEvent 要求该字段必须是字符串。
        self.model = str(model_path)
        self.device = os.getenv(
            "EMBEDDING_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self._encoder = SentenceTransformer(
            model_path,
            device=self.device,
        )
        self._encode_lock = threading.Lock()

    def embed_documents(self, texts):
        with self._encode_lock:
            if hasattr(self._encoder, "encode_document"):
                embeddings = self._encoder.encode_document(
                    texts,
                    normalize_embeddings=True,
                    batch_size=4,
                )
            else:
                embeddings = self._encoder.encode(
                    texts,
                    prompt_name="document",
                    normalize_embeddings=True,
                    batch_size=4,
                )
        return embeddings.tolist()

    def embed_query(self, text):
        with self._encode_lock:
            if hasattr(self._encoder, "encode_query"):
                embeddings = self._encoder.encode_query(
                    [text],
                    normalize_embeddings=True,
                )
            else:
                embeddings = self._encoder.encode(
                    [text],
                    prompt_name="query",
                    normalize_embeddings=True,
                )
        return embeddings[0].tolist()


class KnowledgeRetriever(BaseRetriever):
    """
    自定义检索器，封装基础检索器并添加额外的方法
    """
    base_retriever: BaseRetriever
    embeddings: Embeddings
    vectorstore: Any
    hybrid_engine: Optional[Any] = None

    def _get_relevant_documents(self, query: str, *, run_manager):
        """实现基类方法，委托给基础检索器"""
        if self.hybrid_engine is not None:
            return self.hybrid_engine.search(query)
        return self.base_retriever.invoke(query)


def get_knowledge_retriever() -> Optional[BaseRetriever]:
    """
    获取或创建基础检索器实例。
    采用两阶段切片法：先按文件结构提取独立语义块，再对超长文本进行递归切片。
    """
    global _knowledge_retriever_cache
    if _knowledge_retriever_cache is not None:
        return _knowledge_retriever_cache

    # 添加日志记录
    print("="*50)
    print("开始初始化知识库检索器...")
    print("="*50)

    # ==================== 1. 读取 .env 中的配置 ====================
    folder_path = os.getenv("KNOWLEDGE_FILE", str(PROJECT_ROOT / "knowledge_data"))
    persist_directory = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
    embedding_path = os.getenv(
        "EMBEDDING_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "Qwen3-8B-native"),
    )

    # ==================== 2. 初始化本地 HuggingFace Embedding 模型 ====================
    try:
        print("正在初始化本地 qwen3-embedding-8b-native 模型...")

        embeddings = LocalSentenceEmbeddings(model_path=embedding_path)
        print("[OK] 本地 qwen3-embedding-8b-native 模型加载成功！")
    except Exception as e:
        print(f"[ERROR] 加载本地 Embedding 模型失败: {e}")
        embeddings = None

    if embeddings is None:
        return None

    # ==================== 3. 加载或构建向量数据库 ====================
    if os.path.exists(persist_directory):
        print(f"[INFO] 检测到本地已有 Chroma 数据库({persist_directory})，直接加载，跳过重复导入与向量化！")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print(f"[INFO] 首次运行，开始读取知识库，路径来源：{folder_path}")

        # --- 阶段1：多格式文件提取，支持 TXT 和 XLSX ---
        raw_docs = []
        file_count = 0
        for root, _dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)

                # 处理 TXT 文件
                if file.endswith('.txt'):
                    file_count += 1
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        if text.strip():
                            raw_docs.append(Document(page_content=text.strip(), metadata={"source": file}))
                    except Exception as e:
                        print(f"[WARN] 文件 {file_path} 解析失败: {e}，跳过")

                # 处理 XLSX 文件
                elif file.endswith('.xlsx'):
                    file_count += 1
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                        # 将每一行数据拼接成语义完整的文本块
                        for _, row in df.iterrows():
                            # 过滤掉空值，将列名与值组合，例如："列名: 值"
                            row_data = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
                            text = "\n".join(row_data)
                            if text.strip() and len(text.strip()) >= 10:
                                raw_docs.append(Document(page_content=text.strip(), metadata={"source": file}))
                    except Exception as e:
                        print(f"[WARN] 文件 {file_path} 解析失败: {e}，跳过")

        print(f"[INFO] 一共读取了 {file_count} 个文件，提取出 {len(raw_docs)} 个独立语义块")

        if not raw_docs:
            print("[ERROR] 没有找到可用的TXT/XLSX数据！请检查 .env 中的 KNOWLEDGE_FILE 路径是否正确。")
            return None

        # --- 阶段2：递归长度切分，处理超长文本 ---
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            keep_separator=True,
            add_start_index=True
        )
        split_docs = splitter.split_documents(raw_docs)
        print(f"✂️ 经过递归优化切片后，最终生成 {len(split_docs)} 个知识片段")

        # ==================== 4. 显式文本向量化 ====================
        print(f"🧠 正在使用本地模型，对 {len(split_docs)} 个片段进行文本向量化（耗时操作，请耐心等待）...")
        try:
            # 使用批量处理提高效率
            batch_size = 32
            vectors = []
            for i in range(0, len(split_docs), batch_size):
                batch = split_docs[i:i + batch_size]
                batch_vectors = embeddings.embed_documents([doc.page_content for doc in batch])
                vectors.extend(batch_vectors)
                print(f"已处理 {min(i + batch_size, len(split_docs))}/{len(split_docs)} 个片段")
            print(f"[OK] 向量化完成！成功生成 {len(vectors)} 条向量数据。")
        except Exception as e:
            print(f"[ERROR] 向量化过程中发生错误: {e}")
            return None

        # ==================== 5. 存入 Chroma 本地向量数据库 ====================
        print(f"[INFO] 正在将文本和向量绑定，存入本地 Chroma 数据库...")
        try:
            vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            print(f"[OK] 已成功存入本地 Chroma 数据库，持久化到目录: {persist_directory}")
        except Exception as e:
            print(f"[ERROR] 存入 Chroma 数据库失败: {e}")
            return None

    # ==================== 6. 创建并缓存检索器 ====================
    try:
        # 第一阶段扩大向量召回，再由 BM25/RRF/CPU Reranker 精排。
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 30},
        )

        reranker_path = os.getenv(
            "RERANKER_MODEL_PATH",
            str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3"),
        )
        hybrid_engine = HybridSearchEngine(
            vectorstore=vectorstore,
            dense_retriever=base_retriever,
            reranker_path=reranker_path,
            dense_k=30,
            lexical_k=50,
            rerank_k=10,
            final_k=2,
        )

        # 创建自定义检索器，封装基础检索器和额外方法
        custom_retriever = KnowledgeRetriever(
            base_retriever=base_retriever,
            embeddings=embeddings,
            vectorstore=vectorstore,
            hybrid_engine=hybrid_engine,
        )

    except Exception as e:
        print(f"[WARN] 创建混合检索器时发生错误: {e}，降级使用常规相似度检索")
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
        # 为降级检索器创建自定义检索器
        custom_retriever = KnowledgeRetriever(
            base_retriever=base_retriever,
            embeddings=embeddings,
            vectorstore=vectorstore,
            hybrid_engine=None,
        )

    _knowledge_retriever_cache = custom_retriever
    return custom_retriever
