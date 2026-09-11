import math
import os
from typing import Any, List, Optional, Dict

from datasets import Dataset

# 本项目只允许本地评估：关闭 RAGAS 匿名遥测。
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# 动态检查依赖，避免底层包隐式导入报错
try:
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
        AnswerSimilarity,
    )
    from ragas.run_config import RunConfig
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_rag_response(
    query: str,
    retrieved_contexts: List[str],
    answer: str,
    ground_truth: Optional[str] = None,
    llm: Optional[Any] = None,
    embeddings: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    使用 RAGAS 评估检索增强生成系统的性能

    Args:
        query: 用户查询
        retrieved_contexts: 检索到的上下文列表
        answer: 系统生成的回答
        ground_truth: 参考答案（可选）

    Returns:
        dict: 评估结果
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS 库未正确安装或依赖缺失，无法执行评估。"}

    # 必须显式传入本地模型，禁止 RAGAS 回退到默认的付费 API。
    if llm is None:
        return {"error": "未配置本地 RAGAS LLM，已阻止回退到付费 API"}
    if embeddings is None:
        return {"error": "未配置本地 RAGAS Embeddings，已阻止回退到付费 API"}

    # 准备评估数据
    if not retrieved_contexts:
        return {"error": "本次评测没有拿到知识库检索上下文"}

    data = {
        "user_input": [query],
        "retrieved_contexts": [retrieved_contexts],
        "response": [answer],
    }
    # 每次请求使用独立指标实例，避免并发请求共享/覆盖 LLM 与 Embeddings 状态。
    # Ollama temperature=0 时多次生成的问题几乎完全相同，strictness=1 更快且更稳定。
    metrics = [Faithfulness(), AnswerRelevancy(strictness=1)]
    if ground_truth and ground_truth.strip():
        data["reference"] = [ground_truth]
        metrics.extend(
            [
                ContextPrecision(),
                ContextRecall(),
                AnswerCorrectness(),
                AnswerSimilarity(),
            ]
        )

    # 创建数据集
    dataset = Dataset.from_dict(data)

    # 定义评估指标

    # 执行评估
    try:
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(
                timeout=120,
                max_retries=1,
                max_wait=5,
                # 单个本地 Ollama 实例串行评分，避免结构化响应相互干扰。
                max_workers=1,
            ),
            show_progress=False,
            # 本地模型偶尔无法生成某个指标要求的 JSON；保留其他成功指标。
            raise_exceptions=False,
        )
        if not results.scores:
            return {"error": "RAGAS 没有返回任何评分"}

        normalized_scores = {}
        for metric_name, value in results.scores[0].items():
            try:
                numeric_value = float(value)
                normalized_scores[metric_name] = (
                    numeric_value if math.isfinite(numeric_value) else None
                )
            except (TypeError, ValueError):
                normalized_scores[metric_name] = value

        unavailable_metrics = [
            metric_name
            for metric_name, value in normalized_scores.items()
            if value is None
        ]
        if unavailable_metrics:
            normalized_scores["warning"] = (
                "以下指标未能由本地模型完成: " + ", ".join(unavailable_metrics)
            )
        return normalized_scores
    except Exception as e:
        return {"error": f"评估过程中发生错误: {str(e)}"}
