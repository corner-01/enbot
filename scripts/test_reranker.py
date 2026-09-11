from pathlib import Path

from sentence_transformers import CrossEncoder

local_model_path = Path(__file__).resolve().parents[1] / "models" / "bge-reranker-v2-m3"

print("正在使用 CPU 加载本地重排模型")
reranker = CrossEncoder(
    str(local_model_path),
    device="cpu",
    local_files_only=True,
    max_length=512,
)
scores = reranker.predict(
    [
        ["HESS-HY-T-4~20K-M 电网数据显示不对", "电网数据无法显示的排查方法"],
        ["HESS-HY-T-4~20K-M 电网数据显示不对", "产品认证证书"],
    ],
    batch_size=2,
    show_progress_bar=False,
)
print("CPU 重排得分:", [float(score) for score in scores])
