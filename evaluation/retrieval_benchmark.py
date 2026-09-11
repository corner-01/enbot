import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_CASES = Path(__file__).with_name("retrieval_cases.json")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve_from_api(api_url: str, query: str) -> List[Dict[str, Any]]:
    payload = json.dumps(
        {"query": query, "evaluate": False, "debug": True},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("status") != "success":
        raise RuntimeError(body)
    data = body.get("data", {})
    if "retrieval_debug" not in data:
        raise RuntimeError("API 未返回 retrieval_debug；请先重启 FastAPI 服务加载最新代码")
    return data["retrieval_debug"]


def first_relevant_rank(case: Dict[str, Any], documents: List[Dict[str, Any]]) -> int:
    expected_terms = [term.lower() for term in case.get("expected_any", [])]
    expected_model = case.get("expected_model", "").upper()
    for rank, document in enumerate(documents, start=1):
        text = document.get("preview", "").lower()
        model = document.get("model", "").upper()
        term_matches = not expected_terms or any(term in text for term in expected_terms)
        model_matches = not expected_model or model == expected_model
        if term_matches and model_matches:
            return rank
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local hybrid retrieval through the running FastAPI service."
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    passed = 0
    positive_count = 0
    reciprocal_rank_sum = 0.0

    for case in cases:
        try:
            documents = retrieve_from_api(args.api, case["query"])
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"[ERROR] {case['name']}: {exc}")
            return 2

        if case.get("expect_empty"):
            case_passed = not documents
            detail = f"returned={len(documents)}, expected=0"
        else:
            positive_count += 1
            rank = first_relevant_rank(case, documents)
            case_passed = rank > 0
            reciprocal_rank_sum += 1.0 / rank if rank else 0.0
            detail = f"relevant_rank={rank or 'not found'}, returned={len(documents)}"

        passed += int(case_passed)
        status = "PASS" if case_passed else "FAIL"
        print(f"[{status}] {case['name']}: {detail}")
        for document in documents:
            print(
                "  "
                f"#{document.get('rank')} model={document.get('model') or '-'} "
                f"reranker={document.get('reranker_score')} "
                f"source={document.get('source') or '-'}"
            )

    recall_at_k = passed / len(cases) if cases else 0.0
    mrr = reciprocal_rank_sum / positive_count if positive_count else 0.0
    print(f"\nSummary: passed={passed}/{len(cases)}, pass_rate={recall_at_k:.4f}, MRR={mrr:.4f}")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
