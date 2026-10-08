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
    expected_source = case.get("expected_source", "")
    expected_row = case.get("expected_row")
    acceptable_documents = case.get("acceptable_documents") or [
        {"source": expected_source, "row": expected_row}
    ]
    for rank, document in enumerate(documents, start=1):
        text = document.get("preview", "").lower()
        model = document.get("model", "").upper()
        term_matches = not expected_terms or any(term in text for term in expected_terms)
        model_matches = not expected_model or model == expected_model
        document_matches = any(
            (not expected.get("source") or document.get("source", "") == expected["source"])
            and (expected.get("row") is None or document.get("row") == expected["row"])
            for expected in acceptable_documents
        )
        if term_matches and model_matches and document_matches:
            return rank
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate local hybrid retrieval through the running FastAPI service."
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    passed = 0
    positive_count = 0
    negative_count = 0
    negative_correct = 0
    recall_at_1_count = 0
    recall_at_2_count = 0
    reciprocal_rank_sum = 0.0

    for case in cases:
        try:
            documents = retrieve_from_api(args.api, case["query"])
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"[ERROR] {case['name']}: {exc}")
            return 2

        if case.get("expect_empty"):
            negative_count += 1
            case_passed = not documents
            negative_correct += int(case_passed)
            detail = f"returned={len(documents)}, expected=0"
        else:
            positive_count += 1
            rank = first_relevant_rank(case, documents)
            case_passed = rank > 0
            recall_at_1_count += int(rank == 1)
            recall_at_2_count += int(0 < rank <= 2)
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
                f"source={document.get('source') or '-'} "
                f"row={document.get('row') if document.get('row') is not None else '-'}"
            )

    pass_rate = passed / len(cases) if cases else 0.0
    recall_at_1 = recall_at_1_count / positive_count if positive_count else 0.0
    recall_at_2 = recall_at_2_count / positive_count if positive_count else 0.0
    mrr = reciprocal_rank_sum / positive_count if positive_count else 0.0
    no_answer_accuracy = (
        negative_correct / negative_count if negative_count else 0.0
    )
    print(
        "\nSummary: "
        f"passed={passed}/{len(cases)}, pass_rate={pass_rate:.4f}, "
        f"Recall@1={recall_at_1:.4f}, Recall@2={recall_at_2:.4f}, "
        f"MRR={mrr:.4f}, no_answer_accuracy={no_answer_accuracy:.4f} "
        f"({negative_correct}/{negative_count})"
    )
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
