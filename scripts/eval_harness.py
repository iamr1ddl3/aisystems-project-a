"""
Evaluation Harness — Session 1 Starter

This is a SKELETON. During Session 1, we'll build each function
from scratch to create a complete eval pipeline.

Functions to implement:
  1. check_retrieval_hit() — is the expected source in the top-K results?
  2. calculate_mrr() — how high is the first relevant chunk ranked?
  3. judge_faithfulness() — is the answer grounded in the context? (LLM-as-judge)
  4. judge_correctness() — does the answer match the expected answer? (LLM-as-judge)
  5. run_eval() — orchestrate everything and produce a scorecard

Run: python scripts/eval_harness.py
"""
import os
import sys
import json
from collections import defaultdict
from openai import OpenAI
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
langfuse = Langfuse()

SCRIPT_DIR = os.path.dirname(__file__)


# =========================================================================
# GOLDEN DATASET
# =========================================================================
# TODO: We'll build this together in Session 1.
# Start with 5 hand-written question-answer-context triples.
# Format:
# {
#     "id": "q01",
#     "query": "What is the standard return window?",
#     "expected_answer": "30 calendar days from delivery date.",
#     "expected_source": "01_return_policy.md",
#     "difficulty": "easy",
#     "category": "returns"
# }
# =========================================================================


def load_golden_dataset():
    """Load the golden dataset from JSON file."""
    path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(path):
        print("No golden_dataset.json found. Create one first!")
        return []
    with open(path) as f:
        return json.loads(f.read())


# =========================================================================
# RETRIEVAL METRICS
# =========================================================================

def check_retrieval_hit(retrieved_chunks, expected_source):
    """
    Is the expected source document in the retrieved chunks?
    Returns True/False.
    """
    return any(c['doc_name'] == expected_source for c in retrieved_chunks)


def calculate_mrr(retrieved_chunks, expected_source):
    """
    Mean Reciprocal Rank — how high is the first relevant chunk?
    If relevant chunk is at position 1: MRR = 1.0
    If at position 3: MRR = 0.33
    If not found: MRR = 0.0
    """
    for i, chunk in enumerate(retrieved_chunks):
        if chunk['doc_name'] == expected_source:
            return 1.0 / (i + 1)
    return 0.0


# =========================================================================
# GENERATION METRICS (LLM-as-Judge)
# =========================================================================

def judge_faithfulness(query, answer, context):
    """
    Is the answer grounded in the retrieved context?
    Uses GPT-4o-mini as a judge with a structured rubric.
    Returns: {"score": 1-5, "reason": "explanation"}
    """
    prompt = f"""You are an evaluation judge. Rate how faithfully the answer is grounded in the provided context.

Rubric:
- Score 5: Every claim in the answer is explicitly supported by the context.
- Score 4: Most claims are supported; minor details may be inferred but not fabricated.
- Score 3: Some claims are supported, but others are not found in the context.
- Score 2: Many claims are unsupported or loosely connected to the context.
- Score 1: The answer contains fabricated information not in the context.

Query: {query}

Context:
{context}

Answer: {answer}

Respond with JSON only, no markdown fences:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


def judge_correctness(query, answer, expected_answer):
    """
    Does the answer match the expected answer?
    Uses GPT-4o-mini as a judge.
    Returns: {"score": 1-5, "reason": "explanation"}
    """
    prompt = f"""You are an evaluation judge. Rate how correctly the generated answer addresses the question compared to the expected answer.

Rubric:
- Score 5: The answer fully matches the expected answer in meaning and covers all key details.
- Score 4: The answer is mostly correct with minor omissions or extra (but accurate) detail.
- Score 3: The answer is partially correct but misses important details from the expected answer.
- Score 2: The answer addresses the topic but is largely incorrect or incomplete.
- Score 1: The answer is wrong, irrelevant, or contradicts the expected answer.

Query: {query}

Expected Answer: {expected_answer}

Generated Answer: {answer}

Respond with JSON only, no markdown fences:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(raw)


# =========================================================================
# LANGFUSE SCORE ATTACHMENT
# =========================================================================

def attach_langfuse_scores(trace_id, hit, mrr, faithfulness, correctness):
    """
    Attach eval scores to a LangFuse trace.
    Posts faithfulness, correctness, and retrieval_hit scores so they
    appear on each trace in the LangFuse dashboard.
    """
    langfuse.score(trace_id=trace_id, name="faithfulness", value=faithfulness)
    langfuse.score(trace_id=trace_id, name="correctness", value=correctness)
    langfuse.score(trace_id=trace_id, name="retrieval_hit", value=1.0 if hit else 0.0)
    langfuse.score(trace_id=trace_id, name="mrr", value=mrr)


# =========================================================================
# EVAL RUNNER
# =========================================================================

def run_eval():
    """
    Run the full evaluation:
    1. Load golden dataset
    2. Run each query through the RAG pipeline
    3. Score retrieval (hit rate, MRR)
    4. Score generation (faithfulness, correctness)
    5. Print scorecard
    6. Save results to eval_results.json
    """
    from rag import ask

    dataset = load_golden_dataset()
    if not dataset:
        return

    results = []
    hits, mrrs, faith_scores, correct_scores = [], [], [], []

    print(f"\nRunning eval on {len(dataset)} golden entries...\n")
    print("-" * 70)

    for q in dataset:
        print(f"  [{q['id']}] {q['query']}")

        result = ask(q['query'])
        chunks = result['retrieved_chunks']

        # Retrieval metrics
        hit = check_retrieval_hit(chunks, q['expected_source'])
        mrr = calculate_mrr(chunks, q['expected_source'])

        # Generation metrics (LLM-as-judge)
        faith = judge_faithfulness(q['query'], result['answer'], result['context'])
        correct = judge_correctness(q['query'], result['answer'], q['expected_answer'])

        hits.append(hit)
        mrrs.append(mrr)
        faith_scores.append(faith['score'])
        correct_scores.append(correct['score'])

        entry_result = {
            "id": q['id'],
            "query": q['query'],
            "expected_source": q['expected_source'],
            "expected_answer": q['expected_answer'],
            "generated_answer": result['answer'],
            "trace_id": result['trace_id'],
            "retrieval": {
                "hit": hit,
                "mrr": round(mrr, 4),
                "sources": [c['doc_name'] for c in chunks],
            },
            "faithfulness": faith,
            "correctness": correct,
        }
        results.append(entry_result)

        # Attach scores to LangFuse trace
        attach_langfuse_scores(result['trace_id'], hit, mrr, faith['score'], correct['score'])

        print(f"         hit={hit} | MRR={mrr:.2f} | faith={faith['score']}/5 | correct={correct['score']}/5")

    # Scorecard
    n = len(dataset)
    avg_hit = sum(hits) / n
    avg_mrr = sum(mrrs) / n
    avg_faith = sum(faith_scores) / n
    avg_correct = sum(correct_scores) / n

    print("\n" + "=" * 70)
    print("                         EVAL SCORECARD")
    print("=" * 70)
    print(f"  Golden entries:       {n}")
    print(f"  Hit Rate:             {avg_hit:.0%} ({sum(hits)}/{n})")
    print(f"  Mean Reciprocal Rank: {avg_mrr:.4f}")
    print(f"  Avg Faithfulness:     {avg_faith:.2f} / 5")
    print(f"  Avg Correctness:      {avg_correct:.2f} / 5")
    print("=" * 70)

    # Save results
    output_path = os.path.join(SCRIPT_DIR, "eval_results.json")
    output = {
        "summary": {
            "num_queries": n,
            "hit_rate": round(avg_hit, 4),
            "mean_mrr": round(avg_mrr, 4),
            "avg_faithfulness": round(avg_faith, 2),
            "avg_correctness": round(avg_correct, 2),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Flush all pending LangFuse events
    langfuse.flush()
    print("LangFuse scores attached to all traces.")

    return output


def run_stratified_eval(save_baseline=False):
    """
    Run full eval with per-category and per-difficulty breakdown.
    Identifies 3 worst-performing categories.
    Optionally saves baseline_scores.json.
    """
    output = run_eval()
    if not output:
        return

    results = output["results"]

    # Per-category aggregation
    cat_metrics = defaultdict(lambda: {"hits": [], "mrrs": [], "faith": [], "correct": []})
    diff_metrics = defaultdict(lambda: {"hits": [], "mrrs": [], "faith": [], "correct": []})

    dataset = load_golden_dataset()
    dataset_by_id = {q["id"]: q for q in dataset}

    for r in results:
        q = dataset_by_id[r["id"]]
        cat = q["category"]
        diff = q["difficulty"]

        cat_metrics[cat]["hits"].append(r["retrieval"]["hit"])
        cat_metrics[cat]["mrrs"].append(r["retrieval"]["mrr"])
        cat_metrics[cat]["faith"].append(r["faithfulness"]["score"])
        cat_metrics[cat]["correct"].append(r["correctness"]["score"])

        diff_metrics[diff]["hits"].append(r["retrieval"]["hit"])
        diff_metrics[diff]["mrrs"].append(r["retrieval"]["mrr"])
        diff_metrics[diff]["faith"].append(r["faithfulness"]["score"])
        diff_metrics[diff]["correct"].append(r["correctness"]["score"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    # Print per-category breakdown
    print("\n" + "=" * 70)
    print("                    PER-CATEGORY BREAKDOWN")
    print("=" * 70)
    print(f"  {'Category':<16} {'N':>3} {'Hit%':>6} {'MRR':>6} {'Faith':>7} {'Correct':>8}")
    print("  " + "-" * 50)

    cat_scores = {}
    for cat in sorted(cat_metrics.keys()):
        m = cat_metrics[cat]
        n = len(m["hits"])
        hit_rate = avg(m["hits"])
        mrr = avg(m["mrrs"])
        faith = avg(m["faith"])
        correct = avg(m["correct"])
        composite = (hit_rate + mrr + faith / 5 + correct / 5) / 4
        cat_scores[cat] = {
            "n": n,
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "avg_faithfulness": round(faith, 2),
            "avg_correctness": round(correct, 2),
            "composite": round(composite, 4),
        }
        print(f"  {cat:<16} {n:>3} {hit_rate:>5.0%} {mrr:>6.2f} {faith:>6.2f} {correct:>7.2f}")

    # Print per-difficulty breakdown
    print("\n" + "=" * 70)
    print("                   PER-DIFFICULTY BREAKDOWN")
    print("=" * 70)
    print(f"  {'Difficulty':<16} {'N':>3} {'Hit%':>6} {'MRR':>6} {'Faith':>7} {'Correct':>8}")
    print("  " + "-" * 50)

    diff_scores = {}
    for diff in ["easy", "medium", "hard"]:
        if diff not in diff_metrics:
            continue
        m = diff_metrics[diff]
        n = len(m["hits"])
        hit_rate = avg(m["hits"])
        mrr = avg(m["mrrs"])
        faith = avg(m["faith"])
        correct = avg(m["correct"])
        diff_scores[diff] = {
            "n": n,
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "avg_faithfulness": round(faith, 2),
            "avg_correctness": round(correct, 2),
        }
        print(f"  {diff:<16} {n:>3} {hit_rate:>5.0%} {mrr:>6.2f} {faith:>6.2f} {correct:>7.2f}")

    # Identify 3 worst categories by composite score
    worst_3 = sorted(cat_scores.items(), key=lambda x: x[1]["composite"])[:3]

    print("\n" + "=" * 70)
    print("                  3 WORST-PERFORMING CATEGORIES")
    print("=" * 70)
    for rank, (cat, scores) in enumerate(worst_3, 1):
        print(f"  {rank}. {cat} — composite={scores['composite']:.4f} "
              f"(hit={scores['hit_rate']:.0%}, mrr={scores['mrr']:.2f}, "
              f"faith={scores['avg_faithfulness']:.1f}, correct={scores['avg_correctness']:.1f})")
    print("=" * 70)

    # Save baseline
    if save_baseline:
        baseline = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "pipeline": "naive_rag",
            "aggregate": output["summary"],
            "per_category": cat_scores,
            "per_difficulty": diff_scores,
            "worst_categories": [cat for cat, _ in worst_3],
        }
        baseline_path = os.path.join(SCRIPT_DIR, "baseline_scores.json")
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"\nBaseline saved to {baseline_path}")


if __name__ == "__main__":
    save_baseline = "--save-baseline" in sys.argv
    run_stratified_eval(save_baseline=save_baseline)
