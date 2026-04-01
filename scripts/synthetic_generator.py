"""
Synthetic Question Generator — Session 1

Generates evaluation questions from corpus documents using GPT-4o-mini.
Each generated question includes: id, query, expected_answer, expected_source,
difficulty, category.

Usage:
  python scripts/synthetic_generator.py --doc 02_premium_membership.md
  python scripts/synthetic_generator.py --doc 02_premium_membership.md --count 5
  python scripts/synthetic_generator.py --doc 07_promotional_events.md --persona frustrated
  python scripts/synthetic_generator.py --all --count 3
"""
import os
import sys
import json
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

SCRIPT_DIR = os.path.dirname(__file__)
CORPUS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "corpus")

# =========================================================================
# PERSONA PROMPTS
# =========================================================================

PERSONAS = {
    "standard": """You are a QA dataset generator for an e-commerce customer support system.
Given a company document, generate {count} diverse question-answer pairs that a real customer
or support agent might ask. The questions should test whether a RAG system can retrieve and
use the information in this document.

Requirements:
- Mix difficulty levels: some easy (single fact lookup), some medium (combining 2 facts),
  some hard (requires reasoning across sections or edge cases)
- Each answer must be fully grounded in the document text — do not invent facts
- Questions should sound natural, like a real customer would ask them
- Cover different sections of the document, not just the beginning
- Include the specific numbers, dates, amounts, and conditions from the document""",

    "frustrated": """You are a QA dataset generator simulating FRUSTRATED customers contacting support.
These customers are angry, confused, or in a rush. They ask questions with:
- Vague or incomplete phrasing ("why can't I return this?!", "this is broken what do I do")
- Emotional language and run-on sentences
- Misspellings or informal language ("ur", "cuz", "wth")
- Implicit assumptions (they don't state their membership tier or order details)
- Multi-part complaints disguised as questions

Given a company document, generate {count} question-answer pairs from frustrated customers.
The ANSWERS should still be accurate and grounded in the document, but the QUESTIONS should
be messy and realistic — the kind that stress-test a RAG system's ability to understand intent.

Requirements:
- Questions must be answerable from this document, even if poorly phrased
- Answers must be factually correct and grounded in the document
- Include edge cases where the customer's situation is ambiguous
- Mix difficulty: easy (clear intent despite frustration), medium (implicit context needed),
  hard (multi-issue complaint requiring reasoning)""",

    "mismatch": """You are a QA dataset generator creating DELIBERATELY TRICKY questions that test
retrieval precision. These questions:
- Use vocabulary from one topic but actually need answers from a different section
- Ask about exceptions, edge cases, and policy overlaps
- Combine concepts from multiple sections of the document
- Use synonyms or paraphrases instead of the exact terms in the document
- Ask negative questions ("What is NOT covered?", "When can't I...?")
- Reference scenarios that span multiple policies (e.g., "premium member returning a sale item")

Given a company document, generate {count} question-answer pairs designed to trick naive
retrieval systems. The questions should be answerable from this document but require
careful reading to get right.

Requirements:
- Answers must be factually correct and grounded in the document
- Questions should use indirect or misleading phrasing
- Focus on policy boundaries, exceptions, and cross-section reasoning
- Difficulty should skew medium-hard (these are adversarial by design)""",
}

RESPONSE_FORMAT = """
Document name: {doc_name}

Document text:
{doc_text}

Respond with a JSON array only (no markdown fences). Each entry must have:
- "query": the question
- "expected_answer": the correct answer based solely on this document
- "difficulty": "easy", "medium", or "hard"
- "category": a short category label (e.g., "returns", "shipping", "membership", etc.)

Example format:
[
  {{
    "query": "What is the return window?",
    "expected_answer": "30 calendar days from delivery.",
    "difficulty": "easy",
    "category": "returns"
  }}
]"""


def load_document(doc_name):
    """Load a corpus document by filename."""
    path = os.path.join(CORPUS_DIR, doc_name)
    if not os.path.exists(path):
        print(f"Document not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def generate_questions(doc_name, count=5, persona="standard"):
    """
    Generate synthetic Q&A pairs from a corpus document.
    Uses GPT-4o-mini with a persona-specific prompt.
    Returns a list of question dicts with expected_source and persona set.
    """
    doc_text = load_document(doc_name)
    if doc_text is None:
        return []

    persona_prompt = PERSONAS[persona].format(count=count)
    response_section = RESPONSE_FORMAT.format(doc_name=doc_name, doc_text=doc_text)
    prompt = persona_prompt + "\n" + response_section

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=3000,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    questions = json.loads(raw)

    # Set expected_source and persona on each question
    for q in questions:
        q["expected_source"] = doc_name
        q["persona"] = persona

    return questions


def critique_questions(questions, doc_name):
    """
    Auto-critique generated questions using GPT-4o-mini.
    Rates each question on realism (1-5) and difficulty accuracy (1-5),
    and flags as keep/rewrite/drop.
    Returns (kept_questions, critique_results).
    """
    doc_text = load_document(doc_name)

    questions_text = "\n".join(
        f'{i+1}. Query: {q["query"]}\n   Answer: {q["expected_answer"]}\n   Difficulty: {q["difficulty"]}'
        for i, q in enumerate(questions)
    )

    prompt = f"""You are a QA dataset quality reviewer. Review each question-answer pair below
and rate them for inclusion in a golden evaluation dataset.

For each question, provide:
- realism (1-5): How likely is a real customer to ask this? 5 = very natural, 1 = robotic/contrived
- difficulty_accuracy (1-5): Does the labeled difficulty match the actual difficulty? 5 = perfect match, 1 = completely wrong
- verdict: "keep" (good quality), "rewrite" (has potential but needs work), or "drop" (too vague, duplicate, or unanswerable)
- reason: Brief explanation of your verdict

Source document: {doc_name}

Document text:
{doc_text}

Questions to review:
{questions_text}

Respond with a JSON array only (no markdown fences). One entry per question, in the same order:
[
  {{"index": 1, "realism": 4, "difficulty_accuracy": 5, "verdict": "keep", "reason": "Clear and natural question"}},
  ...
]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    critiques = json.loads(raw)

    # Print critique table
    print(f"\n{'#':<4} {'Realism':>8} {'DiffAcc':>8} {'Verdict':<8} {'Reason'}", file=sys.stderr)
    print("-" * 70, file=sys.stderr)

    kept = []
    for i, (q, c) in enumerate(zip(questions, critiques)):
        verdict = c.get("verdict", "drop")
        realism = c.get("realism", 0)
        diff_acc = c.get("difficulty_accuracy", 0)
        reason = c.get("reason", "")
        print(f"{i+1:<4} {realism:>8} {diff_acc:>8} {verdict:<8} {reason[:50]}", file=sys.stderr)
        q["critique"] = c
        if verdict != "drop":
            kept.append(q)

    total = len(questions)
    dropped = total - len(kept)
    print(f"\nCritique summary: {total} reviewed, {len(kept)} kept, {dropped} dropped ({dropped/total*100:.0f}% drop rate)", file=sys.stderr)

    return kept, critiques


def list_corpus_docs():
    """List all .md files in the corpus directory."""
    return sorted(f for f in os.listdir(CORPUS_DIR) if f.endswith(".md"))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Q&A pairs from corpus docs")
    parser.add_argument("--doc", type=str, help="Specific document filename (e.g., 02_premium_membership.md)")
    parser.add_argument("--all", action="store_true", help="Generate for all corpus documents")
    parser.add_argument("--count", type=int, default=5, help="Number of questions per document (default: 5)")
    parser.add_argument("--persona", type=str, default="standard",
                        choices=["standard", "frustrated", "mismatch"],
                        help="Question persona: standard, frustrated, or mismatch (default: standard)")
    parser.add_argument("--critique", action="store_true", help="Run auto-critique loop on generated questions")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: print to stdout)")
    args = parser.parse_args()

    if not args.doc and not args.all:
        print("Specify --doc <filename> or --all")
        print(f"\nAvailable documents:")
        for doc in list_corpus_docs():
            print(f"  {doc}")
        print(f"\nAvailable personas: {', '.join(PERSONAS.keys())}")
        sys.exit(1)

    docs = list_corpus_docs() if args.all else [args.doc]
    all_questions = []

    for doc in docs:
        print(f"Generating {args.count} questions from {doc} (persona={args.persona})...", file=sys.stderr)
        questions = generate_questions(doc, count=args.count, persona=args.persona)
        print(f"  -> {len(questions)} questions generated", file=sys.stderr)

        if args.critique:
            print(f"  Running critique on {len(questions)} questions...", file=sys.stderr)
            questions, _ = critique_questions(questions, doc)

        all_questions.extend(questions)

    # Assign sequential IDs with persona prefix
    prefix = args.persona[:3]
    for i, q in enumerate(all_questions, 1):
        q["id"] = f"syn_{prefix}_{i:03d}"

    output = json.dumps(all_questions, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n{len(all_questions)} questions saved to {args.output}", file=sys.stderr)
    else:
        sys.stdout.buffer.write(output.encode("utf-8"))
        sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
