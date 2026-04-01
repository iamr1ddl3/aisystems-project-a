# Failure Taxonomy — Week 1 Baseline (Naive RAG)

5 lowest-scoring queries from the 55-entry eval, classified by failure type.

## Failure Types

| Type | Definition |
|------|-----------|
| **Retrieval miss** | Hit Rate = 0 — right doc never retrieved |
| **Wrong ranking** | Right doc retrieved but at position 4-5 (low MRR) |
| **Hallucination** | High Hit Rate, low Faithfulness — model went off-context |
| **Wrong interpretation** | High Faithfulness, low Correctness — right context, wrong answer |

---

## 1. q50 — "What business support response times does Acmera offer?"

- **Composite:** 0.80 | Hit: Yes | MRR: 1.00 | Faith: 5/5 | **Correct: 1/5**
- **Category:** business | **Difficulty:** medium
- **Failure type: Wrong interpretation**
- **Analysis:** The correct source (19_acmera_business.md) was retrieved at rank 1, and the model stayed faithful to context. But the retrieved *chunk* didn't contain the response time details (which are in a different chunk of the same doc). The model correctly said "context doesn't contain this info" — faithful but wrong. This is a **chunking problem**: the 500-char fixed chunks split the business support section away from the response time details.
- **Week 2 fix:** Larger/overlapping chunks, or semantic chunking to keep related sections together.

## 2. q41 — "What are the retention offers for Premium Gold members at risk of downgrading?"

- **Composite:** 0.78 | Hit: Yes | **MRR: 0.33** | Faith: 5/5 | Correct: 4/5
- **Category:** business | **Difficulty:** hard
- **Failure type: Wrong ranking**
- **Analysis:** Expected source (11_internal_pricing.md) ranked 3rd behind support FAQ and membership docs. The internal pricing doc is semantically distant from the query phrasing — "retention offers" doesn't match typical pricing vocabulary. The model still found the answer but only because it appeared in a lower-ranked chunk.
- **Week 2 fix:** Hybrid search (BM25 + dense) to catch keyword matches that semantic search misses.

## 3. q01 — "What is the standard return window for products?"

- **Composite:** 0.80 | Hit: Yes | **MRR: 0.20** | Faith: 5/5 | Correct: 5/5
- **Category:** returns | **Difficulty:** easy
- **Failure type: Wrong ranking**
- **Analysis:** Expected source (01_return_policy.md) ranked 5th — last position. The top 4 chunks came from premium membership, troubleshooting, and corporate gifting docs that mention return windows incidentally. The naive embedding-only retrieval favors documents that discuss returns in passing over the primary return policy document.
- **Week 2 fix:** Re-ranking with a cross-encoder to boost the most relevant doc. BM25 would also help since "return window" appears literally in the return policy.

## 4. q21 — "How does Acmera handle a report of a damaged product during delivery?"

- **Composite:** 0.81 | Hit: Yes | **MRR: 0.25** | Faith: 5/5 | Correct: 5/5
- **Category:** orders | **Difficulty:** easy
- **Failure type: Wrong ranking**
- **Analysis:** Expected source (06_support_faq.md) ranked 4th. The top 3 were shipping policy, return policy, and warranty — all tangentially related to damaged products. The FAQ entry with the direct answer was buried. Despite this, the model assembled a correct answer from the shipping policy chunk that also mentions the 48-hour window.
- **Week 2 fix:** Hybrid search + re-ranking. The FAQ contains the exact phrasing "damaged product" which BM25 would match.

## 5. q04 — "What is the spending threshold for Premium Gold?"

- **Composite:** 0.83 | Hit: Yes | **MRR: 0.50** | Faith: 5/5 | Correct: 4/5
- **Category:** membership | **Difficulty:** easy
- **Failure type: Wrong ranking + Wrong interpretation (mild)**
- **Analysis:** Expected source (02_premium_membership.md) ranked 2nd behind a support ticket (08_support_tickets.md) that discusses a customer approaching Gold threshold. Correctness is 4/5 because the generated answer included extra detail beyond the expected answer. The support ticket chunk "stole" rank 1 because it contains specific spending numbers.
- **Week 2 fix:** Re-ranking to prefer policy docs over support tickets for definitional questions.

---

## Summary of Failure Patterns

| Failure Type | Count | Queries |
|-------------|-------|---------|
| **Wrong ranking** | 4 | q01, q04, q21, q41 |
| **Wrong interpretation** | 1 | q50 |
| **Retrieval miss** | 0 | — |
| **Hallucination** | 0 | — |

**Key insight:** The naive pipeline has **zero retrieval misses and zero hallucinations**. All failures are ranking-related — the right document is always in top-5, but often not at rank 1. This is the classic weakness of dense-only retrieval with small fixed-size chunks.

**Week 2 priorities:**
1. Hybrid search (BM25 + dense) to catch keyword matches
2. Cross-encoder re-ranking to boost the most relevant chunk
3. Larger or overlapping chunks to prevent context fragmentation (q50)
