"""
Retrieval Evaluation Framework (Priority 8) & Confidence Scoring (Priority 9).

Performs:
  - Evaluation of retrieval precision across text queries.
  - Section hierarchy matching validation.
  - Image relevance and association checks.
  - TF-IDF-based semantic retrieval confidence scoring.
  - Simulates the thresholding logic for triggering the Clarification Agent.
"""

import os
import sys
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Set

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "all_chunks.jsonl"
QUERIES_FILE = PROJECT_ROOT / "tests" / "test_queries.json"


def safe_print(*args, **kwargs):
    """Prints safely to Windows console by converting unsupported characters."""
    encoding = sys.stdout.encoding or 'utf-8'
    processed = []
    for arg in args:
        if isinstance(arg, str):
            processed.append(arg.encode(encoding, errors='replace').decode(encoding))
        else:
            processed.append(arg)
    print(*processed, **kwargs)


def load_chunks() -> List[Dict]:
    """Load all chunks from the ingestion pipeline output."""
    chunks = []
    if not CHUNKS_FILE.exists():
        safe_print(f"[ERROR] Chunks file not found at: {CHUNKS_FILE}")
        return []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_queries() -> List[Dict]:
    """Load evaluation test queries."""
    if not QUERIES_FILE.exists():
        safe_print(f"[ERROR] Queries file not found at: {QUERIES_FILE}")
        return []
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    """Clean and tokenize text into lowercase word tokens."""
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


class SimpleSearchEngine:
    """A lightweight BM25-like search engine for evaluation on local CPU."""
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.doc_count = len(chunks)
        
        # Tokenize documents
        self.doc_tokens = [tokenize(c["text"] + " " + (c.get("section") or "") + " " + (c.get("subsection") or "")) for c in chunks]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(self.doc_count, 1)
        
        # Calculate term document frequencies for IDF
        self.df = {}
        for tokens in self.doc_tokens:
            seen = set(tokens)
            for t in seen:
                self.df[t] = self.df.get(t, 0) + 1
                
    def get_idf(self, term: str) -> float:
        """Calculate inverse document frequency."""
        df = self.df.get(term, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Perform text retrieval based on query similarity scoring."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
            
        scores = []
        k1 = 1.5
        b = 0.75
        
        for doc_idx, tokens in enumerate(self.doc_tokens):
            score = 0.0
            doc_len = self.doc_lens[doc_idx]
            
            # Simple TF-IDF score
            token_counts = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1
                
            for q_term in query_tokens:
                if q_term in token_counts:
                    tf = token_counts[q_term]
                    idf = self.get_idf(q_term)
                    # BM25-like length normalization
                    term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / self.avg_doc_len)))
                    score += term_score
                    
            if score > 0:
                scores.append((self.chunks[doc_idx], score))
                
        # Sort by highest score first
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def calculate_confidence(query: str, retrieved_chunk: Dict, score: float) -> float:
    """
    Priority 9 — Confidence Scoring.
    Returns a normalized confidence value [0.0, 1.0].
    """
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(retrieved_chunk["text"] + " " + (retrieved_chunk.get("section") or "")))
    
    if not q_tokens:
        return 0.0
        
    overlap = q_tokens.intersection(c_tokens)
    # Keyword overlap ratio
    ratio = len(overlap) / len(q_tokens)
    
    # Scale search score into confidence
    # Logarithmic compression
    score_factor = min(score / 15.0, 1.0)
    
    # Balanced confidence indicator
    confidence = (ratio * 0.6) + (score_factor * 0.4)
    return round(min(confidence, 1.0), 3)


def run_evaluation():
    safe_print("\n" + "=" * 70)
    safe_print("  RETRIEVAL EVALUATION & QUALITY FRAMEWORK")
    safe_print("=" * 70)
    
    chunks = load_chunks()
    queries = load_queries()
    
    if not chunks or not queries:
        return
        
    safe_print(f"Loaded {len(chunks)} chunks and {len(queries)} test queries.")
    safe_print("Initializing local search engine...")
    search_engine = SimpleSearchEngine(chunks)
    
    # Metric counts
    total_queries = len(queries)
    successful_section_matches_at_1 = 0
    successful_section_matches_at_5 = 0
    mrr_sum = 0.0
    visual_queries = 0
    visual_successes = 0
    
    safe_print("\n" + "-" * 70)
    safe_print("  RUNNING QUERY BENCHMARK")
    safe_print("-" * 70)
    
    for idx, q_item in enumerate(queries):
        query = q_item["query"]
        expected_sect = q_item["expected_section"]
        category = q_item["category"]
        
        # Check if query requests diagrams
        is_visual = any(kw in query.lower() for kw in ["pinout", "diagram", "schematic", "layout", "wire", "connection"])
        if is_visual:
            visual_queries += 1
            
        results = search_engine.search(query, top_k=5)
        
        safe_print(f"\n[{idx + 1}/{total_queries}] Query: '{query}'")
        safe_print(f"  Expected Section: '{expected_sect}'")
        
        if not results:
            safe_print("  [!!] RETRIEVAL FAILED - No matching chunks found")
            continue
            
        # Top-1 analysis
        top_chunk, top_score = results[0]
        top_conf = calculate_confidence(query, top_chunk, top_score)
        
        # Section mapping evaluation
        matched_rank = -1
        for r_idx, (chunk, score) in enumerate(results):
            c_sect = chunk.get("section") or ""
            c_sub = chunk.get("subsection") or ""
            # Case insensitive partial match
            if expected_sect.lower() in c_sect.lower() or expected_sect.lower() in c_sub.lower():
                matched_rank = r_idx + 1
                break
                
        # Update metrics
        if matched_rank == 1:
            successful_section_matches_at_1 += 1
            successful_section_matches_at_5 += 1
            mrr_sum += 1.0
        elif matched_rank > 1:
            successful_section_matches_at_5 += 1
            mrr_sum += 1.0 / matched_rank
            
        # Visual evaluation
        top_has_images = len(top_chunk.get("images", [])) > 0
        if is_visual and top_has_images:
            visual_successes += 1
            
        # Print retrieval results
        match_str = f"MATCHED at Rank {matched_rank}" if matched_rank > 0 else "MISMATCHED"
        visual_str = "IMAGES RETRIEVED" if top_has_images else "NO IMAGES"
        safe_print(f"  Top Match Chunk ID:  {top_chunk['chunk_id']}")
        safe_print(f"  Top Match Section:   {top_chunk.get('section')} -> {top_chunk.get('subsection')}")
        safe_print(f"  BM25 Score:          {top_score:.2f}")
        safe_print(f"  Confidence Score:    {top_conf:.3f} (Threshold: 0.400)")
        safe_print(f"  Evaluation:          {match_str} | {visual_str}")
        
        # Confidence action check (Priority 9)
        if top_conf < 0.40:
            safe_print("  [ACTION] Confidence is LOW -> Clarification Agent would be triggered!")
        else:
            safe_print("  [ACTION] Confidence is HIGH -> Answer Synthesis Agent would proceed.")
            
    # Calculate aggregate scores
    section_precision_at_1 = (successful_section_matches_at_1 / total_queries) * 100
    section_recall_at_5 = (successful_section_matches_at_5 / total_queries) * 100
    mrr = mrr_sum / total_queries
    visual_precision = (visual_successes / max(visual_queries, 1)) * 100
    
    safe_print("\n" + "=" * 70)
    safe_print("  RETRIEVAL PERFORMANCE DASHBOARD")
    safe_print("=" * 70)
    safe_print(f"  Precision@1 (Section):    {section_precision_at_1:.1f}%")
    safe_print(f"  Recall@5 (Section):       {section_recall_at_5:.1f}%")
    safe_print(f"  Mean Reciprocal Rank:     {mrr:.3f}")
    safe_print(f"  Visual Retrieval Precision: {visual_precision:.1f}% ({visual_successes}/{visual_queries} queries)")
    safe_print("=" * 70)
    
    # Assert high-quality benchmarks
    if mrr >= 0.70:
        safe_print("  [PASSED] Retrieval Quality exceeds target MRR of 0.70!")
    else:
        safe_print("  [FAILED] Retrieval Quality did not meet target MRR of 0.70.")
        
    if visual_precision >= 80.0:
        safe_print("  [PASSED] Visual retrieval and context-aware image bindings are highly accurate!")
    else:
        safe_print("  [WARN] Visual retrieval accuracy is below 80.0% — refine parser bindings.")
    safe_print("=" * 70 + "\n")


if __name__ == "__main__":
    run_evaluation()
