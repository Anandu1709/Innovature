"""Recursive character chunking with overlap."""

from typing import Iterable, List


DEFAULT_SEPARATORS = ["\n\n", "\n", " "]


def _merge_splits(splits: List[str], chunk_size: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    cur_len = 0
    for s in splits:
        if not s:
            continue
        add_len = len(s) if not buf else len(s) + 1  # account for join space
        if cur_len + add_len <= chunk_size:
            buf.append(s)
            cur_len += add_len
            continue
        if buf:
            chunks.append(" ".join(buf))
            buf = []
            cur_len = 0
        if len(s) <= chunk_size:
            buf.append(s)
            cur_len = len(s)
        else:
            for i in range(0, len(s), chunk_size):
                chunks.append(s[i : i + chunk_size])
    if buf:
        chunks.append(" ".join(buf))
    return chunks


def split_text_recursive(
    text: str,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    separators: Iterable[str] | None = None,
) -> List[str]:
    """Split preferring coarse separators (\n\n, \n, space); hard-wrap very long fragments."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text.strip():
        return []

    separators = list(separators or DEFAULT_SEPARATORS)
    splits: List[str] = []
    frontier: List[str] = [text]
    depth = 0

    while frontier and depth < len(separators):
        sep = separators[depth]
        next_frontier: List[str] = []
        for frag in frontier:
            if len(frag) <= chunk_size:
                next_frontier.append(frag)
                continue
            pieces = frag.split(sep) if sep else [frag]
            if len(pieces) == 1:
                next_frontier.append(pieces[0])
            else:
                next_frontier.extend(pieces)
        if next_frontier == frontier:
            depth += 1
            frontier = next_frontier
            continue
        frontier = next_frontier
        if all(len(x) <= chunk_size for x in frontier):
            splits = frontier
            break

    merged = _merge_splits(frontier, chunk_size)

    overlapped: List[str] = []
    for chunk in merged:
        if len(chunk) <= chunk_size:
            overlapped.append(chunk.strip())
            continue
        stride = chunk_size - chunk_overlap
        for i in range(0, len(chunk), stride):
            piece = chunk[i : i + chunk_size].strip()
            if piece:
                overlapped.append(piece)

    merged_small: List[str] = []
    for c in overlapped:
        if len(c) <= chunk_size:
            merged_small.append(c)
            continue
        stride = chunk_size - chunk_overlap
        for i in range(0, len(c), stride):
            merged_small.append(c[i : i + chunk_size])

    dedup = [x for x in merged_small if x]
    return dedup
