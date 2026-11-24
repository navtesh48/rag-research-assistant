"""Page-aware text chunking. Whitespace is normalized first because PDF text is messy."""
import re

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):                         # back up to a sentence end if one is near
            cut = text.rfind('. ', start + size // 2, end)
            if cut != -1: end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 40: out.append(piece)
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return out
