"""Cited answer generation. The model returns claims with chunk ids; code then rejects any id it
was not shown. If retrieval is weak, we refuse without calling the model at all."""
from pydantic import BaseModel
from rag.llm import structured
from rag.search import retrieve

REFUSE_BELOW = -3.0       # cross-encoder logit; tune with your eval set

class Claim(BaseModel):
    text: str
    sources: list[int]     # chunk ids

class Draft(BaseModel):
    answerable: bool
    claims: list[Claim]

SYSTEM = ('Answer ONLY from the numbered passages. Each claim must be one sentence and cite the ids of the '
          'passages that directly support it. If the passages do not answer the question, set answerable=false '
          'and return no claims. Never use outside knowledge. Quote numbers exactly as written.')

def validate(draft: Draft, shown: set[int]) -> list[Claim]:
    good = []
    for c in draft.claims:
        ids = [s for s in c.sources if s in shown]
        if ids: good.append(Claim(text=c.text, sources=ids))
    return good

def ask(conn, question: str) -> dict:
    hits = retrieve(conn, question)
    if not hits or hits[0]['score'] is None or hits[0]['score'] < REFUSE_BELOW:
        return {'answerable': False, 'claims': [], 'sources': {}, 'reason': 'low retrieval score'}
    ctx = '\n\n'.join(f"[{h['id']}] ({h['filename']} p.{h['page']}) {h['text']}" for h in hits)
    draft = structured(Draft, SYSTEM, f'Passages:\n{ctx}\n\nQuestion: {question}')
    claims = validate(draft, {h['id'] for h in hits}) if draft.answerable else []
    used = {s for c in claims for s in c.sources}
    return {'answerable': bool(claims), 'claims': [c.model_dump() for c in claims],
            'sources': {h['id']: {'file': h['filename'], 'page': h['page'], 'text': h['text']}
                        for h in hits if h['id'] in used}}
