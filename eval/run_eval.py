"""Usage: python eval/run_eval.py
Build eval/questions.json by hand from your own PDFs: ~35 answerable questions with the gold
filename and page, and ~15 questions the PDFs cannot answer."""
import json, sys
sys.path.insert(0, 'src')
from rag.db import init
from rag.search import retrieve
from rag.answer import ask

conn = init()
qs = json.load(open('eval/questions.json'))
ans = [q for q in qs if q['answerable']]
for mode in ('vector', 'hybrid', 'hybrid_rerank'):
    hit = sum(any(h['filename'] == q['gold']['file'] and h['page'] == q['gold']['page']
                  for h in retrieve(conn, q['q'], mode=mode)) for q in ans)
    print(f'{mode:14s} recall@6 = {hit/len(ans):.2f}')

correct_refusals = sum(not ask(conn, q['q'])['answerable'] for q in qs if not q['answerable'])
false_refusals = sum(not ask(conn, q['q'])['answerable'] for q in ans)
print('refused unanswerable:', correct_refusals, '/', len(qs) - len(ans))
print('wrongly refused answerable:', false_refusals, '/', len(ans))
