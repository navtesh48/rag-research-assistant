"""usage: python -m rag.cli "your question" """
import sys
from rag.db import init
from rag.answer import ask

res = ask(init(), ' '.join(sys.argv[1:]))
if not res['answerable']:
    print('Not found in the documents.')
else:
    for c in res['claims']:
        refs = ', '.join(f"{res['sources'][s]['file']} p.{res['sources'][s]['page']}" for s in c['sources'])
        print(f"- {c['text']}  [{refs}]")
