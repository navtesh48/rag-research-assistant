from rag.chunking import chunk_text

def test_chunks_overlap_and_size():
    t = ('Sentence number one is here. ' * 200)
    cs = chunk_text(t)
    assert all(len(c) <= 900 for c in cs) and len(cs) > 3

def test_short_fragments_are_dropped_and_whitespace_normalized():
    assert chunk_text('   \n\n  ') == []
    assert '\n' not in chunk_text('A fairly long sentence that is okay.\n\nAnd another one right here too.')[0]
