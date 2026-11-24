from rag.answer import validate, Draft, Claim

def test_validation_drops_unshown_citations():
    d = Draft(answerable=True, claims=[Claim(text='a', sources=[1, 99]), Claim(text='b', sources=[98])])
    out = validate(d, {1, 2})
    assert len(out) == 1 and out[0].sources == [1]
