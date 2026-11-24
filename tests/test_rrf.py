from rag.fusion import rrf

def test_rrf_prefers_items_in_both_lists():
    assert rrf([[1, 2, 3], [3, 1, 4]])[:2] == [1, 3]

def test_rrf_includes_items_from_either_list():
    assert set(rrf([[1], [2]])) == {1, 2}
