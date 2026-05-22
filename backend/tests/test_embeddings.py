from app.services.embeddings import hash_embedding


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    first = hash_embedding("机房运维流程", 32)
    second = hash_embedding("机房运维流程", 32)
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-6

