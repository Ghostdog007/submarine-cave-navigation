"""Train / validation / held-out test seed split, in disjoint ranges.

train:      [0, 500_000)            caves the agent trains on
validation: [500_000, 1_000_000)    checkpoint selection during training
test:       [1_000_000, ...)        final reported results only
"""

VAL_SEED_OFFSET = 500_000
TEST_SEED_OFFSET = 1_000_000


def train_seeds(n: int, start: int = 0) -> list[int]:
    if start + n > VAL_SEED_OFFSET:
        raise ValueError("train seeds would overlap the validation range")
    return list(range(start, start + n))


def val_seeds(n: int, start: int = 0) -> list[int]:
    if start + n > TEST_SEED_OFFSET - VAL_SEED_OFFSET:
        raise ValueError("validation seeds would overlap the test range")
    return list(range(VAL_SEED_OFFSET + start, VAL_SEED_OFFSET + start + n))


def test_seeds(n: int, start: int = 0) -> list[int]:
    return list(range(TEST_SEED_OFFSET + start, TEST_SEED_OFFSET + start + n))


def is_test_seed(seed: int) -> bool:
    return seed >= TEST_SEED_OFFSET


def seeds_for(split: str, n: int) -> list[int]:
    return {"train": train_seeds, "val": val_seeds, "test": test_seeds}[split](n)
