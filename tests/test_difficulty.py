from axiom.data.difficulty import STAGES, bucket, score_difficulty
from axiom.data.schemas import Example


def _mk(dataset: str, domain: str, gold: str = "x", answer_type: str = "numeric") -> Example:
    return Example(
        id="1",
        dataset=dataset,
        domain=domain,
        question="q",
        gold_answer=gold,
        answer_type=answer_type,
    )


def test_gsm8k_op_count_buckets():
    assert score_difficulty(_mk("gsm8k", "math", "no ops here #### 4")) == "easy"
    assert (
        score_difficulty(_mk("gsm8k", "math", "<<1+1=2>> <<2>> <<3>> <<4>> #### 4")) == "multihop"
    )


def test_arc_is_adversarial():
    assert score_difficulty(_mk("arc_challenge", "science", "A", "mcq")) == "adversarial"


def test_multihop_domain():
    assert score_difficulty(_mk("hotpotqa", "multihop", "x", "span")) == "multihop"


def test_bucket_has_all_stages():
    assert set(bucket([_mk("gsm8k", "math", "#### 4")])) == set(STAGES)
