from axiom.common.io import read_records, write_records
from axiom.data.schemas import HEAD_NAMES, Example, Step, StepLabel, Trace


def test_example_roundtrip(tmp_path):
    ex = Example(
        id="1",
        dataset="gsm8k",
        domain="math",
        question="q",
        gold_answer="#### 4",
        answer_type="numeric",
    )
    path = tmp_path / "e.jsonl"
    assert write_records(path, [ex]) == 1
    assert next(iter(read_records(path, Example))) == ex


def test_steplabel_masking_and_order():
    label = StepLabel(trace_id="t", step_idx=0, domain="math", logic=0.5, consistency=1.0)
    values = label.head_values()
    assert tuple(values) == HEAD_NAMES  # canonical order preserved
    assert values["logic"] == 0.5
    assert values["commonsense"] is None  # unlabeled -> masked downstream


def test_trace_round_trip(tmp_path):
    trace = Trace(
        trace_id="t:0",
        example_id="t",
        dataset="d",
        domain="math",
        model="m",
        question="q",
        steps=[Step(idx=0, text="a")],
    )
    path = tmp_path / "t.jsonl"
    write_records(path, [trace])
    assert next(iter(read_records(path, Trace))).steps[0].text == "a"
