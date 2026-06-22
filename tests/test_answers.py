from axiom.common.answers import grade


def test_numeric_gsm8k_gold_and_pred():
    out = "Adding gives 18, then dividing by 3 we get the answer is 6."
    v = grade(out, "She had ... <<18/3=6>>6\n#### 6", "numeric")
    assert v.correct and v.pred == "6" and v.gold == "6"


def test_numeric_boxed_preferred():
    v = grade("blah 7 blah \\boxed{42}", "#### 42", "numeric")
    assert v.correct and v.pred == "42"


def test_numeric_handles_commas():
    v = grade("the answer is 1,024", "#### 1024", "numeric")
    assert v.correct


def test_numeric_wrong():
    assert not grade("the answer is 5", "#### 6", "numeric").correct


def test_mcq_letter():
    choices = {"label": ["A", "B", "C"], "text": ["cat", "dog", "fish"]}
    assert grade("I think the answer is B.", "B", "mcq", choices).correct


def test_mcq_by_choice_text():
    choices = {"label": ["A", "B", "C"], "text": ["cat", "dog", "fish"]}
    assert grade("clearly a dog", "B", "mcq", choices).correct


def test_mcq_digit_labels_arc():
    # ARC items can use numeric labels; both stated-label and choice-text must grade.
    choices = {"label": ["1", "2", "3", "4"], "text": ["red", "green", "blue", "yellow"]}
    assert grade("so the answer is 3", "3", "mcq", choices).correct
    assert grade("it is clearly blue", "3", "mcq", choices).correct
    assert not grade("the answer is 1", "3", "mcq", choices).correct


def test_span_exact_and_f1():
    v = grade("The Eiffel Tower", "Eiffel Tower", "span")
    assert v.f1 == 1.0  # articles/case/punct normalized away
    assert grade("Eiffel Tower is in Paris", "Eiffel Tower", "span").f1 < 1.0


def test_boolean_strategyqa():
    # StrategyQA gold is a bool stringified to True/False; yes/no/true/false all grade.
    assert grade("So the answer is yes.", "True", "boolean").correct
    assert grade("the answer is no", "False", "boolean").correct
    assert not grade("yes, definitely", "False", "boolean").correct
    assert grade("Therefore the answer is False.", "False", "boolean").correct
