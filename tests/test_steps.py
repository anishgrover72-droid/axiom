from axiom.common.steps import normalize_text, render_with_sentinels, segment


def test_segments_numbered_steps():
    text = "Step 1: add the numbers.\nStep 2: divide by two.\nStep 3: round up."
    assert segment(text) == [
        "Step 1: add the numbers.",
        "Step 2: divide by two.",
        "Step 3: round up.",
    ]


def test_segments_blank_line_blocks():
    text = "First we set x = 2.\n\nThen y = x + 3.\n\nSo the answer is 5."
    assert len(segment(text)) == 3


def test_idempotent():
    text = "1. premise one here\n2. premise two here\n3. conclusion follows here"
    once = segment(text)
    twice = segment("\n\n".join(once))
    assert once == twice


def test_short_fragment_merged():
    # "OK." is below min_words and must fold into a neighbour, not survive alone.
    assert "OK." not in segment("OK.\n\nNow compute the total carefully here.")


def test_empty():
    assert segment("   \n\n  ") == []
    assert normalize_text("a\r\n\n\n\nb") == "a\n\nb"


def test_render_with_sentinels():
    rendered = render_with_sentinels(["step one", "step two"])
    assert rendered.count("<|step|>") == 2
