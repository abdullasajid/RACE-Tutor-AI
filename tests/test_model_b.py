from backend.model_b import build_generated_options, generate_distractors, generate_hints


def test_generate_distractors_returns_distinct_non_answer_phrases():
    article = (
        "Maria visited the science museum with her class. "
        "The museum displayed ancient fossils, weather instruments, and space models."
    )

    distractors = generate_distractors(
        article=article,
        question="Where did Maria go?",
        correct_answer="science museum",
    )

    assert len(distractors) == 3
    assert len(set(distractors)) == 3
    assert "Science Museum" not in distractors


def test_generate_hints_are_graduated():
    article = "Ali practiced every day. His practice helped him win the contest."

    hints = generate_hints(
        article=article,
        question="Why did Ali win?",
        correct_answer="practice",
    )

    assert len(hints) == 3
    assert "Focus on" in hints[0]
    assert "[...]" in hints[1]


def test_build_generated_options_places_correct_answer():
    options = build_generated_options(
        article="The passage mentions rivers, mountains, forests, and deserts.",
        question="What landform is mentioned?",
        correct_answer="mountains",
        correct_label="C",
    )

    assert set(options) == {"A", "B", "C", "D"}
    assert options["C"] == "mountains"
