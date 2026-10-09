"""A list answer from the model must not reach a form as a Python repr.

"""
from unittest.mock import patch

from jobbot.apply_questions import FormQuestion, _ai_answer


def _run(model_answer, question_text="Write 3 concise bullet points."):
    q = FormQuestion(text=question_text)
    q.kind = "essay"
    with patch("jobbot.ai_client.answer_application_questions",
               return_value={question_text: model_answer}), \
         patch("jobbot.apply_questions.apply_free_text_answer",
               side_effect=lambda qq, source, draft: setattr(qq, "answer", draft)):
        _ai_answer([q], "resume text", "Role", "Company", "jd")
    return q.answer or ""


def test_list_answer_is_not_a_python_repr():
    ans = _run(["First achievement.", "Second achievement.", "Third."])
    assert "['" not in ans, f"Python list repr leaked into the answer: {ans!r}"
    assert "', '" not in ans, f"Python list repr leaked into the answer: {ans!r}"


def test_list_answer_keeps_every_item():
    ans = _run(["First achievement.", "Second achievement.", "Third."])
    for item in ("First achievement.", "Second achievement.", "Third."):
        assert item in ans, f"lost {item!r} from {ans!r}"


def test_list_answer_separates_items_readably():
    ans = _run(["Alpha.", "Beta."])
    # items must not be run together into one blob
    assert "Alpha." in ans and "Beta." in ans
    between = ans[ans.index("Alpha.") + len("Alpha."):ans.index("Beta.")]
    assert between.strip("-*• \n\t") == "", f"odd separator {between!r}"
    assert "\n" in ans or "•" in ans or "- " in ans, (
        f"items should be separated by newlines or bullets: {ans!r}")


def test_plain_string_answer_is_untouched():
    ans = _run("A single sentence answer.")
    assert ans == "A single sentence answer."


def test_nested_and_non_string_items_are_coerced_safely():
    ans = _run(["Fine.", 42, None])
    assert "['" not in ans
    assert "Fine." in ans
    assert "42" in ans
    assert "None" not in ans, f"empty items should be dropped, got {ans!r}"


def test_dict_answer_does_not_leak_braces():
    ans = _run({"point1": "Alpha.", "point2": "Beta."})
    assert "{'" not in ans and "':" not in ans, f"dict repr leaked: {ans!r}"
    assert "Alpha." in ans and "Beta." in ans
