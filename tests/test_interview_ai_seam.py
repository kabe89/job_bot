"""The coach's LLM seam must exist on every provider — ai_client dispatches by
attribute name, so a provider missing one AttributeErrors at call time."""
import inspect

import pytest

from jobbot import ai_client, claude_client, gemini_client, ollama_client

COACH_FNS = ("interview_question_bank", "score_answer", "interview_session_summary")


@pytest.mark.parametrize("fn_name", COACH_FNS)
@pytest.mark.parametrize("mod", [ollama_client, gemini_client, claude_client])
def test_every_provider_implements_the_coach_seam(mod, fn_name):
    assert callable(getattr(mod, fn_name, None)), \
        f"{mod.__name__} is missing {fn_name}"


@pytest.mark.parametrize("fn_name", COACH_FNS)
def test_provider_signatures_match_each_other(fn_name):
    sigs = {m.__name__: list(inspect.signature(getattr(m, fn_name)).parameters)
            for m in (ollama_client, gemini_client, claude_client)}
    assert len(set(map(tuple, sigs.values()))) == 1, f"signature drift: {sigs}"


@pytest.mark.parametrize("fn_name", COACH_FNS)
def test_ai_client_exports_the_dispatcher(fn_name):
    assert callable(getattr(ai_client, fn_name, None))


def test_normalize_bank_coerces_garbage_to_canonical_shape():
    out = gemini_client._normalize_bank({"behavioral": [{"q": "Tell me"}],
                                         "technical": "not a list",
                                         "junk": [{"q": "ignored"}]})
    assert set(out) == {"behavioral", "technical", "research", "contact"}
    assert out["behavioral"] == [{"q": "Tell me", "why": "", "answer_hook": "",
                                  "grounded_in": ""}]
    assert out["technical"] == [] and out["research"] == []


def test_normalize_bank_survives_a_non_dict():
    assert gemini_client._normalize_bank("nope") == {
        "behavioral": [], "technical": [], "research": [], "contact": []}


def test_normalize_rubric_clamps_and_defaults_to_neutral():
    out = gemini_client._normalize_rubric({"relevance": 9, "specificity": -2,
                                           "structure": "bad", "critique": " ok "})
    assert out["relevance"] == 5.0 and out["specificity"] == 0.0
    assert out["structure"] == 3.0 and out["fit"] == 3.0
    assert out["critique"] == "ok" and out["follow_up"] == ""
