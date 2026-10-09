"""The cover letter must be written under the same truthfulness contract as the resume.

`generate_cover_letter` was the ONLY document-producing call in ollama_client
with no system contract, and the highest temperature of any of them:

    return _generate(prompt, temperature=0.7)      # no system=

Every sibling passes `system=_g.SYSTEM_RESUME_TAILOR` at 0.2-0.45. So the
2,907-char truthfulness contract -- "the ONLY source of truth", "you must not
add it", the no-hedging rules -- never reached the cover letter, while the job
description sat in its context at nearly 4x the creativity of the resume.

It produced exactly what that setup invites. From the real Acme Corp 3968 run,
attached to the Office of Sustainability role:

    "This history of maintaining technical debt-free codebases while supporting
     cross-functional teams prepares me to improve existing repositories and
     refactor sequence engineering tools within your platform applications."

The source for that role says, in full: "Managed the data-analysis division for
campus data on energy, food waste, and water usage." The words codebase,
repository, refactor, technical debt, version control and git appear NOWHERE in
the candidate's entire source.

The Acme Corp JD asks for refactoring a multi-contributor codebase -- a known gap.
The model manufactured precisely the missing qualification out of a campus
sustainability job. That is the single most costly thing this pipeline can do:
it invents the exact claim a screener will probe.

Same class as the forked-contract bug in 1c83b9a, where claude_client carried
its own 401-char contract instead of the shared 2,907-char one. A document path
that skips the contract will fabricate, whichever provider it runs on.
"""
from unittest.mock import patch

import pytest

from jobbot import gemini_client as _g
from jobbot import ollama_client as oc

_RESUME = """Jane M. Doe
Senior Operations Manager - Office of Sustainability, Example State University
- Managed the data-analysis division for campus data on energy, food waste, and water usage.
"""
_JD = ("Scientist, mRNA Sequence Engineering. You will refactor a "
       "multi-contributor codebase and maintain existing repositories.")


def _call_args():
    """Capture the kwargs generate_cover_letter hands to _generate."""
    with patch.object(oc, "_generate", return_value="letter") as gen:
        oc.generate_cover_letter(_RESUME, "Scientist", "Acme Corp", _JD,
                                 "Jane Doe")
        assert gen.call_count == 1
        return gen.call_args


def test_the_cover_letter_runs_under_the_truthfulness_contract():
    """The whole bug in one assertion."""
    kwargs = _call_args().kwargs
    assert kwargs.get("system") == _g.SYSTEM_RESUME_TAILOR, (
        "the cover letter is generated with no truthfulness contract; this is "
        "how 'maintaining technical debt-free codebases' reached a real letter")


def test_the_cover_letter_is_not_the_most_creative_document_we_produce():
    """0.7 on the one document with no contract, vs 0.2-0.45 everywhere else."""
    kwargs = _call_args().kwargs
    assert kwargs.get("temperature", 0.7) <= 0.45


def test_the_contract_is_the_shared_one_not_a_local_copy():
    """1c83b9a: a forked, weaker contract is how this class of bug hides."""
    kwargs = _call_args().kwargs
    assert kwargs.get("system") is _g.SYSTEM_RESUME_TAILOR


def test_the_prompt_still_carries_the_job_and_resume():
    args, kwargs = _call_args()
    prompt = args[0]
    assert "Acme Corp" in prompt and "Office of Sustainability" in prompt


def test_the_prompt_names_the_experience_import_defect():
    """Naming the exact failure is what made the hedging rules stick on a small
    local model. The JD is the temptation: the model must not read a
    requirement and write it back as the candidate's history."""
    args, _ = _call_args()
    low = args[0].lower()
    assert "job description" in low
    assert "requirement" in low and "experience" in low


def test_the_rule_is_shared_across_providers_not_forked():
    """1c83b9a: claude_client had its own 401-char copy of the 2,907-char
    contract, so one provider silently tailored under a 7x weaker rule. The
    same gap existed here in BOTH ollama_client and gemini_client. One source."""
    assert _g.COVER_LETTER_NO_IMPORT_RULE in _call_args().args[0]


def test_gemini_cover_letter_also_runs_under_the_contract():
    """gemini_client had the identical bug: _generate(prompt, temperature=0.7)
    with no system=. Inert today (ai_provider=ollama) but one /model away."""
    with patch.object(_g, "_generate", return_value="letter") as gen:
        _g.generate_cover_letter(_RESUME, "Scientist", "Acme Corp", _JD, "Jane")
        kwargs = gen.call_args.kwargs
    assert kwargs.get("system") is _g.SYSTEM_RESUME_TAILOR
    assert kwargs.get("temperature", 0.7) <= 0.45


def test_claude_cover_letter_gets_the_contract_via_build_system():
    """claude_client is safe by a different route -- _build_system always
    injects SYSTEM_RESUME_TAILOR. Pin that so a refactor cannot quietly drop it."""
    from jobbot import claude_client as _c
    blocks = _c._build_system("resume text", include_profile=False)
    assert blocks[0]["text"].startswith(_c.SYSTEM_RESUME_TAILOR[:80])
