"""A stored salary answer must not override a range the posting states.

Real defect, 2026-09-10, job 5934 (Apex Tech, Peoria IL). The posting publishes
"Postdoctoral Research Associate: $62,354 - $65,000 / yr". The answer bank holds
a fixed industry-scientist default of "$135,000 to $155,000", which fires before
profile/config/AI and never sees the job description, so the kit asked for more
than double the top of the advertised band.

The bank cannot tell an academic postdoc from an industry Scientist role: the
question text is "What are your salary expectations?" in both cases. So the
posting itself has to break the tie.

Policy under test: when the description states a range and the stored answer
falls outside it, do NOT submit the stored answer. Hand it to the user with the
posted range as the starting point. Picking a number is the candidate's call.
"""
import pytest

from jobbot.apply_questions import (_posted_salary_bands, _posted_salary_range,
                                    _salary_answer_conflicts)


class TestExtractPostedRange:
    @pytest.mark.parametrize("text,expected", [
        ("Postdoctoral Research Associate: $62,354 - $65,000 / yr", (62354.0, 65000.0)),
        ("The base pay range for this role is $71,000.00 - $89,000.00.", (71000.0, 89000.0)),
        ("Salary range: $120,000 to $150,000 per year", (120000.0, 150000.0)),
        ("compensation of $95,000-$115,000 annually", (95000.0, 115000.0)),
    ])
    def test_finds_annual_ranges(self, text, expected):
        assert _posted_salary_range(text) == expected

    def test_returns_none_when_no_range(self):
        assert _posted_salary_range("Competitive salary and great benefits.") is None

    def test_ignores_a_placeholder_zero_range(self):
        """Metro Health job 5537 posted 'Salary Range: $0.00 - $0.00'."""
        assert _posted_salary_range("Salary Range: $0.00 - $0.00") is None

    def test_ignores_hourly_only(self):
        """An hourly figure must not be read as an annual band."""
        assert _posted_salary_range("Research Assistant: $16.00 - $22.00 / hr") is None

    def test_multiple_bands_yield_no_single_range(self):
        """A talent pool spans several levels, so no one band can be assumed.

        An earlier version of this returned the min-to-max span
        (62,354 to 138,000). That made the guard useless on exactly the posting
        that motivated it: a $135,000 ask sits inside that span because Sr.
        Research Scientist tops out at $138,000, even though the postdoc line
        advertises $62,354. Multi-band postings must fall through to the user.
        """
        text = ("Postdoctoral Research Associate: $62,354 - $65,000 / yr\n"
                "Research Scientist: $66,000 - $95,000 / yr\n"
                "Sr. Research Scientist: $80,000 - $138,000 / yr")
        assert _posted_salary_range(text) is None
        assert _posted_salary_bands(text) == [
            (62354.0, 65000.0), (66000.0, 95000.0), (80000.0, 138000.0)]


class TestConflictDetection:
    POSTED = (62354.0, 65000.0)

    def test_stored_answer_far_above_the_band_conflicts(self):
        assert _salary_answer_conflicts("$135,000 to $155,000", self.POSTED) is True

    def test_stored_answer_inside_the_band_does_not_conflict(self):
        assert _salary_answer_conflicts("$63,000", self.POSTED) is False

    def test_stored_answer_overlapping_the_band_does_not_conflict(self):
        assert _salary_answer_conflicts("$60,000 to $70,000", self.POSTED) is False

    def test_answer_with_no_numbers_never_conflicts(self):
        assert _salary_answer_conflicts("Negotiable", self.POSTED) is False

    def test_no_posted_range_never_conflicts(self):
        assert _salary_answer_conflicts("$135,000 to $155,000", None) is False

    def test_slightly_above_top_is_tolerated(self):
        """Asking a little over the top of the band is normal negotiation."""
        assert _salary_answer_conflicts("$68,000", self.POSTED) is False

    def test_far_below_the_band_also_conflicts(self):
        """Underselling by half is as wrong as overshooting."""
        assert _salary_answer_conflicts("$25,000", self.POSTED) is True


class TestIntegration:
    """The helpers being right is not enough; the answer flow must actually use them.

    The first version of this fix keyed on FormQuestion.kind == "salary_expectation",
    but classify() buckets salary questions as "screening" and
    "salary_expectation" is a screening-pattern key, not a kind. The helper
    tests passed while the real path still sent $135K to a $62K posting.
    """

    SAMPLE_DESC = ("Postdoctoral Research Associate: $62,354 - $65,000 / yr\n"
                   "Research Scientist: $66,000 - $95,000 / yr")

    def _ask(self, description, bank_answer="$135,000 to $155,000, open to discussion"):
        from unittest.mock import patch

        from jobbot.apply_questions import FormQuestion, answer_questions, classify
        q = FormQuestion(text="What are your salary expectations?", required=True)
        q.kind = classify(q)
        with patch("jobbot.apply_questions.load_answer_bank",
                   return_value=[{"match": "salary", "answer": bank_answer}]), \
             patch("jobbot.apply_questions.personal_info_bank_entries_safe",
                   return_value=[]):
            answer_questions([q], resume="r", job_title="Postdoc", company="Apex Tech",
                             job_description=description, use_ai=False)
        return q

    def test_conflicting_bank_answer_is_not_submitted(self):
        q = self._ask(self.SAMPLE_DESC)
        assert q.needs_user is True
        assert "135,000" not in (q.answer or ""), (
            f"the out-of-band answer was submitted anyway: {q.answer!r}")

    def test_the_posted_band_is_offered_as_the_starting_point(self):
        q = self._ask(self.SAMPLE_DESC)
        assert "62,354" in q.guess and "95,000" in q.guess, q.guess

    def test_no_posted_range_leaves_the_bank_answer_alone(self):
        q = self._ask("Competitive salary and excellent benefits.")
        assert q.needs_user is False
        assert "135,000" in q.answer

    def test_multi_band_posting_always_asks_even_for_an_in_band_answer(self):
        """The ambiguity is which LEVEL applies, not just the amount."""
        q = self._ask(self.SAMPLE_DESC, bank_answer="$64,000")
        assert q.needs_user is True
        assert "62,354" in q.guess

    def test_single_band_in_range_answer_is_submitted_normally(self):
        one_band = "The base pay range for this role is $60,000 - $70,000 per year."
        q = self._ask(one_band, bank_answer="$64,000")
        assert q.needs_user is False
        assert q.answer == "$64,000"

    def test_single_band_out_of_range_answer_is_held(self):
        one_band = "The base pay range for this role is $60,000 - $70,000 per year."
        q = self._ask(one_band)
        assert q.needs_user is True
        assert "135,000" not in (q.answer or "")
