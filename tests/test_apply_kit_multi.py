"""`jobbot apply-kit` accepts several JOB_IDs and keeps going when one fails."""
from unittest.mock import patch

from click.testing import CliRunner

from jobbot.cli import cli


class _Pkg:
    """Minimal stand-in for browser_apply.build_package's return value."""

    def __init__(self, job_id):
        self.job_id = job_id
        self.title = f"Role {job_id}"
        self.company = "Acme"
        self.blanks = []
        self.questions = []
        self.kit_markdown_path = f"output/{job_id}_apply_kit.md"
        self.package_json_path = f"output/{job_id}_application_package.json"
        self.resume_path = ""
        self.cover_letter_path = ""
        self.url = ""


def test_single_job_id_still_works():
    with patch("jobbot.browser_apply.build_package", side_effect=lambda j, **k: _Pkg(j)) as bp, \
         patch("jobbot.cli._print_apply_kit"):
        result = CliRunner().invoke(cli, ["apply-kit", "42", "--no-interactive"])
    assert result.exit_code == 0, result.output
    assert [c.args[0] for c in bp.call_args_list] == [42]


def test_multiple_job_ids_build_each_in_order():
    with patch("jobbot.browser_apply.build_package", side_effect=lambda j, **k: _Pkg(j)) as bp, \
         patch("jobbot.cli._print_apply_kit"):
        result = CliRunner().invoke(
            cli, ["apply-kit", "3897", "5462", "5793", "--no-interactive"])
    assert result.exit_code == 0, result.output
    assert [c.args[0] for c in bp.call_args_list] == [3897, 5462, 5793]


def test_one_failure_does_not_abort_the_rest():
    """A dead posting or missing kit must not strand the jobs queued behind it."""
    def _flaky(job_id, **_kw):
        if job_id == 2:
            raise ValueError("no form found")
        return _Pkg(job_id)

    with patch("jobbot.browser_apply.build_package", side_effect=_flaky) as bp, \
         patch("jobbot.cli._print_apply_kit"):
        result = CliRunner().invoke(cli, ["apply-kit", "1", "2", "3", "--no-interactive"])

    # every job was attempted, not just the ones before the failure
    assert [c.args[0] for c in bp.call_args_list] == [1, 2, 3]
    # and the run reports a non-zero exit so a caller notices
    assert result.exit_code != 0, result.output
    assert "no form found" in result.output or "failed" in result.output.lower()


def test_summary_is_printed_for_a_batch():
    with patch("jobbot.browser_apply.build_package", side_effect=lambda j, **k: _Pkg(j)), \
         patch("jobbot.cli._print_apply_kit"):
        result = CliRunner().invoke(cli, ["apply-kit", "7", "8", "--no-interactive"])
    assert result.exit_code == 0, result.output
    assert "2" in result.output  # "2 built" / "2 of 2" style summary


def test_no_job_ids_is_an_error():
    result = CliRunner().invoke(cli, ["apply-kit", "--no-interactive"])
    assert result.exit_code != 0


def test_gap_fill_runs_per_job_only_when_interactive():
    """--no-interactive must never trigger the prompt loop, even with blanks."""
    def _with_blanks(job_id, **_kw):
        p = _Pkg(job_id)
        p.blanks = ["Why this company?"]
        return p

    with patch("jobbot.browser_apply.build_package", side_effect=_with_blanks), \
         patch("jobbot.cli._print_apply_kit"), \
         patch("jobbot.cli._run_gap_fill") as gap:
        result = CliRunner().invoke(cli, ["apply-kit", "5", "6", "--no-interactive"])
    assert result.exit_code == 0, result.output
    gap.assert_not_called()
