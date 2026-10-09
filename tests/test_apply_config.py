# tests/test_apply_config.py
from jobbot.config import Settings, settings


def test_apply_settings_have_defaults():
    assert settings.playwright_headed is True
    assert settings.evidence_dir == "output/evidence"
    # Asserted against the DECLARED default rather than the live singleton:
    # conftest deliberately redirects answer_memory_path (and answer_bank_path)
    # to a temp dir so a test can never write the user's real, git-untracked
    # answer stores. Reading the live value here would just re-assert that
    # isolation is on, which is not what this test is about.
    assert (Settings.model_fields["answer_memory_path"].default
            == "data/answer_memory.json")
    assert (Settings.model_fields["answer_bank_path"].default
            == "data/answer_bank.json")
