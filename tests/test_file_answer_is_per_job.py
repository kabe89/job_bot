"""A file answer is a path to a per-job tailored document, so it must never be
reused from answer memory.

Regression for a real defect found on a live Workday walk (job 3311, Innotech):
the kit's Resume/CV question was filled with
`output/5118_globex-corp_ai-product-manager_resume.pdf` -- another
company's resume -- because `_STATIC_KINDS` contained "file", so the path had
been remembered as a reusable static answer and won over the resume that
build_package had just generated for this job.
"""
import json

from jobbot import answer_memory
from jobbot.apply_questions import FormQuestion, answer_questions


def test_a_file_question_is_never_a_static_memory_kind():
    q = FormQuestion(text="Resume/CV", qtype="file", kind="file", required=True)
    assert answer_memory.classify_kind(q) == "tailored", (
        "a per-job document path must not be classified reusable")


def test_remembered_resume_path_never_overrides_this_jobs_resume(tmp_path, monkeypatch):
    """Even with a poisoned memory file already on disk, the kit must use the
    resume built for THIS job."""
    mem_path = tmp_path / "answer_memory.json"
    mem_path.write_text(json.dumps([{
        "key": "resume cv",
        "kind": "static",                      # what the old classifier wrote
        "answer": "output/5118_globex-corp_ai-product-manager_resume.pdf",
        "source": "file",
        "last_used": "2026-07-30T18:10:14.861472",
    }]), encoding="utf-8")

    from jobbot import apply_questions
    monkeypatch.setattr(apply_questions.settings, "answer_memory_path", str(mem_path))

    this_job_resume = "output/3311_innotech_principal-engineer_resume.pdf"
    q = FormQuestion(text="Resume/CV", qtype="file", kind="file", required=True)
    answer_questions([q], resume="", job_title="Principal Engineer",
                     company="Innotech", job_description="",
                     resume_path=this_job_resume, use_ai=False)

    assert q.answer == this_job_resume, (
        f"kit used {q.answer!r} instead of this job's own resume")
    assert q.answer_source == "file"


def test_a_cover_letter_upload_also_ignores_remembered_paths(tmp_path, monkeypatch):
    mem_path = tmp_path / "answer_memory.json"
    mem_path.write_text(json.dumps([{
        "key": "cover letter",
        "kind": "static",
        "answer": "output/5118_globex-corp_ai-product-manager_cover.txt",
        "source": "file",
        "last_used": "2026-07-30T18:10:14.861472",
    }]), encoding="utf-8")

    from jobbot import apply_questions
    monkeypatch.setattr(apply_questions.settings, "answer_memory_path", str(mem_path))

    this_cover = "output/3311_innotech_principal-engineer_cover.txt"
    q = FormQuestion(text="Cover Letter", qtype="file", kind="file")
    answer_questions([q], resume="", job_title="Principal Engineer",
                     company="Innotech", job_description="",
                     resume_path="r.pdf", cover_letter_path=this_cover,
                     use_ai=False)

    assert q.answer == this_cover


def test_static_answer_for_refuses_a_poisoned_file_entry(tmp_path):
    """One helper owns both guards, so a call site cannot forget one.

    `apply_runner`'s pre-fill and `apply_questions._memory_lookup` each used to
    check only `hit.kind == "static"` -- the kind recorded on disk. That trusts
    a classification made by an older classifier. The helper also re-derives the
    kind from the question in hand.
    """
    mem_path = tmp_path / "m.json"
    mem_path.write_text(json.dumps([{
        "key": "resume cv", "kind": "static", "source": "file",
        "answer": "output/5118_globex-corp_ai-product-manager_resume.pdf",
    }]), encoding="utf-8")

    mem = answer_memory.AnswerMemory(str(mem_path))
    file_q = FormQuestion(text="Resume/CV", qtype="file", kind="file")
    assert mem.static_answer_for(file_q) is None

    # A genuine static answer still comes back.
    mem2_path = tmp_path / "m2.json"
    mem2_path.write_text(json.dumps([{
        "key": "first name", "kind": "static", "source": "profile",
        "answer": "Jane",
    }]), encoding="utf-8")
    mem2 = answer_memory.AnswerMemory(str(mem2_path))
    name_q = FormQuestion(text="First Name", kind="identity")
    assert mem2.static_answer_for(name_q) == "Jane"
