from jobbot.browser_engine import BrowserEngine


class _Loc:
    def __init__(self, count=1, raises=False):
        self._count = count
        self._raises = raises
        self.uploaded = None

    @property
    def first(self):
        return self

    def count(self):
        return self._count

    def set_input_files(self, path):
        if self._raises:
            raise RuntimeError("no such label")
        self.uploaded = path


class _Page:
    def __init__(self, label_loc, file_loc):
        self._label_loc = label_loc
        self._file_loc = file_loc
        self.label_query = None

    def get_by_label(self, label, exact=False):
        self.label_query = label
        return self._label_loc

    def locator(self, selector):
        assert selector == "input[type=file]"
        return self._file_loc


def _engine(page):
    eng = BrowserEngine.__new__(BrowserEngine)  # bypass __init__/browser launch
    eng.page = page
    return eng


def test_upload_by_label_success():
    file_loc = _Loc()
    label_loc = _Loc()
    eng = _engine(_Page(label_loc, file_loc))
    assert eng.set_file_by_label("Resume", "/tmp/r.pdf") is True
    assert label_loc.uploaded == "/tmp/r.pdf"


def test_upload_falls_back_to_file_input_when_label_fails():
    file_loc = _Loc()
    label_loc = _Loc(raises=True)          # label path throws
    eng = _engine(_Page(label_loc, file_loc))
    assert eng.set_file_by_label("Attach resume", "/tmp/r.pdf") is True
    assert file_loc.uploaded == "/tmp/r.pdf"


def test_upload_returns_false_when_no_file_input():
    file_loc = _Loc(count=0)               # no file inputs on page
    label_loc = _Loc(raises=True)
    eng = _engine(_Page(label_loc, file_loc))
    assert eng.set_file_by_label("Resume", "/tmp/r.pdf") is False


from jobbot import browser_apply
from jobbot.apply_questions import FormQuestion


def test_ensure_resume_question_adds_when_missing():
    qs = [FormQuestion(text="Why this role?", qtype="textarea")]
    out = browser_apply._ensure_resume_question(qs)
    assert any(q.qtype == "file" and q.kind == "file" for q in out)


def test_ensure_resume_question_noop_when_present():
    qs = [FormQuestion(text="Resume/CV", qtype="file", kind="file")]
    out = browser_apply._ensure_resume_question(qs)
    assert sum(1 for q in out if q.qtype == "file") == 1
