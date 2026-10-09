"""Resume loading (md/txt/pdf/docx) and rendering (md -> pdf)."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

log = logging.getLogger("jobbot.resume")


@lru_cache(maxsize=1)
def load_template() -> str:
    """The SOTA style template (HTML comments stripped).

    Returns "" if no template is configured or the file is missing, so callers
    can inject it unconditionally and get a no-op when it's absent.
    """
    from .config import settings
    path = getattr(settings, "resume_template_path", "") or ""
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return ""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def refresh_template() -> None:
    load_template.cache_clear()


def template_block(kind: str = "resume") -> str:
    """Prompt-ready injection: the house style template under a clear header.

    Empty string when no template is configured (so prompts stay unchanged).
    `kind` only tweaks the lead-in wording (resume vs cover letter).
    """
    tpl = load_template()
    if not tpl:
        return ""
    what = "cover letter" if kind == "cover" else "resume"
    return (
        f"\n\nHOUSE STYLE TEMPLATE — follow this {what} structure, formatting, and "
        "professionalism standard exactly. It governs FORM only; all CONTENT must "
        "still come solely from the verified resume/profile, and the truthfulness "
        "contract overrides anything here if they ever conflict:\n\n" + tpl
    )


def load_resume(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Resume not found: {p}. Create it or set BASE_RESUME_PATH in .env")
    suffix = p.suffix.lower()
    if suffix in (".md", ".txt"):
        return p.read_text(encoding="utf-8")
    if suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            return "\n\n".join((pg.extract_text() or "") for pg in pdf.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(p)
        return "\n".join(par.text for par in doc.paragraphs)
    if suffix == ".doc":
        raise ValueError("Legacy .doc not supported — save as .docx in Word first.")
    raise ValueError(f"Unsupported resume format: {suffix}")


_NAVY = (0x1A, 0x3A, 0x6B)
_GRAY = (0x55, 0x55, 0x55)


def _set_bottom_border(paragraph, color: str = "1A3A6B", size: str = "6",
                       space: str = "2") -> None:
    """Add a horizontal rule under a paragraph (section-header underline)."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), space)
    bottom.set(qn("w:color"), color)
    borders.append(bottom)
    p_pr.append(borders)


def markdown_to_docx(md_text: str, out_path: str | Path, page_budget: str = "2-page") -> Path:
    """Render a Markdown resume to a polished, ATS-safe .docx.

    Professional house style with adaptive page budgeting:
    - 2-page (default): 0.5" margins, Calibri 10pt body, generous readable spacing.
    - 1-page: 0.4" margins, Calibri 9.5pt body, compact spacing with orphan-line elimination.
    Plain single-column text only (no tables / columns / graphics) so applicant-tracking systems parse it cleanly.
    """
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor, Inches

    is_one_page = str(page_budget).lower().startswith("1")

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = Inches(0.4 if is_one_page else 0.5)
    sec.left_margin = sec.right_margin = Inches(0.65 if is_one_page else 0.7)

    body_size = Pt(9.5 if is_one_page else 10)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = body_size
    normal.font.color.rgb = RGBColor(15, 23, 42)
    pf = normal.paragraph_format
    pf.space_after = Pt(1 if is_one_page else 2)
    pf.line_spacing = 1.03 if is_one_page else 1.08

    try:
        bullet_style = doc.styles["List Bullet"]
        bullet_style.font.name = "Calibri"
        bullet_style.font.size = body_size
        bullet_style.font.color.rgb = RGBColor(15, 23, 42)
    except Exception:
        pass

    lines = md_text.splitlines()
    # Header zone = lines after the H1 name, up to the first section/divider.
    # Those (title, contact) are centered under the name.
    seen_name = False
    in_header = False

    def _para(spacing_after=2, before=0, keep_next=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(spacing_after)
        p.paragraph_format.space_before = Pt(before)
        if keep_next:
            p.paragraph_format.keep_with_next = True
        return p

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            continue  # collapse blank lines; spacing is handled per-paragraph

        if stripped in ("---", "***", "___"):
            if in_header:           # divider closing the header zone
                in_header = False
            continue                # section underlines carry the structure

        if line.startswith("# "):
            p = _para(spacing_after=0 if is_one_page else 1, keep_next=True)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(line[2:].strip())
            r.bold = True
            r.font.size = Pt(16 if is_one_page else 19)
            r.font.color.rgb = RGBColor(*_NAVY)
            seen_name = True
            in_header = True
            continue

        if line.startswith("## ") or line.startswith("### "):
            text = line.lstrip("#").strip()
            p = _para(spacing_after=1 if is_one_page else 2,
                      before=4 if is_one_page else 6,
                      keep_next=True)
            r = p.add_run(text.upper())
            r.bold = True
            r.font.size = Pt(11 if is_one_page else 11.5)
            r.font.color.rgb = RGBColor(*_NAVY)
            _set_bottom_border(p, size="4" if is_one_page else "6")
            in_header = False
            continue

        if line.lstrip().startswith(("- ", "* ")):
            text = line.lstrip()[2:]
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(0.75 if is_one_page else 1.5)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.left_indent = Inches(0.2 if is_one_page else 0.22)
            _add_runs(p, text)
            in_header = False
            continue

        # Plain paragraph
        if in_header:
            p = _para(spacing_after=0, keep_next=True)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_runs(p, stripped)
            for run in p.runs:
                run.font.size = Pt(8.5 if is_one_page else 9.5)
                run.font.color.rgb = RGBColor(*_GRAY)
        else:
            if stripped.startswith("**") and ("|" in stripped or "—" in stripped or "–" in stripped):
                p = _para(spacing_after=0.5 if is_one_page else 1,
                          before=3 if is_one_page else 4.5,
                          keep_next=True)
            elif stripped.startswith("*") and stripped.endswith("*"):
                p = _para(spacing_after=1 if is_one_page else 2, before=0, keep_next=True)
            else:
                p = _para(spacing_after=1 if is_one_page else 2, before=0)
            _add_runs(p, stripped)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def _add_runs(paragraph, text: str) -> None:
    """Render **bold**, *italic*, ***bold italic***, and [links](url) markdown inline."""
    import re
    from docx.shared import RGBColor
    pattern = re.compile(
        r"(\[([^\]]+)\]\(([^)]+)\))|"
        r"(\*\*\*([^*]+)\*\*\*)|"
        r"(\*\*([^*]+)\*\*)|"
        r"(\*([^*]+)\*)"
    )
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        if m.group(1):  # [text](url)
            link_text = m.group(2)
            r = paragraph.add_run(link_text)
            r.font.color.rgb = RGBColor(*_NAVY)
            r.underline = True
        elif m.group(4):  # ***bold italic***
            r = paragraph.add_run(m.group(5))
            r.bold = True
            r.italic = True
        elif m.group(6):  # **bold**
            r = paragraph.add_run(m.group(7))
            r.bold = True
        elif m.group(8):  # *italic*
            r = paragraph.add_run(m.group(9))
            r.italic = True
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def markdown_to_pdf(md_text: str, out_path: str | Path, page_budget: str = "2-page") -> Path:
    """Render Markdown resume to a clean, submission-ready PDF via reportlab, matching DOCX styling."""
    import markdown as md_mod
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.lib.colors import HexColor
    from html.parser import HTMLParser

    is_one_page = str(page_budget).lower().startswith("1")
    html = md_mod.markdown(md_text, extensions=["extra", "sane_lists"])

    # Convert HTML to a sequence of (style_name, text)
    class Collector(HTMLParser):
        def __init__(self):
            super().__init__()
            self.items: list[tuple[str, str]] = []
            self.buf = ""
            self.tag_stack: list[str] = []

        def handle_starttag(self, tag, attrs):
            if tag in {"h1", "h2", "h3", "p", "li"}:
                self._flush()
                self.tag_stack.append(tag)
            elif tag == "br":
                self.buf += "<br/>"
            elif tag in {"strong", "b"}:
                self.buf += "<b>"
            elif tag in {"em", "i"}:
                self.buf += "<i>"

        def handle_endtag(self, tag):
            if tag in {"strong", "b"}:
                self.buf += "</b>"
            elif tag in {"em", "i"}:
                self.buf += "</i>"
            elif tag in {"h1", "h2", "h3", "p", "li"}:
                self._flush()
                if self.tag_stack and self.tag_stack[-1] == tag:
                    self.tag_stack.pop()

        def handle_data(self, data):
            self.buf += data.replace("\n", " ")

        def _flush(self):
            if self.buf.strip():
                tag = self.tag_stack[-1] if self.tag_stack else "p"
                self.items.append((tag, self.buf.strip()))
            self.buf = ""

    c = Collector()
    c.feed(html)
    c._flush()

    styles = getSampleStyleSheet()
    navy = HexColor("#1a3a6b")
    gray = HexColor("#475569")
    body_size = 9.5 if is_one_page else 10
    leading_size = 11.8 if is_one_page else 12.8

    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=body_size, leading=leading_size, alignment=TA_LEFT, spaceAfter=1.5 if is_one_page else 2.5)
    header_contact = ParagraphStyle("contact", parent=body, fontSize=8.5 if is_one_page else 9.5, leading=11, alignment=TA_CENTER, textColor=gray, spaceAfter=2)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16 if is_one_page else 19, leading=19 if is_one_page else 22, alignment=TA_CENTER, textColor=navy, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11 if is_one_page else 11.5, leading=13 if is_one_page else 14, spaceBefore=4 if is_one_page else 6, spaceAfter=1, textColor=navy, keepWithNext=True)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=10 if is_one_page else 10.5, leading=12 if is_one_page else 13, spaceBefore=3 if is_one_page else 4, spaceAfter=1, keepWithNext=True)
    li = ParagraphStyle("li", parent=body, leftIndent=12 if is_one_page else 14, bulletIndent=4, spaceAfter=0.8 if is_one_page else 1.5)

    style_map = {"h1": h1, "h2": h2, "h3": h3, "p": body, "li": li}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        leftMargin=(0.5 if is_one_page else 0.6) * inch,
        rightMargin=(0.5 if is_one_page else 0.6) * inch,
        topMargin=(0.4 if is_one_page else 0.5) * inch,
        bottomMargin=(0.4 if is_one_page else 0.5) * inch,
    )
    flow = []
    in_header_zone = False
    for tag, text in c.items:
        if tag == "h1":
            in_header_zone = True
            flow.append(Paragraph(text, h1))
            continue
        if tag in {"h2", "h3"}:
            in_header_zone = False
            flow.append(Paragraph(text, style_map.get(tag, h2)))
            if tag == "h2":
                flow.append(HRFlowable(width="100%", thickness=0.8, color=navy, spaceBefore=1, spaceAfter=3))
            continue
        if tag == "li":
            text = "• " + text
            flow.append(Paragraph(text, li))
            continue

        # Regular paragraph
        if in_header_zone:
            flow.append(Paragraph(text, header_contact))
        else:
            flow.append(Paragraph(text, body))

    doc.build(flow)
    return out_path


def get_canonical_letterhead(applicant_name: str = "") -> dict:
    from .config import settings
    name = applicant_name or getattr(settings, "applicant_name", "") or "Alex Taylor"
    name_display = getattr(settings, "applicant_name_display", "") or name
    loc = getattr(settings, "applicant_location", "") or "New York, NY 10001"
    addr1 = getattr(settings, "applicant_address", "") or "123 Innovation Way"
    city_st_zip = loc
    dept = getattr(settings, "applicant_title", "") or "Senior Scientist"
    inst = getattr(settings, "applicant_institution", "") or "Research Institute"
    email = getattr(settings, "applicant_email", "") or "alex.taylor@example.com"
    phone = getattr(settings, "applicant_phone", "") or "+1-555-0199"

    return {
        "name": name_display,
        "address_line1": addr1,
        "city_state_zip": city_st_zip,
        "department": dept,
        "institution": inst,
        "email": email,
        "phone": phone,
    }


def cover_letter_to_docx(cover_text: str, out_path: str | Path,
                        recipient_info: dict | None = None) -> Path:
    """Render a formal cover letter to a polished, professional Word document (.docx).

    Follows the executive & scientific cover letter standard:
    - 0.8" top/bottom margins, 0.85" left/right margins
    - Clean Calibri 11pt typography, 1.15 line spacing, dark slate text
    - Structured business letter sections:
      1. Sender contact block (Name [Bold 11.5pt], Address, City/State/ZIP)
      2. Date line (clean 10pt space below)
      3. Recipient block (Recipient/Title/Committee, Organization, Address)
      4. Salutation (Dear ...,)
      5. Body paragraphs (peer-to-peer scientific voice, 8pt space after)
      6. Complimentary close (Sincerely,)
      7. Formal signature block (Name [Bold], Affiliation/Lab, University/Institution, Contact info)
    """
    from datetime import datetime
    from docx import Document
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.8)
    sec.left_margin = Inches(0.85)
    sec.right_margin = Inches(0.85)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(15, 23, 42)  # slate-900
    pf = normal.paragraph_format
    pf.line_spacing = 1.15
    pf.space_after = Pt(8)

    letterhead = get_canonical_letterhead()
    lines = [l.strip() for l in cover_text.splitlines() if l.strip()]

    # Find salutation index
    salutation_idx = -1
    for i, line in enumerate(lines):
        if line.lower().startswith("dear ") or line.lower().startswith("to "):
            salutation_idx = i
            break

    # Find closing index ("Sincerely" or "Best regards" or "Respectfully")
    closing_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        low = lines[i].lower()
        if low.startswith("sincerely") or low.startswith("respectfully") or low.startswith("best regards"):
            closing_idx = i
            break

    # Handle header zone
    if salutation_idx > 0:
        header_lines = lines[:salutation_idx]
        date_idx = -1
        date_pattern = re.compile(
            r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}",
            re.I
        )
        for i, hl in enumerate(header_lines):
            if date_pattern.match(hl) or re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", hl):
                date_idx = i
                break

        if date_idx != -1:
            sender_lines = header_lines[:date_idx]
            date_line = header_lines[date_idx]
            recipient_lines = header_lines[date_idx + 1:]
        else:
            sender_lines = header_lines[:3]
            date_line = datetime.now().strftime("%B %d, %Y")
            recipient_lines = header_lines[3:]

        # Render Sender
        for j, sl in enumerate(sender_lines):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1.5)
            r = p.add_run(sl)
            if j == 0:
                r.bold = True
                r.font.size = Pt(11.5)
            else:
                r.font.size = Pt(10.5)

        # Render Date
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(10)
        r = p.add_run(date_line)
        r.font.size = Pt(11)

        # Render Recipient
        for j, rl in enumerate(recipient_lines):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1.5 if j < len(recipient_lines) - 1 else 10)
            r = p.add_run(rl)
            r.font.size = Pt(10.5)

        # Render Salutation
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(8)
        _add_runs(p, lines[salutation_idx])

        body_start = salutation_idx + 1
    else:
        # Prepend canonical sender, date, and recipient blocks if not in text
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        r = p.add_run(letterhead["name"])
        r.bold = True
        r.font.size = Pt(11.5)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        p.add_run(letterhead["address_line1"]).font.size = Pt(10.5)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        p.add_run(letterhead["city_state_zip"]).font.size = Pt(10.5)

        # Date
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(10)
        p.add_run(datetime.now().strftime("%B %d, %Y")).font.size = Pt(11)

        # Recipient
        rec_info = recipient_info or {}
        comp = rec_info.get("company", "Hiring Organization")
        title = rec_info.get("title", "")
        loc = rec_info.get("location", "")

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        rec_title = f"Search Committee / Hiring Team for {title}".strip() if title else "Search Committee / Hiring Team"
        p.add_run(rec_title).font.size = Pt(10.5)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5 if loc else 10)
        p.add_run(comp).font.size = Pt(10.5)

        if loc:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(10)
            p.add_run(loc).font.size = Pt(10.5)

        # Salutation
        if salutation_idx == 0:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            _add_runs(p, lines[0])
            body_start = 1
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            p.add_run("Dear Hiring Team and Search Committee Members,").font.size = Pt(11)
            body_start = 0

    body_end = closing_idx if closing_idx != -1 else len(lines)

    # Render Body
    for i in range(body_start, body_end):
        bline = lines[i]
        is_thank_you = "thank you" in bline.lower() and (closing_idx != -1 and i == closing_idx - 1)
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(8 if not is_thank_you else 6)
        p.paragraph_format.line_spacing = 1.15
        _add_runs(p, bline)

    # Render Closing and Signature Block
    if closing_idx != -1:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        _add_runs(p, lines[closing_idx])

        sig_lines = lines[closing_idx + 1:]
        if sig_lines:
            for j, sl in enumerate(sig_lines):
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(1.5 if j < len(sig_lines) - 1 else 0)
                if j == 0:
                    r = p.add_run(sl)
                    r.bold = True
                    r.font.size = Pt(11.5)
                elif "@" in sl or "|" in sl or "+" in sl:
                    r = p.add_run(sl)
                    r.font.size = Pt(10)
                    r.font.color.rgb = RGBColor(71, 85, 105)
                else:
                    r = p.add_run(sl)
                    r.font.size = Pt(10.5)
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1.5)
            r = p.add_run(letterhead["name"])
            r.bold = True
            r.font.size = Pt(11.5)

            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1.5)
            p.add_run(letterhead["department"]).font.size = Pt(10.5)

            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(1.5)
            p.add_run(letterhead["institution"]).font.size = Pt(10.5)

            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            contact_str = f"{letterhead['email']} | {letterhead['phone']}"
            r = p.add_run(contact_str)
            r.font.size = Pt(10)
            r.font.color.rgb = RGBColor(71, 85, 105)
    else:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(6)
        p.add_run("Sincerely,").font.size = Pt(11)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        r = p.add_run(letterhead["name"])
        r.bold = True
        r.font.size = Pt(11.5)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        p.add_run(letterhead["department"]).font.size = Pt(10.5)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(1.5)
        p.add_run(letterhead["institution"]).font.size = Pt(10.5)

        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        contact_str = f"{letterhead['email']} | {letterhead['phone']}"
        r = p.add_run(contact_str)
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(71, 85, 105)

    doc.save(str(out_path))
    return out_path

