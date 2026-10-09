"""NY State's results table is: Item # | Title | Grade | Posted | Deadline | Agency | County

The agency scan took the FIRST cell containing any of ("Department", "Health",
"DOH", "Research", "University", "State"). Title is column 1 and Agency is
column 5, so any title carrying one of those words matched first and was stored
as the company -- 6 of 21 real rows ("Research Scientist 2 (Health Services
Research)" became its own employer).

Keyword-sniffing was never sound anyway: real agencies like "People With
Developmental Disabilities, Office for", "Children & Family Services, Office" and
"Podiatry Board" contain none of the keywords, so they could never be found. The
header row names the columns -- read them.

State agencies are a search target, so a wrong company here also
breaks watchlist and outreach matching against "Health, Department of".
"""
from jobbot.scrapers.nystate import _parse_vacancy_table

_HEADERS = ["Item #", "Title", "Grade", "Posted", "Deadline", "Agency", "County"]


def _table(*rows, headers=_HEADERS):
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    return f"<html><body><table>{head}{''.join(rows)}</table></body></html>"


def _row(vac_id, title, agency, county="Onondaga", grade="22"):
    return (f"<tr><td>{vac_id}</td>"
            f'<td><a href="vacancyDetailsView.cfm?id={vac_id}">{title}</a></td>'
            f"<td>{grade}</td><td>07/13/26</td><td>07/27/26</td>"
            f"<td>{agency}</td><td>{county}</td></tr>")


def test_title_containing_health_does_not_hijack_the_agency():
    """The exact shape of the real bug: Title (col 1) precedes Agency (col 5)."""
    html = _table(_row("24735",
                       "Research Scientist 3 (Biostatistics/Health Services Research)",
                       "Health, Department of"))
    got = _parse_vacancy_table(html)[0]
    assert got["agency"] == "Health, Department of"
    assert got["title"] == "Research Scientist 3 (Biostatistics/Health Services Research)"
    assert got["agency"] != got["title"]


def test_agency_without_any_keyword_is_still_found():
    """Keyword-sniffing could never find these. The header column can."""
    html = _table(_row("219847", "Research Scientist 2 (Health Services Research)",
                       "People With Developmental Disabilities, Office for"))
    got = _parse_vacancy_table(html)[0]
    assert got["agency"] == "People With Developmental Disabilities, Office for"


def test_another_keywordless_agency():
    html = _table(_row("220070", "Research Associate/Senior Research Scientist",
                       "Children & Family Services, Office"))
    assert _parse_vacancy_table(html)[0]["agency"] == "Children & Family Services, Office"


def test_title_containing_state_does_not_hijack_the_agency():
    html = _table(_row("100", "Research Scientist 2, New York State Psychiatric Institute",
                       "Mental Health, Office of"))
    assert _parse_vacancy_table(html)[0]["agency"] == "Mental Health, Office of"


def test_title_containing_university_does_not_hijack_the_agency():
    html = _table(_row("101", "Assistant Research Scientist - State University",
                       "Health, Department of"))
    assert _parse_vacancy_table(html)[0]["agency"] == "Health, Department of"


def test_county_column_is_used_for_location():
    """The old hardcoded city regex silently dropped counties like Rockland."""
    html = _table(_row("219896", "Research Scientist 4, Nathan S. Kline Institute",
                       "Mental Health, Office of", county="Rockland"))
    assert _parse_vacancy_table(html)[0]["location"] == "Rockland"


def test_title_and_url_still_parse():
    html = _table(_row("105", "Research Scientist 3 (Epidemiology)",
                       "Health, Department of", county="Monroe"))
    got = _parse_vacancy_table(html)[0]
    assert got["title"] == "Research Scientist 3 (Epidemiology)"
    assert "vacancyDetailsView.cfm?id=105" in got["url"]


def test_multiple_rows_each_keep_their_own_agency():
    html = _table(
        _row("1", "Public Health Specialist 1", "Health, Department of"),
        _row("2", "Research Scientist 4", "Mental Health, Office of"),
    )
    rows = _parse_vacancy_table(html)
    assert [r["agency"] for r in rows] == ["Health, Department of",
                                           "Mental Health, Office of"]
    assert all(r["agency"] != r["title"] for r in rows)


def test_agency_falls_back_to_ny_state_when_the_column_is_absent():
    """Unknown layout -> a safe generic company, never the title."""
    html = _table(_row("103", "Public Health Specialist 1", "Podiatry Board"),
                  headers=["Item #", "Title", "Grade", "Posted", "Deadline"])
    got = _parse_vacancy_table(html)[0]
    assert got["agency"] != got["title"]


def test_row_without_a_vacancy_link_is_skipped():
    html = _table("<tr><td>a</td><td>b</td><td>c</td><td>d</td>"
                  "<td>e</td><td>f</td><td>g</td></tr>")
    assert _parse_vacancy_table(html) == []


def test_headerless_table_does_not_crash():
    """Fail-open: no headers -> still return rows, never raise, never use the title."""
    html = ("<html><body><table>"
            + _row("9", "Public Health Specialist 1", "Podiatry Board")
            + "</table></body></html>")
    rows = _parse_vacancy_table(html)
    assert len(rows) == 1
    assert rows[0]["agency"] != rows[0]["title"]
