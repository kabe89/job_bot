from jobbot.discovery.model import DiscoveredTarget


def _t(**kw):
    base = dict(provider="greenhouse", key="acme", display_name="Acme", source="harvest")
    base.update(kw)
    return DiscoveredTarget(**base)


def test_coord_is_provider_key():
    assert _t().coord() == ("greenhouse", "acme")


def test_roundtrip_to_from_dict():
    t = _t(wd_host=None, valid=True, fit_score=0.8, sample_titles=["Scientist"], job_count=3)
    d = t.to_dict()
    back = DiscoveredTarget.from_dict(d)
    assert back == t
    assert back.sample_titles == ["Scientist"]


def test_defaults_are_pending_and_unvalidated():
    t = _t()
    assert t.status == "pending" and t.valid is False and t.fit_score == 0.0
