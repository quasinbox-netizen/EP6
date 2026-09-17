"""Tests for the Strategy Reality Check.

Two things are worth more than the rest here:

* the loader refuses bad input instead of auditing it. A tool that returns a
  confident verdict on a misread file is worse than one that crashes, so every
  refusal has a test.
* the verdict is not gameable by the licence. An evaluation run and a licensed
  run must reach the same verdict on the same data; only precision differs.
"""
from __future__ import annotations

import base64
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from audit import checks as C
from audit import license as lic
from audit.loader import LoadError, load_csv
from audit.render import to_html, to_text
from audit.report import REJECTED, SURVIVES, UNPROVEN, run_audit

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# --------------------------------------------------------------------------
# fixtures: synthetic track records with a KNOWN answer
# --------------------------------------------------------------------------
def _price_path(days: int = 900, drift: float = 0.0006, vol: float = 0.02, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, size=days)))
    return dates, close


def write_random_rule(path, days: int = 900, seed: int = 7):
    """A rule whose timing carries no information: positions drawn at random.

    The audit must reject this. If it ever passes, the permutation test has
    stopped working.
    """
    dates, close = _price_path(days, seed=seed)
    rng = np.random.default_rng(seed + 1)
    position = (rng.random(days) > 0.5).astype(float)
    pd.DataFrame({"date": dates, "close": close, "position": position}).to_csv(path, index=False)
    return path


def write_real_edge(path, days: int = 900, seed: int = 11):
    """A rule with a genuine edge, built by construction rather than by search.

    The position is set from the SIGN OF THE NEXT DAY, which is look-ahead and
    therefore cheating - deliberately. It is the positive control: if the
    permutation test cannot detect an edge that was placed there on purpose,
    it cannot detect one that is real either.
    """
    dates, close = _price_path(days, seed=seed)
    forward = np.diff(close, append=close[-1])
    position = (forward > 0).astype(float)
    pd.DataFrame({"date": dates, "close": close, "position": position}).to_csv(path, index=False)
    return path


@pytest.fixture
def random_rule(tmp_path):
    return load_csv(write_random_rule(tmp_path / "random.csv"))


@pytest.fixture
def real_edge(tmp_path):
    return load_csv(write_real_edge(tmp_path / "edge.csv"))


# --------------------------------------------------------------------------
# loader: what it accepts
# --------------------------------------------------------------------------
def test_positions_form_reconstructs_returns(random_rule):
    assert random_rule.form == "positions"
    assert random_rule.can_retime
    assert len(random_rule.net_returns) > 800


def test_equity_curve_starts_at_exactly_one(random_rule):
    """The first day's return must not divide out of the totals."""
    assert random_rule.equity.iloc[0] == pytest.approx(1.0)
    recomputed = float((1 + random_rule.net_returns).prod())
    assert float(random_rule.equity.iloc[-1]) == pytest.approx(recomputed, rel=1e-9)


def test_equity_form_disables_retiming(tmp_path):
    dates = pd.date_range("2021-01-01", periods=400, freq="D")
    rng = np.random.default_rng(3)
    equity = 10_000 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, size=400)))
    path = tmp_path / "equity.csv"
    pd.DataFrame({"Date": dates, "Balance": equity}).to_csv(path, index=False)
    data = load_csv(path)
    assert data.form == "equity"
    assert not data.can_retime
    assert any("re-timing test is unavailable" in note for note in data.notes)


def test_return_column_in_percent_is_rescaled(tmp_path):
    dates = pd.date_range("2021-01-01", periods=300, freq="D")
    rng = np.random.default_rng(5)
    path = tmp_path / "returns.csv"
    pd.DataFrame({"date": dates, "daily_return": rng.normal(0.05, 1.2, size=300)}).to_csv(
        path, index=False
    )
    data = load_csv(path)
    assert data.form == "returns"
    assert data.net_returns.abs().max() < 0.2
    assert any("divided by 100" in note for note in data.notes)


def test_european_decimals_and_semicolons(tmp_path):
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    rng = np.random.default_rng(9)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=200)))
    path = tmp_path / "eu.csv"
    rows = ["date;close;position"]
    for stamp, price in zip(dates, close):
        rows.append(f"{stamp.date()};{price:.2f}".replace(".", ",") + ";1,0")
    path.write_text("\n".join(rows), encoding="utf-8")
    data = load_csv(path)
    assert data.form == "positions"
    assert data.net_returns.notna().all()


def test_column_aliases_are_case_insensitive(tmp_path):
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    path = tmp_path / "alias.csv"
    pd.DataFrame({
        "Timestamp": dates,
        "Close Price": np.linspace(100, 160, 200),
        "Weight": 1.0,
    }).to_csv(path, index=False)
    data = load_csv(path)
    assert data.form == "positions"


def test_benchmark_column_is_used(tmp_path):
    dates = pd.date_range("2021-01-01", periods=300, freq="D")
    rng = np.random.default_rng(13)
    path = tmp_path / "bench.csv"
    pd.DataFrame({
        "date": dates,
        "equity": 1000 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=300))),
        "benchmark": 1000 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, size=300))),
    }).to_csv(path, index=False)
    data = load_csv(path)
    assert data.benchmark_returns is not None


def test_held_convention_shifts_the_position(tmp_path):
    path = write_random_rule(tmp_path / "conv.csv")
    decided = load_csv(path, convention="decided")
    held = load_csv(path, convention="held")
    assert not decided.positions.equals(held.positions)
    # `held` is the same series one step earlier: row t+1 of the file.
    assert held.positions.iloc[0] == pytest.approx(decided.positions.iloc[1])


def test_duplicate_dates_keep_the_last_row(tmp_path):
    dates = list(pd.date_range("2021-01-01", periods=200, freq="D"))
    dates.append(dates[-1])
    path = tmp_path / "dupes.csv"
    pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 200, len(dates)),
        "position": 1.0,
    }).to_csv(path, index=False)
    data = load_csv(path)
    assert any("duplicate date" in note for note in data.notes)


# --------------------------------------------------------------------------
# loader: what it refuses. Each of these would otherwise produce a confident
# verdict about something that is not what the user meant.
# --------------------------------------------------------------------------
def test_refuses_missing_file(tmp_path):
    with pytest.raises(LoadError, match="no such file"):
        load_csv(tmp_path / "absent.csv")


def test_refuses_without_a_date_column(tmp_path):
    path = tmp_path / "nodate.csv"
    pd.DataFrame({"close": np.arange(100.0), "position": 1.0}).to_csv(path, index=False)
    with pytest.raises(LoadError, match="no date column"):
        load_csv(path)


def test_refuses_too_few_rows(tmp_path):
    path = tmp_path / "short.csv"
    pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=30, freq="D"),
        "close": np.linspace(100, 110, 30),
        "position": 1.0,
    }).to_csv(path, index=False)
    with pytest.raises(LoadError, match="usable rows"):
        load_csv(path)


def test_refuses_unparseable_dates(tmp_path):
    path = tmp_path / "baddate.csv"
    dates = [f"2021-01-{d:02d}" for d in range(1, 29)] * 3
    dates[5] = "not a date"
    pd.DataFrame({
        "date": dates, "close": np.linspace(100, 200, len(dates)), "position": 1.0,
    }).to_csv(path, index=False)
    with pytest.raises(LoadError, match="unreadable date"):
        load_csv(path)


def test_refuses_positions_that_are_really_percentages(tmp_path):
    path = tmp_path / "pct.csv"
    pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=200, freq="D"),
        "close": np.linspace(100, 200, 200),
        "position": 100.0,
    }).to_csv(path, index=False)
    with pytest.raises(LoadError, match="percent"):
        load_csv(path)


def test_refuses_negative_prices(tmp_path):
    path = tmp_path / "neg.csv"
    close = np.linspace(100, 200, 200)
    close[10] = -5.0
    pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=200, freq="D"),
        "close": close, "position": 1.0,
    }).to_csv(path, index=False)
    with pytest.raises(LoadError, match="negative"):
        load_csv(path)


def test_refuses_an_equity_curve_that_hits_zero(tmp_path):
    path = tmp_path / "wipeout.csv"
    equity = np.linspace(1000, 100, 200)
    equity[150] = 0.0
    pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=200, freq="D"), "equity": equity,
    }).to_csv(path, index=False)
    with pytest.raises(LoadError, match="zero or negative"):
        load_csv(path)


def test_refuses_an_unknown_convention(tmp_path):
    path = write_random_rule(tmp_path / "c.csv")
    with pytest.raises(LoadError, match="convention"):
        load_csv(path, convention="whatever")


def test_refuses_an_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(LoadError, match="empty"):
        load_csv(path)


# --------------------------------------------------------------------------
# the checks themselves
# --------------------------------------------------------------------------
def test_periods_per_year_distinguishes_crypto_from_equities():
    daily = pd.date_range("2021-01-01", periods=400, freq="D")
    business = pd.date_range("2021-01-01", periods=400, freq="B")
    assert C.infer_periods_per_year(daily) == 365
    assert C.infer_periods_per_year(business) == 252


def test_retiming_detects_an_edge_placed_on_purpose(real_edge):
    check = C.retiming(real_edge, permutations=400, cost_rate=0.0, periods_per_year=365)
    assert check.verdict == C.PASS, check.headline
    assert check.numbers["p_value"] < 0.05


def test_retiming_rejects_random_timing(random_rule):
    check = C.retiming(random_rule, permutations=400, cost_rate=0.0, periods_per_year=365)
    assert check.verdict == C.FAIL, check.headline


def test_retiming_is_not_available_without_positions(tmp_path):
    dates = pd.date_range("2021-01-01", periods=300, freq="D")
    rng = np.random.default_rng(21)
    path = tmp_path / "eq.csv"
    pd.DataFrame({
        "date": dates, "equity": 1000 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300))),
    }).to_csv(path, index=False)
    check = C.retiming(load_csv(path), permutations=200, cost_rate=0.0, periods_per_year=365)
    assert check.verdict == C.NA
    assert "position column" in check.headline


def test_multiplicity_arithmetic_matches_bonferroni():
    check = C.multiplicity(0.01, variants_tried=20)
    assert check.verdict == C.FAIL
    assert check.numbers["p_value_bonferroni"] == pytest.approx(0.20)
    assert check.numbers["expected_false_positives"] == pytest.approx(1.0)


def test_multiplicity_can_be_survived():
    check = C.multiplicity(0.0001, variants_tried=10)
    assert check.verdict == C.PASS
    assert check.numbers["p_value_bonferroni"] == pytest.approx(0.001)


def test_multiplicity_warns_when_not_declared():
    check = C.multiplicity(0.01, variants_tried=None)
    assert check.verdict == C.WARN
    assert "--variants-tried" in check.detail


def test_concentration_catches_a_record_carried_by_one_day():
    index = pd.date_range("2021-01-01", periods=300, freq="D")
    returns = pd.Series(-0.0005, index=index)
    returns.iloc[100] = 0.60
    data = type("Stub", (), {
        "net_returns": returns,
        "equity": (1 + returns).cumprod(),
    })()
    check = C.concentration(data)
    assert check.verdict == C.FAIL
    assert "day" in check.headline


def test_cost_sensitivity_breakeven_is_exact():
    """gross_log = 0.10 over 10 units of turnover means breakeven at 100 bps."""
    index = pd.date_range("2021-01-01", periods=201, freq="D")
    positions = pd.Series(0.0, index=index)
    # Ten held stretches of ten days each, alternating with ten flat ones.
    for start in range(0, 200, 20):
        positions.iloc[start:start + 10] = 1.0
    asset = pd.Series(0.0, index=index)
    # Put all the gross return inside held stretches, applied to the next day.
    held = positions.to_numpy()[:-1] > 0
    asset.iloc[1:] = np.where(held, np.expm1(0.10 / held.sum()), 0.0)
    data = type("Stub", (), {"positions": positions, "asset_returns": asset})()
    check = C.cost_sensitivity(data, cost_rate=0.0, applied_bps=0.0)

    # 10 entries and 9 exits inside the window, each one unit of position.
    assert check.numbers["turnover"] == pytest.approx(19.0)
    assert check.numbers["gross_log_return"] == pytest.approx(0.10, rel=1e-6)
    # The defining property: at the breakeven cost the net log return is zero.
    breakeven = check.numbers["breakeven_bps"] / 10_000.0
    assert (check.numbers["gross_log_return"]
            - breakeven * check.numbers["turnover"]) == pytest.approx(0.0, abs=1e-12)
    assert check.numbers["breakeven_bps"] == pytest.approx(0.10 / 19.0 * 10_000, rel=1e-6)


def test_survivability_fails_a_ruinous_drawdown():
    index = pd.date_range("2021-01-01", periods=500, freq="D")
    returns = pd.Series(0.0, index=index)
    returns.iloc[10:200] = -0.01
    data = type("Stub", (), {
        "net_returns": returns, "equity": (1 + returns).cumprod(),
    })()
    check = C.survivability(data, periods_per_year=365)
    assert check.verdict == C.FAIL


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def test_random_rule_is_rejected(random_rule):
    report = run_audit(random_rule, permutations=300, variants_tried=25)
    assert report.verdict == REJECTED
    assert report.failures


def test_verdict_grades_are_the_only_three(real_edge, random_rule):
    for data in (real_edge, random_rule):
        report = run_audit(data, permutations=200, variants_tried=1)
        assert report.verdict in {SURVIVES, UNPROVEN, REJECTED}


def test_report_counts_add_up(random_rule):
    report = run_audit(random_rule, permutations=200)
    counts = report.meta["counts"]
    assert (counts["passed"] + counts["failed"] + counts["weak"] + counts["not_run"]
            == counts["blocking_total"])


def test_json_round_trips(tmp_path, random_rule):
    report = run_audit(random_rule, permutations=200)
    path = report.to_json(tmp_path / "out.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["verdict"] == report.verdict
    assert len(payload["checks"]) == len(report.checks)
    # Every number must survive JSON: NaN and numpy scalars are the usual leak.
    assert "NaN" not in path.read_text(encoding="utf-8")


def test_evaluation_mode_reaches_the_same_verdict(random_rule):
    """The free tier is less precise, never differently honest."""
    licensed = run_audit(random_rule, licence=lic.Licence(valid=True, reason="", licensee="T"),
                         permutations=400, variants_tried=25)
    evaluation = run_audit(random_rule, permutations=400, variants_tried=25)
    assert evaluation.meta["permutations"] <= lic.EVAL_PERMUTATIONS
    assert licensed.verdict == evaluation.verdict


def test_undeclared_variants_cannot_produce_a_clean_pass(real_edge):
    """Without a declared variant count the report can never say SURVIVES."""
    report = run_audit(real_edge, permutations=400, variants_tried=None)
    assert report.by_key("multiplicity").verdict == C.WARN
    assert report.verdict != SURVIVES


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def test_html_is_self_contained(random_rule):
    report = run_audit(random_rule, permutations=200)
    page = to_html(report, random_rule)
    assert page.startswith("<!doctype html>")
    for forbidden in ("http://", "https://", "<script"):
        assert forbidden not in page, f"the report reaches outside itself: {forbidden}"
    assert report.verdict in page
    assert "not advice" in page


def test_html_marks_an_evaluation_copy(random_rule):
    report = run_audit(random_rule, permutations=200)
    assert "EVALUATION COPY" in to_html(report, random_rule)


def test_text_report_names_every_check(random_rule):
    report = run_audit(random_rule, permutations=200)
    text = to_text(report)
    for check in report.checks:
        assert check.title.upper() in text


def test_html_escapes_a_hostile_label(random_rule):
    report = run_audit(random_rule, permutations=200, label='<script>alert(1)</script>')
    page = to_html(report, random_rule)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


# --------------------------------------------------------------------------
# licensing
# --------------------------------------------------------------------------
def _keypair():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    public = base64.b64encode(
        private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode()
    return private, public


def _token(private, **overrides) -> str:
    payload = {
        "product": lic.PRODUCT,
        "licensee": "Test Sp. z o.o.",
        "issued": date.today().isoformat(),
        "expires": (date.today() + timedelta(days=365)).isoformat(),
        "seats": 1,
        "ref": "T1",
    }
    payload.update(overrides)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    signature = private.sign(raw)
    enc = lambda b: base64.urlsafe_b64encode(b).decode().rstrip("=")  # noqa: E731
    return f"{enc(raw)}.{enc(signature)}"


def test_a_signed_licence_verifies():
    private, public = _keypair()
    result = lic.verify(_token(private), public_key_b64=public)
    assert result.valid and result.licensee == "Test Sp. z o.o."


def test_an_edited_payload_is_rejected():
    """The whole design rests on this: editing the expiry must break the signature."""
    private, public = _keypair()
    payload_b64, signature_b64 = _token(private).split(".")
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
    payload["expires"] = "2099-01-01"
    forged = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    result = lic.verify(f"{forged}.{signature_b64}", public_key_b64=public)
    assert not result.valid and "signature" in result.reason


def test_an_expired_licence_is_rejected():
    private, public = _keypair()
    token = _token(private, expires=(date.today() - timedelta(days=1)).isoformat())
    result = lic.verify(token, public_key_b64=public)
    assert not result.valid and "expired" in result.reason


def test_a_licence_for_another_product_is_rejected():
    private, public = _keypair()
    result = lic.verify(_token(private, product="something-else"), public_key_b64=public)
    assert not result.valid


def test_a_licence_signed_by_the_wrong_key_is_rejected():
    private, _ = _keypair()
    _, other_public = _keypair()
    result = lic.verify(_token(private), public_key_b64=other_public)
    assert not result.valid


@pytest.mark.parametrize("junk", ["", "garbage", "a.b", "....", "!!!.???", "x" * 4000])
def test_malformed_keys_never_raise(junk):
    _, public = _keypair()
    result = lic.verify(junk, public_key_b64=public)
    assert not result.valid and result.reason


def test_a_build_without_a_vendor_key_runs_in_evaluation():
    result = lic.verify("anything", public_key_b64="")
    assert not result.valid and "no vendor key" in result.reason


def test_read_key_prefers_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(lic.KEY_ENV, "from-env")
    assert lic.read_key(root=tmp_path) == "from-env"
    monkeypatch.delenv(lic.KEY_ENV)
    (tmp_path / lic.KEY_FILENAME).write_text("from-file\n", encoding="utf-8")
    assert lic.read_key(root=tmp_path) == "from-file"


def test_keytool_issues_a_key_its_own_verifier_accepts(monkeypatch, capsys, tmp_path):
    from cryptography.hazmat.primitives import serialization

    from audit import keytool

    private, public = _keypair()
    raw = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    monkeypatch.setenv(keytool.SIGNING_KEY_ENV, base64.b64encode(raw).decode())
    out = tmp_path / "licence.key"
    assert keytool.main(["issue", "--licensee", "ACME", "--months", "6",
                         "--out", str(out)]) == 0
    token = out.read_text(encoding="utf-8").strip()
    result = lic.verify(token, public_key_b64=public)
    assert result.valid and result.licensee == "ACME"


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------
def test_cli_writes_an_example_and_audits_it(tmp_path, capsys):
    import cli

    sample = tmp_path / "demo.csv"
    assert cli.main(["audit", "--example", str(sample)]) == 0
    assert sample.exists()

    assert cli.main(["audit", "--file", str(sample), "--variants-tried", "5",
                     "--cost-bps", "20", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "demo-reality-check.html").exists()
    assert (tmp_path / "demo-reality-check.json").exists()
    assert "VERDICT" in capsys.readouterr().out


def test_cli_reports_a_bad_file_without_a_traceback(tmp_path, capsys):
    import cli

    path = tmp_path / "junk.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    assert cli.main(["audit", "--file", str(path)]) == 1
    assert "cannot audit" in capsys.readouterr().out


def test_cli_strict_exits_non_zero_on_rejection(tmp_path):
    import cli

    path = write_random_rule(tmp_path / "random.csv")
    assert cli.main(["audit", "--file", str(path), "--variants-tried", "50",
                     "--strict", "--out", str(tmp_path)]) == 1


def test_column_names_with_diacritics_are_matched(tmp_path):
    """A Polish export should not need renaming before it can be audited."""
    path = tmp_path / "pl.csv"
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    header = "Data;Kurs;Pozycja"  # non-english-ok: this is the fixture, not a leak
    rows = [header]
    for stamp, price in zip(dates, np.linspace(100, 180, 200)):
        rows.append(f"{stamp.date()};{price:.2f};1,0".replace(".", ",", 1))
    path.write_text("\n".join(rows), encoding="utf-8")
    data = load_csv(path)
    assert data.form == "positions"
