# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Which clock a piece of code reads, and what it does when that clock moves.

Three rules, from docs/time_audit_20260731.md:

  * a timestamp that crosses the API is UTC, and a value with no zone suffix
    means UTC rather than local;
  * an interval is measured with time.perf_counter() and a deadline is set on
    time.monotonic(), because the wall clock can step backwards;
  * except across processes, where the wall clock is the only shared one --
    time.monotonic() has no origin and means nothing outside the process that
    read it.

The scans at the bottom exist because the sites they cover cannot be reached
from a unit test without standing up an export or a model.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core import summary_cache

_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clean_cache():
    summary_cache.invalidate_projects_summary_cache()
    yield
    summary_cache.invalidate_projects_summary_cache()


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, start: float = 1234.5):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_the_summary_cache_serves_a_fresh_entry(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(summary_cache.time, "monotonic", clock)
    summary_cache.set_cached_summary(["a"])
    clock.advance(summary_cache.PROJECTS_SUMMARY_TTL_SEC / 2)
    assert summary_cache.get_cached_summary() == ["a"]


def test_the_summary_cache_expires_once_the_ttl_has_passed(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(summary_cache.time, "monotonic", clock)
    summary_cache.set_cached_summary(["a"])
    clock.advance(summary_cache.PROJECTS_SUMMARY_TTL_SEC + 1)
    assert summary_cache.get_cached_summary() is None


def test_a_backward_wall_clock_step_cannot_extend_the_cache(monkeypatch):
    """An NTP correction, or a user fixing the clock, used to hold the summary
    frozen for the length of the step -- the deadline sat an hour in the future
    of a wall clock that had just moved an hour into the past."""
    clock = _Clock()
    monkeypatch.setattr(summary_cache.time, "monotonic", clock)
    wall = _Clock(start=1_700_000_000.0)
    monkeypatch.setattr(summary_cache.time, "time", wall)

    summary_cache.set_cached_summary(["a"])
    wall.advance(-3600)  # the wall clock jumps an hour backwards
    clock.advance(summary_cache.PROJECTS_SUMMARY_TTL_SEC + 1)

    assert summary_cache.get_cached_summary() is None


def test_the_invalidated_deadline_sits_outside_the_clocks_range():
    """time.monotonic() has no defined origin, so a reading can be small, and a
    sentinel inside its range is not a sentinel: 0.0 would read as a live
    deadline on a machine whose counter had not passed it yet.

    Asserted on the field rather than through get_cached_summary(), which also
    clears ``data`` and would pass whatever the deadline said -- so the test
    would pin nothing.
    """
    summary_cache.set_cached_summary(["a"])
    summary_cache.invalidate_projects_summary_cache()
    assert summary_cache._cache["expires_at"] < 0


# ---------------------------------------------------------------------------
# The sites a unit test cannot reach
# ---------------------------------------------------------------------------
_NAIVE_ISO = re.compile(r"datetime\.now\(\s*\)\s*\.isoformat\(\)")


def _app_sources() -> list[Path]:
    files = []
    for app in ("trainer_api", "serving_api"):
        files.extend(sorted((_ROOT / "apps" / app / "app").rglob("*.py")))
    assert files, "found no application sources to scan"
    return files


def test_no_shipped_module_serialises_a_naive_now():
    """datetime.now() is local. Written without a zone it is read as UTC by
    anything following the convention, which in JST is nine hours out -- the
    export manifest carried exactly that. The fix is datetime.now(timezone.utc);
    a filename stamp that genuinely wants local time does not go through
    isoformat() and so does not match here.
    """
    offenders = [
        f"{f.relative_to(_ROOT).as_posix()}:{i}"
        for f in _app_sources()
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        if _NAIVE_ISO.search(line)
    ]
    assert offenders == [], f"naive local timestamp serialised at {offenders}"


def test_the_matcher_above_recognises_a_naive_now():
    """A matcher that never matches would pass the test above on any tree."""
    assert _NAIVE_ISO.search('"exported_at": datetime.now().isoformat(),')
    assert not _NAIVE_ISO.search('"exported_at": datetime.now(timezone.utc).isoformat(),')


@pytest.mark.parametrize("rel", [
    "apps/trainer_api/app/core/summary_cache.py",
    "apps/trainer_api/app/routers/training_library.py",
])
def test_the_in_process_caches_set_their_deadline_on_monotonic(rel):
    """Both caches are module globals compared only against themselves, so
    nothing needs a shared origin -- and a wall clock that steps backwards used
    to hold them unexpired for the length of the step."""
    source = (_ROOT / rel).read_text(encoding="utf-8")
    assert "time.monotonic()" in source
    assert "time.time()" not in source


@pytest.mark.parametrize("rel", [
    "apps/trainer_api/app/core/crack_trace.py",
    "apps/trainer_api/app/routers/ai_assist.py",
    "apps/serving_api/app/main.py",
])
def test_reported_latencies_are_not_measured_on_the_wall_clock(rel):
    """Every time.time() in these three was a t0/elapsed pair whose result is
    returned to a caller as a duration. Across a clock step such a difference is
    wrong and can be negative. A timestamp here would want
    datetime.now(timezone.utc) rather than time.time() anyway.
    """
    source = (_ROOT / rel).read_text(encoding="utf-8")
    assert "time.time()" not in source
    assert "time.perf_counter()" in source


# ---------------------------------------------------------------------------
# The deprecated UTC constructors, removed without moving a single character of
# the output. Unifying the API on tz-aware strings is a separate decision.
# ---------------------------------------------------------------------------
_DEPRECATED_UTC = re.compile(r"datetime\.utc(now|fromtimestamp)\(")

# 2023-11-14T22:13:20Z
_FIXED_EPOCH = 1_700_000_000.0


def test_a_last_modified_header_keeps_its_shape():
    """The format string carries the zone itself, so the naive and the aware
    datetime render identically -- the swap is invisible on the wire."""
    from app.routers.annotate import _http_date

    assert _http_date(_FIXED_EPOCH) == "Tue, 14 Nov 2023 22:13:20 GMT"


def test_the_pretrained_timestamp_still_carries_no_zone_suffix():
    """This field has always been unsuffixed and readers take that as UTC.
    Removing a deprecation must not be how the API format changes.
    """
    from app.routers.pretrained import _naive_utc_iso

    assert _naive_utc_iso(_FIXED_EPOCH) == "2023-11-14T22:13:20"


def _deprecation_scan_targets() -> list[Path]:
    files = _app_sources()
    cli = _ROOT / "scripts" / "cli_train.py"
    assert cli.exists(), "scripts/cli_train.py moved"
    return [*files, cli]


def test_nothing_shipped_calls_the_deprecated_utc_constructors():
    """Both return naive datetimes and both are deprecated from Python 3.12.
    The naive return is also where the mixing of naive and tz-aware values in
    the runs listing starts, so they are worth keeping out.
    """
    offenders = [
        f"{f.relative_to(_ROOT).as_posix()}:{i}"
        for f in _deprecation_scan_targets()
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        if _DEPRECATED_UTC.search(line)
    ]
    assert offenders == [], f"deprecated UTC constructor at {offenders}"


def test_the_deprecation_matcher_recognises_both_calls():
    assert _DEPRECATED_UTC.search("datetime.utcnow().isoformat()")
    assert _DEPRECATED_UTC.search("datetime.utcfromtimestamp(mtime)")
    assert not _DEPRECATED_UTC.search("datetime.now(timezone.utc).isoformat()")
    assert not _DEPRECATED_UTC.search("datetime.fromtimestamp(ts, tz=timezone.utc)")


# ---------------------------------------------------------------------------
# Local time is for names. Everything else is UTC.
# ---------------------------------------------------------------------------
class _FrozenClock:
    """09:00 UTC, which the local zone in this test renders as 18:00."""

    _UTC = datetime(2026, 7, 31, 9, 0, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls._UTC.astimezone(tz) if tz is not None else datetime(2026, 7, 31, 18, 0)


def test_a_filename_stamp_reads_the_local_clock(monkeypatch):
    """Frozen rather than compared against the machine's own clock: on a
    UTC runner the two readings coincide and the test would pass either way.
    """
    from app.core import paths

    monkeypatch.setattr(paths, "datetime", _FrozenClock)
    assert paths.local_file_stamp() == "20260731_180000"
    assert paths.local_file_stamp("%Y%m%d_%H%M") == "20260731_1800"
    # And the UTC reading of the same instant is a different string, which is
    # the whole reason the two had to stop being chosen site by site.
    assert paths.local_file_stamp() != _FrozenClock.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def test_only_one_place_turns_the_clock_into_a_filename():
    """Exports were named in local time and reports in UTC, so two files made
    in the same minute were named hours apart. Whichever way that is settled,
    it is settled once -- a second site formatting its own is how they drifted
    apart in the first place, and nothing about either site announced it.
    """
    pattern = re.compile(r"datetime\.now\([^)]*\)\.strftime\(")
    files = sorted({
        f.relative_to(_ROOT).as_posix()
        for f in _app_sources()
        for line in f.read_text(encoding="utf-8").splitlines()
        if pattern.search(line)
    })
    assert files == ["apps/trainer_api/app/core/paths.py"], (
        f"a filename stamp is formatted outside local_file_stamp(): {files}"
    )


def _report_meta(report_id: str, created_at: str) -> dict:
    return {
        "report_id": report_id,
        "report_type": "quality",
        "run_id": "run-1",
        "files": [],
        "created_at": created_at,
    }


def _seed_reports(tmp_path, monkeypatch, metas):
    import json as _json

    from app.routers import reports as reports_router

    root = tmp_path / "reports"
    root.mkdir()
    for meta in metas:
        d = root / meta["report_id"]
        d.mkdir()
        (d / "meta.json").write_text(_json.dumps(meta), encoding="utf-8")
    monkeypatch.setattr(reports_router, "_reports_dir", lambda _pid: root)
    return reports_router


def test_the_report_list_is_ordered_by_when_the_report_was_made(tmp_path, monkeypatch):
    """It used to be ordered by the directory's name, which is stamped in local
    time -- so around the moment that stamp stopped being UTC the list would
    reshuffle, by the size of the reader's offset and in whichever direction it
    runs. The instant inside the report does not move.
    """
    router = _seed_reports(tmp_path, monkeypatch, [
        # The name says oldest, the timestamp says newest.
        _report_meta("20260731_0000_aaaa", "2026-07-31T23:00:00+00:00"),
        _report_meta("20260731_1200_bbbb", "2026-07-31T01:00:00+00:00"),
        _report_meta("20260731_2300_cccc", "2026-07-31T12:00:00+00:00"),
    ])
    order = [it.report_id for it in router.list_reports("p1")]
    assert order == ["20260731_0000_aaaa", "20260731_2300_cccc", "20260731_1200_bbbb"]


def test_a_report_without_a_zone_does_not_break_the_ordering(tmp_path, monkeypatch):
    """Comparing a naive datetime against a tz-aware one raises, so this would
    have been a 500 rather than an odd order. Naive means UTC."""
    router = _seed_reports(tmp_path, monkeypatch, [
        _report_meta("20260731_0000_aaaa", "2026-07-31T23:00:00"),
        _report_meta("20260731_1200_bbbb", "2026-07-31T01:00:00+00:00"),
    ])
    order = [it.report_id for it in router.list_reports("p1")]
    assert order == ["20260731_0000_aaaa", "20260731_1200_bbbb"]
