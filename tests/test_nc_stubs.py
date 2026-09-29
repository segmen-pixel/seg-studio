# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""The files this project may not redistribute leave by every route.

The installer replaced them in its own staging and nothing else did, so
``install.py --offline-pack`` -- the other route the README documents -- saved
the wheels as ``pip download`` left them and the pack carried the file the
installer exists to remove. One table, one behaviour, and a wheel that is
handed on no longer holds it.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import re
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _nc_stubs as nc  # noqa: E402


def _wheel(tmp_path, rel: str, body: str = "raise SystemExit\n") -> Path:
    """A wheel holding one file and a RECORD that agrees with it."""
    wheel = tmp_path / "torchmetrics-1.0.0-py3-none-any.whl"
    payload = body.encode()
    record = "\n".join([nc._record_line(rel, payload),
                         "torchmetrics-1.0.0.dist-info/RECORD,,"]) + "\n"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(rel, payload)
        zf.writestr("torchmetrics-1.0.0.dist-info/METADATA", "Name: torchmetrics\n")
        zf.writestr("torchmetrics-1.0.0.dist-info/RECORD", record)
    return wheel


class TestTheWheelsThemselves:
    """A pack is a redistribution: fixing the unpacked tree is not enough."""

    def test_the_file_is_replaced_inside_the_wheel(self, tmp_path):
        rel = next(iter(nc.NONCOMMERCIAL_STUBS))
        wheel = _wheel(tmp_path, rel, "def extended_edit_distance():\n    return 1\n")
        assert nc.purge_wheel(wheel) == [rel]
        with zipfile.ZipFile(wheel) as zf:
            got = zf.read(rel).decode()
        assert "non-commercial" in got and "return 1" not in got

    def test_the_record_still_describes_the_contents(self, tmp_path):
        """A wheel whose RECORD disagrees fails --require-hashes and confuses
        anything that verifies a package after the fact."""
        rel = next(iter(nc.NONCOMMERCIAL_STUBS))
        wheel = _wheel(tmp_path, rel)
        nc.purge_wheel(wheel)
        with zipfile.ZipFile(wheel) as zf:
            record = zf.read("torchmetrics-1.0.0.dist-info/RECORD").decode()
            data = zf.read(rel)
        line = next(ln for ln in record.splitlines() if ln.startswith(rel))
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        assert line == f"{rel},sha256={digest},{len(data)}"

    def test_a_wheel_without_one_is_left_alone(self, tmp_path):
        wheel = _wheel(tmp_path, "torchmetrics/functional/classification/f1.py")
        before = wheel.read_bytes()
        assert nc.purge_wheel(wheel) == []
        assert wheel.read_bytes() == before, "an untouched wheel keeps its bytes"

    def test_a_directory_of_wheels(self, tmp_path):
        rel = next(iter(nc.NONCOMMERCIAL_STUBS))
        _wheel(tmp_path, rel)
        assert nc.purge_wheel_dir(tmp_path) == {"torchmetrics-1.0.0-py3-none-any.whl": [rel]}


class TestTheUnpackedTree:
    def test_every_module_is_replaced(self, tmp_path):
        for rel in nc.NONCOMMERCIAL_STUBS:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("the upstream implementation\n", encoding="utf-8")
        (tmp_path / "pyphen" / "dictionaries").mkdir(parents=True)
        assert sorted(nc.purge_site_packages(tmp_path)) == sorted(nc.NONCOMMERCIAL_STUBS)
        for rel in nc.NONCOMMERCIAL_STUBS:
            assert "non-commercial" in (tmp_path / rel).read_text(encoding="utf-8")

    def test_a_missing_module_is_an_error_when_it_should_be_there(self, tmp_path):
        """The dependency changed layout, which means its licensing has to be
        looked at again -- not that there is nothing to do."""
        (tmp_path / "pyphen" / "dictionaries").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="torchmetrics"):
            nc.purge_site_packages(tmp_path)

    def test_and_is_not_when_only_checking(self, tmp_path):
        assert nc.purge_site_packages(tmp_path, required=False) == []


def _site(tmp_path: Path, *extra: str) -> Path:
    """A site-packages holding every listed file, as upstream ships them."""
    for rel in [*nc.NONCOMMERCIAL_STUBS, *nc.REMOVED_FILES, *extra]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("upstream\n", encoding="utf-8")
    return tmp_path


class TestTheDataFilesThatAreDeleted:
    """pyphen's GPL-only, unclear and unlicensed hyphenation dictionaries: nothing reads them."""

    def test_they_go_and_the_other_dictionaries_stay(self, tmp_path):
        site = _site(tmp_path, "pyphen/dictionaries/hyph_en_US.dic")
        done = nc.purge_site_packages(site)
        assert set(nc.REMOVED_FILES) <= set(done)
        assert not any((site / rel).exists() for rel in nc.REMOVED_FILES)
        assert (site / "pyphen/dictionaries/hyph_en_US.dic").is_file()

    def test_applying_the_table_twice_is_fine(self, tmp_path):
        site = _site(tmp_path)
        nc.purge_site_packages(site)
        assert nc.pending(site) == []
        assert sorted(nc.purge_site_packages(site)) == sorted(nc.NONCOMMERCIAL_STUBS)

    def test_pending_names_everything_still_to_do(self, tmp_path):
        site = _site(tmp_path)
        assert sorted(nc.pending(site)) == sorted([*nc.NONCOMMERCIAL_STUBS, *nc.REMOVED_FILES])

    def test_a_missing_dictionary_directory_is_an_error_when_required(self, tmp_path):
        for rel in nc.NONCOMMERCIAL_STUBS:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("upstream\n", encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="pyphen/dictionaries"):
            nc.purge_site_packages(tmp_path)
        assert nc.purge_site_packages(tmp_path, required=False) == sorted(nc.NONCOMMERCIAL_STUBS)

    def test_a_wheel_loses_them_and_their_record_lines(self, tmp_path):
        wheel = tmp_path / "pyphen-0.17.2-py3-none-any.whl"
        keep, gone = "pyphen/dictionaries/hyph_en_US.dic", next(iter(nc.REMOVED_FILES))
        record = "\n".join([nc._record_line(keep, b"k"), nc._record_line(gone, b"g"),
                             "pyphen-0.17.2.dist-info/RECORD,,"]) + "\n"
        with zipfile.ZipFile(wheel, "w") as zf:
            zf.writestr(keep, b"k")
            zf.writestr(gone, b"g")
            zf.writestr("pyphen-0.17.2.dist-info/RECORD", record)
        assert nc.purge_wheel(wheel) == [gone]
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            rec = zf.read("pyphen-0.17.2.dist-info/RECORD").decode()
        assert gone not in names and keep in names
        assert gone not in rec and keep in rec


def test_the_tables_were_written_for_the_pinned_versions():
    """A new torchmetrics or pyphen can move a file or change a licence; the
    required-file check catches the first, only a re-read catches the second."""
    lock = (ROOT / "apps/trainer_api/requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^torchmetrics==1\.9\.0$", lock, re.M), "re-check NONCOMMERCIAL_STUBS"
    assert re.search(r"^pyphen==0\.17\.2$", lock, re.M), "re-check REMOVED_FILES"


def test_no_report_turns_hyphenation_on():
    """WeasyPrint reads a pyphen dictionary only for text styled hyphens: auto.
    That is what makes deleting the dictionaries safe."""
    pat = re.compile(r"hyphens\s*:\s*auto", re.I)
    hits = [p.relative_to(ROOT).as_posix() for p in (ROOT / "apps/trainer_api/app").rglob("*")
            if p.is_file() and p.suffix in {".py", ".html", ".css", ".j2", ".jinja"}
            and pat.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert hits == []


def test_the_command_line_applies_it_and_fails_closed(tmp_path, capsys):
    site = _site(tmp_path / "ok")
    assert nc.main([str(site)]) == 0
    assert nc.pending(site) == []
    assert nc.main([str(tmp_path / "empty")]) == 1
    only_dicts = tmp_path / "only-dicts"
    (only_dicts / "pyphen" / "dictionaries").mkdir(parents=True)
    capsys.readouterr()
    assert nc.main([str(only_dicts)]) == 1
    assert "torchmetrics" in capsys.readouterr().err


class TestWhereverPipInstalled:
    """pip falls back to the user site when the environment's is not writable."""

    def test_the_packages_may_sit_in_different_roots(self, tmp_path):
        a, b = tmp_path / "purelib", tmp_path / "user"
        for rel in nc.NONCOMMERCIAL_STUBS:
            (a / rel).parent.mkdir(parents=True, exist_ok=True)
            (a / rel).write_text("upstream\n", encoding="utf-8")
        for rel in nc.REMOVED_FILES:
            (b / rel).parent.mkdir(parents=True, exist_ok=True)
            (b / rel).write_text("upstream\n", encoding="utf-8")
        assert len(nc.pending_installed([a, b])) == len(nc.NONCOMMERCIAL_STUBS) + len(nc.REMOVED_FILES)
        nc.purge_installed([a, b])
        assert nc.pending_installed([a, b]) == []

    def test_a_required_path_in_no_root_is_an_error_and_changes_nothing(self, tmp_path):
        a = _site(tmp_path / "a")
        (a / "torchmetrics/text/eed.py").unlink()
        with pytest.raises(FileNotFoundError, match="torchmetrics/text/eed.py"):
            nc.purge_installed([a])
        assert all((a / rel).exists() for rel in nc.REMOVED_FILES)

    def test_only_roots_holding_a_listed_package_count(self, tmp_path, monkeypatch):
        import site
        import sysconfig
        pure, user = tmp_path / "pure", tmp_path / "user"
        (pure / "pyphen").mkdir(parents=True)
        user.mkdir()
        monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"purelib": str(pure)})
        monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
        monkeypatch.setattr(site, "getusersitepackages", lambda: str(user))
        assert nc.installed_roots() == [pure]
        (user / "torchmetrics").mkdir()
        assert nc.installed_roots() == [pure, user]


def test_the_packaged_build_ships_the_table_the_startup_check_loads():
    src = (ROOT / "scripts/build_installer.py").read_text(encoding="utf-8")
    assert 'LICENCE_TABLE_FILE = "scripts/_nc_stubs.py"' in src
    assert "_stage_agent_bridge(staging)\n    _stage_licence_table(staging)" in src.replace("\r\n", "\n")


class TestEveryInstallRoute:
    """Each route applies the table right after installing the lockfile."""

    def _read(self, rel: str) -> str:
        return (ROOT / rel).read_text(encoding="utf-8")

    def test_windows_applies_it_after_the_lockfile_and_stops_on_failure(self):
        bat = self._read("scripts/windows/install_windows.bat")
        lock = bat.index('-r "%VENV_DIR%\\requirements-notorch.txt"')
        run = bat.index('"%PYTHON_EXE%" scripts\\_nc_stubs.py')
        assert lock < run
        assert "if errorlevel 1 goto :nc_purge_failed" in bat[run:run + 200]
        assert ":nc_purge_failed" in bat[:run]

    def test_macos_applies_it_after_the_lockfile_and_stops_on_failure(self):
        sh = self._read("scripts/macos/install_macos.sh")
        lock = sh.index("requirements-macos-filtered.txt\" 2>&1")
        run = sh.index('scripts/_nc_stubs.py" >> "$LOG_FILE"')
        assert lock < run
        assert "fail " in sh[run:run + 200]

    def test_install_py_applies_it_in_install_python_deps(self):
        tree = ast.parse(self._read("scripts/install.py"))
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "install_python_deps")
        called = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_apply_licence_exclusions" in called

    def test_the_offline_installer_fails_closed(self):
        src = self._read("scripts/install.py")
        assert "required=False" not in src
        assert "skipping the check" not in src

    def test_the_docker_image_applies_it(self):
        df = self._read("apps/trainer_api/Dockerfile")
        assert df.index("pip install") < df.index("RUN python /app/scripts/_nc_stubs.py") < df.index("COPY . /app")

    def test_the_trainer_api_checks_before_it_is_ready(self):
        main = self._read("apps/trainer_api/app/main.py")
        body = main[main.index("def _background_startup"):]
        assert "_enforce_licence_exclusions()" in body[:body.index('_startup_state["ready"] = True')]


def test_the_table_is_not_duplicated_anywhere():
    """It was in build_installer.py, which is why one route missed it."""
    for name in ("scripts/build_installer.py", "scripts/install.py"):
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "NONCOMMERCIAL_STUBS = {" not in src, f"{name} has its own copy again"
        assert "_nc_stubs import" in src, f"{name} does not use the shared table"
