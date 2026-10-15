# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Segmen-Pixel and Seg-Studio contributors
"""Nothing in the tree names somebody else's data.

The leaks this catches all got in the same way: a fixture copied off a live
project, a usage example written against the dataset that happened to be open,
a run id pasted into a comment beside its measurement, a playbook paragraph
that counted one installation's projects. Every one of them passed the tests,
the linters and the licence audit, because none of those ask the question.

The structural pass runs here so it runs in CI. The installation pass -- every
project id, project name and image name on the machine doing the release --
runs from the release script, where the data actually is.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import check_no_customer_data as guard  # noqa: E402


def test_the_tree_names_nobody():
    findings = guard.structural(guard._tracked_files())
    assert not findings, "\n".join(
        f"{rel}:{n} [{kind}] {what}" for rel, n, kind, what, _ in sorted(set(findings)))


class TestWhatItCatches:
    """The rules, on lines written for the purpose."""

    @staticmethod
    def _scan(tmp_path, name, text):
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        # structural() reports paths relative to the repo root, so point the
        # module at the temporary tree for the duration.
        old, guard.ROOT = guard.ROOT, tmp_path
        try:
            return guard.structural([p])
        finally:
            guard.ROOT = old

    def test_a_project_id_in_shipped_code(self, tmp_path):
        got = self._scan(tmp_path, "ship.py", "# measured on run 5e81b0c47d2a last week\n")
        assert [f[2] for f in got] == ["project id"]

    def test_an_id_a_person_typed_is_not_one(self, tmp_path):
        for fake in ("aaaaaaaa0001", "0123456789ab", "deadbeef0001"):
            assert not self._scan(tmp_path, "ship.py", f'pid = "{fake}"\n'), fake

    def test_a_job_number(self, tmp_path):
        got = self._scan(tmp_path, "ship.py", 'name = "250102_something"\n')
        assert [f[2] for f in got] == ["job number"]

    def test_an_address_on_somebody_lan(self, tmp_path):
        got = self._scan(tmp_path, "doc.md", "point it at http://192.168.1.50:8080\n")
        assert [f[2] for f in got] == ["private address"]

    def test_the_documentation_ranges_are_fine(self, tmp_path):
        assert not self._scan(tmp_path, "doc.md", "point it at http://198.51.100.10:8080\n")
        assert not self._scan(tmp_path, "doc.md", "bind 0.0.0.0, or 127.0.0.1 for local only\n")

    def test_a_version_pin_is_not_an_address(self, tmp_path):
        assert not self._scan(tmp_path, "requirements.txt", "nvidia-curand==10.4.0.35\n")

    def test_a_home_directory_with_a_name_in_it(self, tmp_path):
        got = self._scan(tmp_path, "doc.md", r"unzip it to C:\Users\jdoe\Desktop" + "\n")
        assert [f[2] for f in got] == ["home directory"]
        assert not self._scan(tmp_path, "doc.md", r"unzip it to C:\Users\NAME\Desktop" + "\n")

    def test_a_sentence_only_its_author_can_read(self, tmp_path):
        got = self._scan(tmp_path, "doc.md", "Surveyed on one server, eight of nine projects\n")
        assert [f[2] for f in got] == ["private voice"]

    def test_a_count_of_one_installation_s_projects(self, tmp_path):
        for said in ("Measured on the four gasket projects, it held.\n",
                     "On the 3 sample-plate projects the floor was kept.\n"):
            got = self._scan(tmp_path, "doc.md", said)
            assert [f[2] for f in got] == ["private voice"], said
        assert not self._scan(tmp_path, "doc.md", "Import the tutorial project to try it.\n")

    def test_a_test_may_talk_about_ids_and_addresses(self, tmp_path):
        """`is_nonlocal_bind("192.168.1.9")` is the assertion, not a leak."""
        text = 'assert is_nonlocal_bind("192.168.1.9") and pid == "5e81b0c47d2a"\n'
        assert not self._scan(tmp_path, "test_bind.py", text)

    def test_a_line_can_say_why_it_is_fine(self, tmp_path):
        text = "host = '192.168.1.50'  # allow-customer-data: the example in the manual\n"
        assert not self._scan(tmp_path, "doc.md", text)


def test_the_installation_pass_needs_no_list_of_customers(tmp_path):
    """It reads the projects on the machine instead, so the repository never
    holds the names it is protecting."""
    src = (ROOT / "scripts" / "ci" / "check_no_customer_data.py").read_text(encoding="utf-8")
    projects = tmp_path / "projects"
    (projects / "0123456789ab").mkdir(parents=True)
    (projects / "0123456789ab" / "project.json").write_text(
        '{"name": "250102_a customer job"}', encoding="utf-8")
    (projects / "0123456789ab" / "index.json").write_text(
        '{"items": [{"id": "DSC_0001_left"}, {"id": "img001"}]}', encoding="utf-8")
    words = guard._installation_words(projects)
    assert words == {"0123456789ab": "project id", "250102_a customer job": "project name",
                     "DSC_0001_left": "image name"}, words
    assert "250102" not in src.replace("(1[6-9]|2[0-9])", ""), "no customer string in the checker"


def test_the_e2e_suite_s_own_projects_are_not_collected(tmp_path):
    """The Playwright suite makes its projects under zz-e2e- names, and its
    setup, its specs and CONTRIBUTING.md name them; an installation that has
    run the suite holds them. Their names are exempt and nothing else is:
    their ids and images are collected, and a name that merely contains the
    prefix is a name like any other."""
    projects = tmp_path / "projects"
    for pid, name in (("0123456789ab", "zz-e2e-seed-9"), ("123456789abc", "250102 zz-e2e-seed-9 copy")):
        (projects / pid).mkdir(parents=True)
        (projects / pid / "project.json").write_text(json.dumps({"name": name}), encoding="utf-8")
    (projects / "0123456789ab" / "index.json").write_text(
        '{"items": [{"id": "DSC_0001_left"}]}', encoding="utf-8")
    words = guard._installation_words(projects)
    assert words == {"0123456789ab": "project id", "123456789abc": "project id",
                     "250102 zz-e2e-seed-9 copy": "project name", "DSC_0001_left": "image name"}, words

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "global-setup.ts").write_text('const SEED = "zz-e2e-seed-9";\n', encoding="utf-8")
    args = ["--pass", "local", "--root", str(tree), "--projects-dir", str(projects)]
    assert guard.main(args) == 0
    (tree / "notes.md").write_text("see DSC_0001_left\n", encoding="utf-8")
    assert guard.main(args) == 1


def test_this_file_names_nothing_on_the_installation_beside_it():
    """The structural patterns skip this file, whose examples are what they
    look for; its ids and names must still be made up. On a checkout with
    projects beside it, none of them may be one of those projects' own."""
    words = guard._installation_words(ROOT / "projects")
    if not words:
        pytest.skip("no projects beside this checkout")
    text = Path(__file__).read_text(encoding="utf-8")
    kinds = sorted({kind for word, kind in words.items() if word in text})
    assert not kinds, f"this file carries a real {' and a real '.join(kinds)}"


class TestNothingCheckedIsNotAPass:
    """A gate with nothing to compare against must not report a clean tree."""

    def test_the_installation_pass_fails_without_projects(self, tmp_path):
        assert guard.main(["--pass", "local", "--projects-dir", str(tmp_path / "missing")]) == 2
        (tmp_path / "empty").mkdir()
        assert guard.main(["--pass", "both", "--projects-dir", str(tmp_path / "empty")]) == 2

    def test_an_export_is_read_as_it_ships(self, tmp_path):
        """A tracked build/ ships; only a walked working tree leaves it out."""
        tree = tmp_path / "tree"
        (tree / "build").mkdir(parents=True)
        (tree / "build" / "start.py").write_text("# measured on run 5e81b0c47d2a\n", encoding="utf-8")
        assert guard.main(["--root", str(tree)]) == 1
        (tree / "build" / "start.py").write_text("# nothing to see\n", encoding="utf-8")
        assert guard.main(["--root", str(tree)]) == 0

    def test_files_without_a_known_suffix_are_read(self, tmp_path):
        """A Dockerfile, a dot-file or nginx.conf ships like any source file."""
        for n, name in enumerate(("Dockerfile", "Dockerfile.gpu", ".env.example",
                                  ".gitignore", ".gitattributes", "nginx.conf", "pre-commit")):
            tree = tmp_path / f"tree{n}"
            tree.mkdir()
            (tree / name).write_text("# measured on run 5e81b0c47d2a\n", encoding="utf-8")
            assert guard.main(["--root", str(tree)]) == 1, name

    def test_an_empty_export_is_not_a_clean_one(self, tmp_path):
        (tmp_path / "tree").mkdir()
        assert guard.main(["--root", str(tmp_path / "tree")]) == 2

    def test_the_installation_pass_reads_the_checker_and_its_test(self, tmp_path):
        projects = tmp_path / "projects"
        (projects / "5e81b0c47d2a").mkdir(parents=True)
        (projects / "5e81b0c47d2a" / "project.json").write_text('{"name": "sample"}', encoding="utf-8")
        tree = tmp_path / "tree"
        (tree / "tests").mkdir(parents=True)
        (tree / "README.md").write_text("nothing to see\n", encoding="utf-8")
        (tree / "tests" / "test_no_customer_data.py").write_text('pid = "5e81b0c47d2a"\n', encoding="utf-8")
        args = ["--root", str(tree), "--projects-dir", str(projects)]
        assert guard.main(["--pass", "structural", *args]) == 0
        assert guard.main(["--pass", "local", *args]) == 1

    def test_the_release_gate_needs_an_installation_or_an_explicit_skip(self, tmp_path, monkeypatch):
        spec = importlib.util.spec_from_file_location(
            "make_release_artifacts", ROOT / "scripts" / "release" / "make_release_artifacts.py")
        release = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(release)
        dist = tmp_path / "dist"
        monkeypatch.setattr(sys, "argv", [
            "make_release_artifacts.py", "--version", "0.0.0", "--skip-source",
            "--dist", str(dist), "--projects-dir", str(tmp_path / "missing")])
        assert release.main() == 1
        assert not dist.exists(), "nothing is written when the gate cannot run"
