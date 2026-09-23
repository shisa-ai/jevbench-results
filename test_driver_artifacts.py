"""Run-directory handling: overwrite, staleness, and comparison scope.

These cover the three defects found in review on 2026-09-23: a partial
`--force` rerun mixing old tiers into a new run, the published-system table
claiming a same-item comparison it did not restrict, and a stop fallback that
could reach another server (that one lives in serve-de1.sh and is checked by
inspection and by `PORT=9999 ./serve-de1.sh --stop`).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import jevbench_public as jp


def write_run(directory: Path, tiers: dict[str, list[str]], manifest_tiers: list[str] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for tier, ids in tiers.items():
        (directory / f"{tier}.jsonl").write_text(
            "".join(json.dumps({"task_id": task_id, "probs": {"a": 1.0}}) + "\n" for task_id in ids),
            encoding="utf-8",
        )
        (directory / f"{tier}.summary.json").write_text("{}\n", encoding="utf-8")
    covered = list(tiers) if manifest_tiers is None else manifest_tiers
    (directory / "manifest.json").write_text(
        json.dumps({"tiers": {tier: {} for tier in covered}}), encoding="utf-8"
    )


class OverwriteTests(unittest.TestCase):
    def test_force_clears_every_previous_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            write_run(out, {"easy": ["e1"], "hard": ["h1"]})
            scratch = out / "scratch"
            (scratch / "hard" / "raw").mkdir(parents=True)
            (scratch / "hard" / "raw" / "response.json").write_text("{}", encoding="utf-8")
            (scratch / "hard.ledger.jsonl").write_text("", encoding="utf-8")
            (scratch / "easy").mkdir(parents=True, exist_ok=True)
            (out / "timing.json").write_text("{}", encoding="utf-8")
            (out / "notes.txt").write_text("not ours", encoding="utf-8")

            cleared = jp.clear_previous_run(out, scratch)

            self.assertEqual(sorted(path.name for path in out.iterdir()), ["notes.txt", "scratch"])
            for name in ("easy.jsonl", "hard.jsonl", "hard.summary.json", "manifest.json", "timing.json"):
                self.assertIn(name, cleared)
            self.assertFalse((scratch / "hard").exists())
            self.assertFalse((scratch / "hard.ledger.jsonl").exists())
            self.assertFalse((scratch / "easy").exists())
            self.assertTrue((out / "notes.txt").exists())


class StalenessTests(unittest.TestCase):
    def manifest(self, out: Path) -> dict:
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    def test_tiers_the_manifest_does_not_cover_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_run(out, {"easy": ["e1"], "hard": ["h1"]}, manifest_tiers=["easy"])
            self.assertEqual(jp.stale_tier_files(out, self.manifest(out)), ["hard.jsonl"])

    def test_a_manifest_without_tiers_makes_no_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_run(out, {"easy": ["e1"]}, manifest_tiers=[])
            self.assertEqual(jp.stale_tier_files(out, self.manifest(out)), [])


class ComparisonScopeTests(unittest.TestCase):
    def test_task_ids_come_from_the_run_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_run(out, {"easy": ["e1", "e2"], "hard": ["h1"]})
            self.assertEqual(jp.run_task_ids(out), {"e1", "e2", "h1"})

    def test_published_rows_are_restricted_to_the_run_items(self) -> None:
        jev = jp.load_jevbench()
        two = sorted(jp.run_task_ids(Path("results/de-1-public")))[:2]
        rows = jp.published_public_items(jev, set(two))
        self.assertTrue(rows)
        for row in rows:
            self.assertLessEqual(sum(n for _correct, n in row["counts"].values()), len(two))


if __name__ == "__main__":
    unittest.main()
