"""scripts.pick_showcase (caves to film) and scripts.make_preview (README previews), on synthetic inputs."""

import csv
import sys

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from scripts import make_preview, pick_showcase

FIELDS = ["policy", "layout", "cave_seed", "outcome", "time"]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for policy, layout, seed, outcome, time in rows:
            w.writerow(dict(policy=policy, layout=layout, cave_seed=seed, outcome=outcome, time=time))
    return str(path)


def run_pick(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["pick_showcase", *argv])
    pick_showcase.main()
    out, err = capsys.readouterr()
    return out.split(), err


@pytest.fixture
def evals(tmp_path):
    models = write_csv(tmp_path / "models.csv", [
        ("mem", "rings", 1, "success", 80.0), ("ctrl", "rings", 1, "timeout", 150.0),   # memory wins
        ("mem", "rings", 2, "success", 60.0), ("ctrl", "rings", 2, "timeout", 150.0),   # wins, faster
        ("mem", "rings", 3, "success", 50.0), ("ctrl", "rings", 3, "success", 55.0),    # both succeed
        ("mem", "rings", 4, "success", 40.0), ("ctrl", "rings", 4, "timeout", 150.0),   # oracle failed here
        ("mem", "hub", 7, "timeout", 150.0), ("ctrl", "hub", 7, "success", 90.0),       # control wins
        ("mem", "hub", 8, "success", 70.0),                                             # control not evaluated
    ])
    oracle = write_csv(tmp_path / "oracle.csv", [
        ("nav-oracle", "rings", s, "success" if s != 4 else "timeout", 50.0) for s in (1, 2, 3, 4)
    ] + [("nav-oracle", "hub", s, "success", 50.0) for s in (7, 8)])
    return models, oracle


def test_pick_showcase_finds_winner_only_caves_fastest_first(monkeypatch, capsys, evals):
    models, oracle = evals
    picks, err = run_pick(monkeypatch, capsys, "--csv", models, "--winner", "mem", "--losers", "ctrl",
                          "--reference-csv", oracle, "--layouts", "rings", "--per-layout", "5")
    assert picks == ["rings:2", "rings:1"]             # cave 3: both succeed; cave 4: the oracle failed
    assert "2 such caves" in err


def test_pick_showcase_without_reference_keeps_unverified_caves(monkeypatch, capsys, evals):
    models, _ = evals
    picks, _ = run_pick(monkeypatch, capsys, "--csv", models, "--winner", "mem", "--losers", "ctrl",
                        "--layouts", "rings")
    assert picks == ["rings:4"]                         # fastest winning dive, oracle not required


def test_pick_showcase_reports_layouts_with_no_candidate(monkeypatch, capsys, evals):
    models, oracle = evals
    picks, err = run_pick(monkeypatch, capsys, "--csv", models, "--winner", "mem", "--losers", "ctrl",
                          "--reference-csv", oracle)       # all layouts: hub has no clean win
    assert picks == ["rings:2"]                         # hub 7: control won; hub 8: control never evaluated
    assert "hub: no cave where mem succeeds and ctrl fail" in err


def test_pick_showcase_needs_every_loser_to_fail(monkeypatch, capsys, tmp_path):
    rows = [("mem", "maze", 1, "success", 60.0), ("a", "maze", 1, "timeout", 150.0), ("b", "maze", 1, "success", 70.0),
            ("mem", "maze", 2, "success", 65.0), ("a", "maze", 2, "collision", 30.0), ("b", "maze", 2, "timeout", 150.0)]
    f = write_csv(tmp_path / "e.csv", rows)
    picks, _ = run_pick(monkeypatch, capsys, "--csv", f, "--winner", "mem", "--losers", "a", "b")
    assert picks == ["maze:2"]


# ---- make_preview ---------------------------------------------------------------------------------------------

W, H, FPS = 64, 48, 20


def make_clip(path, seconds, color):
    """A tiny .mp4 whose brightness changes every frame (so the encoder can't merge frames)."""
    n = int(seconds * FPS)
    with imageio.get_writer(str(path), fps=FPS, macro_block_size=16) as w:
        for i in range(n):
            frame = np.zeros((H, W, 3), np.uint8)
            frame[:] = np.array(color) * (0.4 + 0.6 * i / max(n - 1, 1))
            w.append_data(frame)
    return str(path)


def run_preview(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["make_preview", *argv, "--effort", "0", "--quality", "90"])
    make_preview.main()
    capsys.readouterr()


def duration_ms(path):
    im = Image.open(path)
    total = 0
    for i in range(im.n_frames):
        im.seek(i)
        im.load()                                        # Pillow fills in the frame's timing on load
        total += im.info["duration"]
    return im.size, total


def test_single_clip_excerpt_has_requested_size_and_length(monkeypatch, capsys, tmp_path):
    clip = make_clip(tmp_path / "a.mp4", 3, (255, 0, 0))
    out = tmp_path / "media" / "a.webp"                 # parent folder is created
    run_preview(monkeypatch, capsys, clip, "--start", "1", "--duration", "1.5", "--width", "32", "--out", str(out))
    size, ms = duration_ms(out)
    assert size == (32, 24)
    assert abs(ms - 1500) <= 100


def test_side_by_side_holds_the_shorter_clip_on_its_last_frame(monkeypatch, capsys, tmp_path):
    short = make_clip(tmp_path / "short.mp4", 1, (255, 0, 0))     # red, ends after 1 s
    long = make_clip(tmp_path / "long.mp4", 2, (0, 0, 255))       # blue
    out = tmp_path / "pair.webp"
    run_preview(monkeypatch, capsys, short, long, "--duration", "2", "--width", "64", "--out", str(out))
    size, ms = duration_ms(out)
    assert size == (2 * W, H)
    assert abs(ms - 2000) <= 100                         # runs until the longer clip's end, not the shorter's
    im = Image.open(out)
    im.seek(im.n_frames - 1)
    last = np.asarray(im.convert("RGB"), dtype=float)
    left, right = last[:, :W].mean((0, 1)), last[:, W:].mean((0, 1))
    assert left[0] > 150 and left[2] < 80                # left panel: still the red clip's (bright) last frame
    assert right[2] > 150 and right[0] < 80


def test_vertical_stack_and_speed_up(monkeypatch, capsys, tmp_path):
    a = make_clip(tmp_path / "a.mp4", 2, (0, 255, 0))
    b = make_clip(tmp_path / "b.mp4", 2, (255, 255, 0))
    out = tmp_path / "v.webp"
    run_preview(monkeypatch, capsys, a, b, "--stack", "v", "--duration", "2", "--speed", "2", "--width", "64",
                "--out", str(out))
    size, ms = duration_ms(out)
    assert size == (W, 2 * H)
    assert abs(ms - 1000) <= 100                         # 2 s of video at 2x speed


def test_mp4_grid_with_labels(monkeypatch, capsys, tmp_path):
    clips = [make_clip(tmp_path / f"{i}.mp4", 1 + i * 0.5, c)
             for i, c in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)])]
    out = tmp_path / "grid.mp4"
    monkeypatch.setattr(sys, "argv", ["make_preview", *clips, "--grid", "2", "--labels", "a", "b", "c", "d",
                                      "--label-height", "16", "--width", "64", "--fps", "10", "--duration", "2",
                                      "--out", str(out)])
    make_preview.main()
    capsys.readouterr()
    r = imageio.get_reader(str(out))
    frames = [f for f in r]
    assert frames[0].shape == (2 * (H + 16), 2 * W, 3)      # 2x2 panels, each with a 16 px label bar on top
    assert abs(len(frames) - 20) <= 1                        # 2 s at 10 fps, though the first clip lasts 1 s
    top = frames[0][:16, :W].astype(float)
    assert top[:3, :3].mean() < 40                           # dark label bar (corner) ...
    assert top.max() > 200                                   # ... with light text in it
    last = frames[-1].astype(float)
    assert last[16:16 + H, :W, 0].mean() > 150               # top-left panel: red clip held on its last frame


def test_make_preview_rejects_bad_labels_and_formats(monkeypatch, tmp_path):
    clip = make_clip(tmp_path / "a.mp4", 1, (255, 0, 0))
    for extra in (["--labels", "a", "b", "--out", str(tmp_path / "x.mp4")], ["--out", str(tmp_path / "x.gif")]):
        monkeypatch.setattr(sys, "argv", ["make_preview", clip, "--duration", "1", *extra])
        with pytest.raises(SystemExit):
            make_preview.main()
