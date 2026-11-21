import imageio.v2 as imageio
import pytest

from submarine.baselines import GapFollower, Oracle
from submarine.evaluation import evaluate
from submarine.render.replay import record_replay


@pytest.mark.parametrize("ext", ["mp4", "gif"])
def test_replay_writes_video_and_matches_evaluation(tmp_path, ext):
    out = tmp_path / f"r.{ext}"
    res = record_replay([Oracle(), GapFollower()], None, 1_000_000, 0.0, out, speed=8.0, hold_seconds=0.5)
    assert out.exists() and out.stat().st_size > 10_000
    frame = imageio.get_reader(out).get_data(0)
    assert frame.shape[0] % 2 == 0 and frame.shape[1] % 2 == 0

    # same cave + noise as evaluate(), so outcomes and returns agree
    ev = evaluate(Oracle(), None, [0.0], n_caves=1)[0]
    assert res[0]["outcome"] == ev.outcome
    assert res[0]["return"] == pytest.approx(ev.episode_return)
