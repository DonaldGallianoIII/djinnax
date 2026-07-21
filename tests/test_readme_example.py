"""The README usage snippet, executed verbatim — if the public example
drifts from the API, this fails (CPU-runnable; CI core)."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_usage_snippet_runs():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert blocks, "README has no python usage block"
    ns = {}
    exec(compile(blocks[0], str(README), "exec"), ns)  # noqa: S102
    state, reward = ns["state"], ns["reward"]
    assert state.board.shape == (8192, 4, 4)
    assert reward.shape == (8192,)


def test_package_exports():
    import djinnax

    for name in djinnax.__all__:
        if name == "run_megakernel_rng":
            continue  # lazy: touching it imports pallas, covered on GPU
        assert getattr(djinnax, name) is not None, name
