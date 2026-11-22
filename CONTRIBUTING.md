# Contributing to CATCH

Bug reports and pull requests are welcome on
[GitHub](https://github.com/laltman2/CATCH/issues).

## Setting up

Keep the virtual environment **outside** the repository:

```sh
python3 -m venv ~/python/venvs/catch
~/python/venvs/catch/bin/pip install -e ".[dev]"
~/python/venvs/catch/bin/python -m pytest
```

The repository root is the Python package `CATCH`; the editable install maps
it to that name, so `import CATCH` works from anywhere.  The `fit` and
`train` extras (pylorenzmie, for the fits, the training and the evaluation
harness) are optional; the tests use models with random weights
(`tests/random_models.py`), except a few that fine-tune from the shipped ones.

Other commands: `python -m pytest --cov` (coverage), `ruff check .` (lint),
`pyright` (types; CI pins the version, `pip install pyright==1.1.414`),
`python -m sphinx -W -b html docs docs/_build/html` (the documentation, after
`pip install -r docs/requirements.txt`) and `pre-commit install` (runs ruff
before each commit).  The package is in `src/CATCH`; install it
with `pip install -e ".[dev,train]"` so that the tests import it.

## Style

* Python 3.11 or later. Lines are at most 79 characters.
* Single quotes, including in docstrings. American spelling.
* NumPy-style docstrings with `Parameters`, `Returns`, `Raises` and `Notes`;
  give the default and the units of every parameter.  Lengths are in
  micrometers, except coordinates in an image, which are in pixels.
* `snake_case` for attributes. Annotate public signatures.
* Fix lint findings rather than silencing them.

## Tests

Tests are in `tests/`, use `unittest`, and run under pytest.  Mock hardware
and heavy models (trained weights).  Mark `if __name__ == '__main__':`
guards `# pragma: no cover`.  Test data go under `tests/data/`.

## Media

Media files (PNG, JPG, GIF, AVI, MP4) belong under `docs/`.
