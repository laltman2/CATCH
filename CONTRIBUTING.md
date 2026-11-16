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
it to that name, so `import CATCH` works from anywhere.  The `yolo` extra
(YOLOv5 6.0.7 or older, for the Localizer) and the `train` extra (pylorenzmie,
for generating training data) are optional; the tests mock YOLOv5 and the
trained weights.

Other commands: `python -m pytest --cov` (coverage), `ruff check .` (lint),
`pre-commit install` (runs ruff before each commit).

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
and heavy models (YOLOv5, trained weights).  Mark `if __name__ == '__main__':`
guards `# pragma: no cover`.  Test data go under `tests/data/`.

## Media

Media files (PNG, JPG, GIF, AVI, MP4) belong under `docs/`.
