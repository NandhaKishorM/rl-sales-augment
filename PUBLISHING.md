# Publishing to PyPI

The distributions are already built in `dist/` and pass `twine check`. The trained model is bundled
inside them, so users get everything from `pip install` with no extra downloads.

## 0. One-time setup
- Create accounts on [PyPI](https://pypi.org/account/register/) and (for testing)
  [TestPyPI](https://test.pypi.org/account/register/).
- Create an API token on each (Account settings → API tokens).
- Confirm the name `rl-sales-augment` is free on PyPI (https://pypi.org/project/rl-sales-augment/).
  If taken, change `name` in `pyproject.toml` (the import name `rl_sales_augment` can stay).

## 1. Build (already done; rerun after any change)
```bash
python -m build            # writes dist/*.whl and dist/*.tar.gz
twine check dist/*         # must say PASSED
```

## 2. Test on TestPyPI first
```bash
twine upload -r testpypi dist/*
# then, in a clean venv, install from TestPyPI but pull real deps from PyPI:
pip install -i https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ rl-sales-augment
python -c "import rl_sales_augment as r; print(r.__version__, r.MODEL_PATH)"
```

## 3. Publish to PyPI
```bash
twine upload dist/*        # paste your PyPI API token (username = __token__)
```
Then `pip install rl-sales-augment` works for everyone.

## Notes
- **Model size:** the wheel is ~16 MB (the 37 MB model compresses in the zip), well under PyPI's
  100 MB per-file limit. No PyPI size-increase request needed.
- **Python support:** the wheel is `py3-none-any` (pure Python), so it works on every CPython that
  PyTorch supports; `requires-python = ">=3.9"`. `numpy`/`torch` wheels are resolved per version by pip.
- **New releases:** bump `version` in `pyproject.toml` (and `__version__` in `__init__.py`), rebuild,
  re-upload. PyPI does not allow re-uploading the same version.
- **Trusted publishing (optional):** configure a GitHub Actions OIDC publisher on PyPI to upload on
  tag pushes without storing a token.
