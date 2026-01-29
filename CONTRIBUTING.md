# Contributing

## Development install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run tests

```bash
pytest -q
```

## Build package

```bash
python -m build
```
