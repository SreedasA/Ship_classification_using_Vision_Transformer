## Contributing

Contributions are welcome (bug fixes, improvements, docs).

### Development setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
```

### Lint

```bash
ruff check .
```

### Pull requests

- Keep changes focused and clearly described.
- Update `README.md` if behavior/usage changes.
- Avoid committing datasets, large artifacts, or model checkpoints.

