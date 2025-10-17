# gor-toolkit (Private)

Measurement toolkit for two-layer meta-documents (Geometry of Reporting Series).
This repository is **private** and distributed on an access-request basis.

## Quickstart
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -v
python run.py examples/mwe_a_2025-109.yaml
```

## Design
- Union-K encoding (TF-IDF, ja/en)
- Preflight (E100–E140): preregistration / complete log / reversible-only postprocess
- Q1–Q4 metrics, Q5 value-neutrality (in tests), deterministic runs

## Structure
- `src/`: implementation
- `data/schema.json`: YAML schema (single source of truth)
- `examples/`: MWEs (2025-109, 2025-113, 2025-116)
- `tests/`: pytest suite
- `ci/`: private CI workflow

## License
MIT (code). Please see license and internal terms-of-use enclosed with the distribution package.

# test commit from develop
