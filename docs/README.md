# Docs

Clean build:

```bash
poetry run make clean
rm -rf source/generated source/enn_trainer
poetry run make html
```

To serve docs:

```bash
poetry run python -m http.server --directory build/html
```

