# Build and upload a new version of YASA

```bash
python setup.py sdist
python setup.py sdist bdist_wheel --universal
twine upload dist/*
```
