# Cells Calculator Developer Manual

## Command Reference

#### Setup
Install dev dependencies.
```commandline
pip install -r requirements-dev.txt
```
This installs `mypy` and `pytest` libraries and binaries,
and other dependencies required by them.
This will also automatically install everything from `requirements.txt`.

##### Running
```commandline
python main.py
```

##### Type Checking

```commandline
mypy .
```
To check only specific folder, run
```commandline
mypy model/
```

##### Running tests
```commandline
pytest tests
```