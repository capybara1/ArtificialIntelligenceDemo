# Preconditions

[venv](https://docs.python.org/3/library/venv.html) is installed

# Setup

```
virtualenv --system-site-packages -p python3 ./venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
pre-commit install
deactivate
```

# Usage (Windows)

## Jupyter Notebook

```
.\venv\Scripts\activate
python -m notebook
deactivate
```

## UI for MNIST Handriting

```
.\venv\Scripts\activate
python .\mnist_ui.py
deactivate
```
