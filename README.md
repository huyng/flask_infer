# Install

```
pip install flask_infer
```

# Usage

```
python -m flask_infer /path/to/saved_model
```

# Calling the webserver

```
curl -d '{"inputs": {"examples": [[1,2]]}, "outputs": ["probs"]}' -H "Content-Type: application/json" -X POST http://localhost:5000/api
```
