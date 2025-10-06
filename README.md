# perplexity-agents

A Python project workspace for "perplexity-agents". This repository currently contains project scaffolding and configuration files.

## What this repo contains

- Project source (add your code under a suitable package or `src/` folder)
- This `README.md` with setup and run instructions
- `.gitignore` tailored for Python projects
- `.env` example file (sensitive values are ignored by `.gitignore` — fill in your secrets locally)
- `LICENSE` (MIT)

## Quick start

1. Install Python 3.10+ (3.11 recommended).
2. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies (if you have a `requirements.txt`):

```bash
pip install -r requirements.txt
```

4. Populate the `.env` file with required environment variables. See the `.env` file in the repo for keys to set.

5. Run your app (adjust the command below to how your project starts):

```bash
# Example for a CLI or script
python -m your_package.main

# Example for Flask
# export FLASK_APP=your_package.app
# flask run
```

## Environment variables

This project uses a `.env` file to hold secrets and configuration for local development. The `.env` file is in `.gitignore` to avoid accidental commits of secrets. Typical entries you may need:

- `SECRET_KEY` — application secret key
- `DATABASE_URL` — connection string for the database
- `OPENAI_API_KEY` — API key for external LLM services

If you use a deployment pipeline, configure these variables in your CI/CD or hosting environment instead of committing them.

## Tests

If you add tests, run them with pytest:

```bash
pytest
```

## Contributing

1. Fork the repo
2. Create a topic branch
3. Add tests for new behavior
4. Open a pull request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
