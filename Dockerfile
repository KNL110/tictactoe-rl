# Production image. Listens on $PORT (set by hosts like Render), defaulting to 7860.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv

RUN useradd -m -u 1000 user
USER user
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY --chown=user tictactoe ./tictactoe
COPY --chown=user webapp ./webapp
COPY --chown=user saved_models ./saved_models

EXPOSE 7860
# One worker on purpose: games live in process memory. Threads handle concurrent visitors.
CMD exec .venv/bin/gunicorn --workers 1 --threads 8 --bind "0.0.0.0:${PORT:-7860}" webapp.app:app
