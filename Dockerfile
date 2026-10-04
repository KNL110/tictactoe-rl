# Image for the Hugging Face Space (Docker SDK). HF runs containers as uid 1000
# and routes traffic to app_port (7860, set in README.md's front matter).
FROM python:3.13-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user tictactoe ./tictactoe
COPY --chown=user webapp ./webapp
COPY --chown=user saved_models ./saved_models

EXPOSE 7860
# One worker on purpose: games live in process memory. Threads handle concurrent visitors.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--bind", "0.0.0.0:7860", "webapp.app:app"]
