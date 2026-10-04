# Production image. Listens on $PORT (set by hosts like Render), defaulting to 7860.
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
CMD exec gunicorn --workers 1 --threads 8 --bind "0.0.0.0:${PORT:-7860}" webapp.app:app
