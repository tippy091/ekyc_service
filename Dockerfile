FROM python:3.11-slim-buster AS builder

WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install poetry
RUN pip install gunicorn

COPY . /app

# Install project dependencies using Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# Expose the port Gunicorn will listen on
EXPOSE 8000

# Command to run the Gunicorn WSGI server.
CMD ["gunicorn", "-c", "gunicorn_config.py", "wsgi:application"]
