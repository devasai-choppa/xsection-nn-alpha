FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        gifsicle \
        libcairo2 \
        libffi-dev \
        libgdk-pixbuf-2.0-0 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        pkg-config \
        shared-mime-info \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY app ./app

RUN pip install --no-cache-dir -e ".[dev]"

CMD ["pytest"]
