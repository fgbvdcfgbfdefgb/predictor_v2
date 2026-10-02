FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir .
RUN useradd --create-home predictor && mkdir -p /app/artifacts && chown -R predictor:predictor /app
USER predictor
CMD ["predictor-live", "--config", "config/default.yaml"]
