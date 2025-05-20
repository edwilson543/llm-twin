# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/app/.cache/uv

# Create non-root user early to avoid duplicate layers
RUN groupadd app && \
    useradd app -g app

# Set working directory
WORKDIR /app

# Copy only files needed for dependency installation first
COPY --chown=app:app uv.lock pyproject.toml ./

# Install dependencies using lockfile (make sure package dependencies are installed)
RUN --mount=type=cache,target=/app/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy application code and entrypoint script
COPY --chown=app:app src ./src

# Install the project itself to ensure all dependencies are properly resolved
RUN --mount=type=cache,target=/app/.cache/uv \
    uv pip install -e .

# Switch to non-root user
USER app
