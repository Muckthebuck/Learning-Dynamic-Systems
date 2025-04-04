#!/bin/bash

set -euo pipefail

# Configuration
CONTAINER_NAME="redis-pubsub"
REDIS_PORT=6379
IMAGE_NAME="redis:latest"

# Start Redis container
start_redis() {
  echo "Starting Redis container..."

  docker pull "$IMAGE_NAME"

  docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true

  docker run -d \
    --name "$CONTAINER_NAME" \
    -p "$REDIS_PORT:6379" \
    "$IMAGE_NAME" > /dev/null

  echo "Waiting for Redis to be ready..."

  until docker exec "$CONTAINER_NAME" redis-cli PING | grep -q PONG; do
    sleep 0.5
  done

  echo "Redis is running at redis://localhost:$REDIS_PORT"
}

# Stop Redis container
stop_redis() {
  echo "Stopping Redis container..."

  docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || echo "Container is not running."
  docker rm "$CONTAINER_NAME" > /dev/null 2>&1 || echo "Container already removed."

  echo "Redis container stopped and removed."
}

# Usage message
usage() {
  echo "Usage: $0 --start | --stop"
  exit 1
}

# Main entry point
case "${1:-}" in
  --start)
    start_redis
    ;;
  --stop)
    stop_redis
    ;;
  *)
    usage
    ;;
esac
