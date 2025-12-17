# Makefile for RL4VLA Docker operations
# Usage: make <target>

.PHONY: help build up down restart shell logs logs-follow clean rebuild ps

# Docker compose file location
COMPOSE_FILE := docker/docker-compose.yml
SERVICE_NAME := rl4vla
CONTAINER_NAME := rl4vla-container

# Default target
help:
	@echo "RL4VLA Docker Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make build      - Build the Docker image"
	@echo "  make up         - Start the container"
	@echo "  make down       - Stop the container"
	@echo "  make restart    - Restart the container"
	@echo "  make shell      - Open interactive shell in the container"
	@echo "  make logs       - Show container logs"
	@echo "  make logs-follow - Follow container logs"
	@echo "  make rebuild    - Rebuild image and restart container"
	@echo "  make clean      - Stop and remove container"
	@echo "  make ps         - Show running containers"

# Build the Docker image
build:
	@echo "Building RL4VLA Docker image..."
	docker compose -f $(COMPOSE_FILE) build

# Start the container
up:
	@echo "Starting RL4VLA container..."
	docker compose -f $(COMPOSE_FILE) up -d
	@echo "Container started. Use 'make shell' to enter the container."

# Stop the container
down:
	@echo "Stopping RL4VLA container..."
	docker compose -f $(COMPOSE_FILE) down

# Restart the container
restart: down up
	@echo "Container restarted."

# Open interactive shell in the container
shell:
	@echo "Opening shell in $(CONTAINER_NAME)..."
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Show container logs
logs:
	docker compose -f $(COMPOSE_FILE) logs

# Follow container logs
logs-follow:
	docker compose -f $(COMPOSE_FILE) logs -f

# Rebuild image and restart container
rebuild: down build up
	@echo "Image rebuilt and container restarted."

# Stop and remove container (keeps image)
clean: down
	@echo "Container stopped and removed."

# Show running containers
ps:
	docker compose -f $(COMPOSE_FILE) ps

