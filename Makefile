# Makefile

IMAGE_FASTAPI   := assignment5-fastapi_service
IMAGE_STREAMLIT := assignment5-streamlit_dashboard
COMPOSE_FILE    := docker-compose.yml

.PHONY: build run logs clean

build:
	docker-compose -f $(COMPOSE_FILE) build

run:
	docker-compose -f $(COMPOSE_FILE) up -d

logs:
	docker-compose -f $(COMPOSE_FILE) logs -f

clean:
	docker-compose -f $(COMPOSE_FILE) down --remove-orphans --volumes
