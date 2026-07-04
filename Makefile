# Aortica — developer & deployment convenience targets.
#
# Full-stack Docker environment (see docs/deployment/DOCKER_QUICKSTART.md):
#   make dev     Start the full dev stack (API + frontend + docs + edge)
#   make prod    Build production images and run behind nginx TLS reverse proxy
#   make down    Stop and remove the stack
#   make logs    Follow logs from all services
#   make ps      Show running services
#   make certs   Generate a self-signed TLS certificate for local prod testing
#   make clean   Stop the stack and remove volumes

COMPOSE       := docker compose
FULL          := -f docker-compose.full.yml
PROD          := -f docker-compose.full.yml -f docker-compose.prod.yml

.DEFAULT_GOAL := help
.PHONY: help dev prod down logs ps certs clean config

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

dev: ## Start the full development stack
	$(COMPOSE) $(FULL) --profile dev up --build

prod: certs ## Build production images and run behind nginx TLS proxy
	$(COMPOSE) $(PROD) --profile prod up --build

down: ## Stop and remove the stack (keeps volumes)
	$(COMPOSE) $(PROD) --profile dev --profile prod down

logs: ## Follow logs from all services
	$(COMPOSE) $(FULL) logs -f

ps: ## Show running services
	$(COMPOSE) $(FULL) ps

certs: ## Generate a self-signed TLS certificate for local prod testing
	@./deploy/nginx/gen-cert.sh

config: ## Validate the compose configuration
	$(COMPOSE) $(PROD) --profile dev --profile prod config >/dev/null && echo "compose config OK"

clean: ## Stop the stack and remove named volumes
	$(COMPOSE) $(PROD) --profile dev --profile prod down -v
