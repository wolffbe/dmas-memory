.PHONY: up down build build-proxy build-dmas build-monitoring logs ps reset set-docker-gid

# Detect OS
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
	SHELL := cmd.exe
	SET_ENV := set
	DOCKER_GID := 0
else
	DETECTED_OS := Linux
	DOCKER_GID := $(shell stat -c '%g' /var/run/docker.sock 2>/dev/null || echo 999)
endif

down:
	@echo Stopping everything on $(DETECTED_OS)...
	cd proxy && docker-compose --env-file ../.env down || echo ""
	cd dmas && docker-compose --env-file ../.env down || echo ""
	cd monitoring && docker-compose --env-file ../.env down || echo ""
	docker network rm dmas-network 2>/dev/null || echo "Network removed"
	@echo Done!

build-proxy:
	@echo Building proxy services...
	cd proxy && docker-compose --env-file ../.env build --no-cache --pull
	@echo Proxy build complete!

build-dmas:
	@echo Building dmas services...
	cd dmas && docker-compose --env-file ../.env build --no-cache --pull
	@echo Dmas build complete!

build-monitoring:
	@echo Building monitoring services...
	cd monitoring && docker-compose --env-file ../.env build --no-cache --pull
	@echo Monitoring build complete!

build: build-proxy build-dmas build-monitoring
	@echo All builds complete!

# ------------------------
# Corrected set-docker-gid
# ------------------------
set-docker-gid:
	@echo Setting DOCKER_GID=$(DOCKER_GID)

ifeq ($(OS),Windows_NT)
	@findstr /V /B "DOCKER_GID=" .env > .env.tmp 2>nul || type .env > .env.tmp
	@move /Y .env.tmp .env >nul
	@echo. >> .env
	@echo DOCKER_GID=0 >> .env
else
	@if grep -q "^DOCKER_GID=" .env 2>/dev/null; then \
		sed -i 's/^DOCKER_GID=.*/DOCKER_GID=$(DOCKER_GID)/' .env; \
	else \
		echo >> .env; \
		echo "DOCKER_GID=$(DOCKER_GID)" >> .env; \
	fi
endif

up: set-docker-gid
	@echo Starting services on $(DETECTED_OS)...
	@echo Docker GID: $(DOCKER_GID)
	docker network rm dmas-network 2>/dev/null || echo ""
	cd proxy && docker-compose --env-file ../.env up -d --build --force-recreate
	cd dmas && docker-compose --env-file ../.env up -d --build --force-recreate
	cd monitoring && docker-compose --env-file ../.env up -d --build --force-recreate
	@echo All services started!

reset:
	@echo "Resetting everything (keeping ollama volume)..."
	cd proxy && docker-compose --env-file ../.env down -v || echo ""
	cd dmas && docker-compose --env-file ../.env down || echo ""
	cd monitoring && docker-compose --env-file ../.env down -v || echo ""
	docker volume rm dmas-long-context-memory_locomo-data 2>/dev/null || echo ""
	docker volume rm dmas-long-context-memory_qdrant-data 2>/dev/null || echo ""
	docker volume rm dmas-long-context-memory_neo4j-data 2>/dev/null || echo ""
	docker volume rm dmas-long-context-memory_neo4j-logs 2>/dev/null || echo ""
	docker volume rm prometheus-data 2>/dev/null || echo ""
	docker network rm dmas-network 2>/dev/null || echo ""
	@echo Reset complete! Ollama volume preserved.

logs:
	docker-compose --env-file .env -f proxy/docker-compose.yml -f dmas/docker-compose.yml -f monitoring/docker-compose.yml logs -f

ps:
	docker ps