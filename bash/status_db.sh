#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Check if the db service is running
if [ -z "$(docker compose -f yfinance_db_saver/docker-compose.yml ps -q db)" ] || [ -z "$(docker ps -q --no-trunc | grep $(docker compose -f yfinance_db_saver/docker-compose.yml ps -q db))" ]; then
  echo "Database container is not running. Please start it with 'docker compose -f yfinance_db_saver/docker-compose.yml up -d'"
  exit 1
fi

docker compose -f yfinance_db_saver/docker-compose.yml exec -T db psql -U "$POSTGRES_USER" -d "postgres" -c "\l"
docker compose -f yfinance_db_saver/docker-compose.yml exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt"
docker compose -f yfinance_db_saver/docker-compose.yml exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT * FROM aapl_1d LIMIT 10;" 
