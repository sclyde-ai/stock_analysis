#!/bin/bash

echo "========================================="
echo " Checking status for yfinance_db_saver"
echo "========================================="
if [ -f "yfinance_db_saver/docker-compose.yml" ]; then
    (cd yfinance_db_saver && docker compose ps)
    echo ""
    echo "--- Database Status (yfinance_db_saver) ---"
    (cd yfinance_db_saver && docker compose exec -T db pg_isready -U user)
else
    echo "docker-compose.yml not found in yfinance_db_saver"
fi

echo ""
echo "========================================="
echo " Checking status for yfinance_analyzer"
echo "========================================="
if [ -f "yfinance_analyzer/.devcontainer/docker-compose.yml" ]; then
    (cd yfinance_analyzer/.devcontainer && docker compose -f docker-compose.yml ps)
else
    echo "docker-compose.yml not found in yfinance_analyzer/.devcontainer"
fi
