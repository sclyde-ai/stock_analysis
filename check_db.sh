if [ $# -eq 0 ]; then
    docker compose exec -T db psql -U user -l
else
    docker compose exec db psql -U user -d $1
fi