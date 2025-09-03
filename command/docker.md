# grammmer 
## docker compose
-f : docker-compose.ymlの名前を指定する
### docker compose up
build + run
--build : imageを構築する（編集後に使う）
-d/--detached : container起動後にterminalが解放されて利用できる
### docker compose down 
-v/--volumes : volumeも削除する（database初期化）
### docker compose build
### docker compose run
### docker compose logs

## docker system
### docker system prune
使われていないdocker containerを削除する
-a : all docker images
--volumes : volumeも削除する

# useful command 
docker compose exec -T db psql -U user -l

# docker-compose.yml
定義、識別
1. container_name
2. image
3. build
設定
4. env_file
5. environment
通信、外部接続
6. ports
7. networks
storage
8. volumes
実行制御
9. depends_on
10. restart
