# echo $1
# dir_name=$1
# echo $dir_name
# mkdir "${dir_name}_db"
# cd "${dir_name}_db"

# touch Dockerfile 
# cat > Dockerfile << EOF
# FROM python:3.9-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY main.py .
# CMD ["python", "main.py"]
# EOF

# touch docker-compose.yml
# cat > docker-compose.yml << EOF
# services:
#   app:
#     container_name: ${dir_name}_app
#     build: .
#     env_file:
#       - ./.env
#     networks:
#       - shared-net
#     volumes:
#       - .:/app
#     depends_on:
#       - db
#     restart: always
#   db:
#     container_name: ${dir_name}_db
#     image: postgres:12
#     env_file:
#       - ./.env
#     ports:
#       - "${DB_PORT}:5432"
#     networks:
#       - shared-net
#     volumes:
#       - postgres_data:/var/lib/postgresql/data/
#       - ./init.sql:/docker-entrypoint-initdb.d/init.sql
#     restart: always
# volumes:
#   postgres_data:
# networks:
#   shared-net:
# EOF

# touch .env
# cat > .env << EOF
# POSTGRES_DB=${dir_name}
# POSTGRES_USER=user
# POSTGRES_PASSWORD=password
# DB_HOST=db
# DB_PORT=
# EOF

# touch requirements.txt
# cat > requirements.txt << EOF
# pandas
# SQLAlchemy
# psycopg2-binary
# python-dotenv
# EOF

# touch "init_${dir_name}.sql"
# cat > "init_${dir_name}.sql" << EOF
# CREATE TABLE IF NOT EXISTS ${dir_name} (
# );
# EOF
# ln "init_${dir_name}.sql" ../initdb.d

# touch main.py
# cat > main.py << EOF
# import os
# import json
# import pandas as pd
# from dotenv import load_dotenv

# load_dotenv()
# DB_NAME = os.getenv('POSTGRES_DB')
# DB_USER = os.getenv('POSTGRES_USER')
# DB_PASSWORD = os.getenv('POSTGRES_PASSWORD')
# DB_HOST = os.getenv('DB_HOST') 
# DB_PORT = os.getenv('DB_PORT')
# DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# EOF