CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    category TEXT,
    source TEXT,
    author TEXT,
    title TEXT,
    description TEXT,
    url TEXT UNIQUE,
    published_at TIMESTAMP,
    content TEXT
);

CREATE TABLE IF NOT EXISTS logs (
    id SERIAL PRIMARY KEY,
    fetched_at TIMESTAMP,
    category TEXT,
    num_articles INTEGER
);