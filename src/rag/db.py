import os, pathlib, psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
load_dotenv()

def connect():
    conn = psycopg.connect(os.environ['DATABASE_URL'], autocommit=True)
    conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)
    return conn

def init():
    conn = connect()
    conn.execute(pathlib.Path('schema.sql').read_text())
    return conn
