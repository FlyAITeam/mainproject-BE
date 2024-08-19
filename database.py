from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

if os.getenv("WEBSITE_HOSTNAME"):
    env_connection_string = os.getenv("AZURE_POSTGRESQL_CONNECTIONSTRING")
    details = dict(item.split('=') for item in env_connection_string.split())
    SQLALCHEMY_DATABASE_URL = f"postgresql://{quote_plus(details['user'])}:{quote_plus(details['password'])}@{details['host']}:{details['port']}/{details['dbname']}"

else:
    load_dotenv(override=True)
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    SQLALCHEMY_DATABASE_URL = f"postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()