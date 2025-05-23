import duckdb
import os


MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = f"md:titanic_test ?motherduck_token={MOTHERDUCK_TOKEN}"

con = duckdb.connect(MOTHERDUCK_DB)

con.execute("""
    CREATE TABLE test AS
    SELECT * FROM read_csv_auto('data/raw/test.csv')
""")

print("Test data uploaded to MotherDuck!")