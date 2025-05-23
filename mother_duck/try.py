import duckdb
import os

# Load MotherDuck token from environment
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DUCKDB_CONN = f"md:titanic_test ?motherduck_token={MOTHERDUCK_TOKEN}"

# Connect to DuckDB
con = duckdb.connect(DUCKDB_CONN)

# Query predictions table
df = con.execute("SELECT * FROM predictions").fetchdf()
print(df)
