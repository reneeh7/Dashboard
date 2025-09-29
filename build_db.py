# features/build_db.py
# Build db/retailrocket_full.db from data/*.csv with chunked loading + indexes.

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_PATH  = ROOT / "db" / "retailrocket_full.db"
CHUNKSIZE = 500_000  # adjust if needed

def _to_sql_chunked(path: Path, table: str, conn: sqlite3.Connection, mode="replace"):
    first = True
    for chunk in pd.read_csv(path, chunksize=CHUNKSIZE):
        chunk.to_sql(table, conn,
                     if_exists=("replace" if first and mode=="replace" else "append"),
                     index=False)
        first = False

def main():
    required = [
        DATA_DIR / "events.csv",
        DATA_DIR / "item_properties_part1.csv",
        DATA_DIR / "item_properties_part2.csv",
        DATA_DIR / "category_tree.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n- " + "\n- ".join(missing))

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print("→ events …")
    _to_sql_chunked(DATA_DIR / "events.csv", "events", conn, "replace")

    print("→ item_properties part1 …")
    _to_sql_chunked(DATA_DIR / "item_properties_part1.csv", "item_properties", conn, "replace")

    print("→ item_properties part2 …")
    _to_sql_chunked(DATA_DIR / "item_properties_part2.csv", "item_properties", conn, "append")

    print("→ category_tree …")
    pd.read_csv(DATA_DIR / "category_tree.csv").to_sql("category_tree", conn, if_exists="replace", index=False)

    print("→ indexes …")
    conn.executescript("""
      CREATE INDEX IF NOT EXISTS idx_events_vid   ON events(visitorid);
      CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(timestamp);
      CREATE INDEX IF NOT EXISTS idx_events_evt   ON events(event);
      CREATE INDEX IF NOT EXISTS idx_props_item   ON item_properties(itemid);
    """)
    conn.commit(); conn.execute("VACUUM;")

    # counts
    for t in ("events","item_properties","category_tree"):
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"{t}: {n:,} rows")

    conn.close()
    print(f"✅ Created {DB_PATH}")

if __name__ == "__main__":
    main()
