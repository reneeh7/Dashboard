# features/shrink_db.py
# Create db/retailrocket_mini.db from db/retailrocket_full.db
# Options: --days (by time) or --top_n (by customers)

import argparse, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "db" / "retailrocket_full.db"
DST  = ROOT / "db" / "retailrocket_mini.db"

def ensure_src(conn):
    tabs = {r[0] for r in conn.execute("SELECT name FROM src.sqlite_master WHERE type='table'")}
    need = {"events","item_properties","category_tree"}
    miss = need - tabs
    if miss:
        raise RuntimeError(f"Source DB missing tables: {', '.join(sorted(miss))}")

def by_time(conn, days: int):
    max_ts = conn.execute("SELECT MAX(timestamp) FROM src.events;").fetchone()[0]
    if max_ts is None: raise RuntimeError("No events in source.")
    cut = int(max_ts - days*24*60*60*1000)
    print(f"Keeping last {days} days (timestamp >= {cut})")
    conn.execute("CREATE TABLE events AS SELECT * FROM src.events WHERE timestamp >= ?;", (cut,))
    conn.execute("CREATE TABLE sel_itemids AS SELECT DISTINCT itemid FROM events;")

def by_customers(conn, top_n: int):
    print(f"Keeping TOP {top_n} visitors by volume")
    conn.execute("""
      CREATE TABLE selected_visitors AS
      SELECT visitorid FROM src.events
      GROUP BY visitorid
      ORDER BY COUNT(*) DESC
      LIMIT ?;
    """, (top_n,))
    conn.execute("""
      CREATE TABLE events AS
      SELECT e.* FROM src.events e JOIN selected_visitors s USING(visitorid);
    """)
    conn.execute("CREATE TABLE sel_itemids AS SELECT DISTINCT itemid FROM events;")

def finish(conn):
    conn.execute("""
      CREATE TABLE item_properties AS
      SELECT p.* FROM src.item_properties p JOIN sel_itemids s USING(itemid);
    """)
    conn.execute("CREATE TABLE category_tree AS SELECT * FROM src.category_tree;")
    conn.executescript("""
      CREATE INDEX IF NOT EXISTS idx_events_vid ON events(visitorid);
      CREATE INDEX IF NOT EXISTS idx_events_ts  ON events(timestamp);
      CREATE INDEX IF NOT EXISTS idx_props_item ON item_properties(itemid);
    """)

def main(strategy: str, days: int, top_n: int):
    DST.parent.mkdir(parents=True, exist_ok=True)
    if DST.exists(): DST.unlink()
    conn = sqlite3.connect(DST)
    try:
        conn.execute(f"ATTACH DATABASE '{SRC}' AS src;")
        ensure_src(conn)
        if strategy == "by_time":
            by_time(conn, days)
        else:
            by_customers(conn, top_n)
        finish(conn)
        conn.commit(); conn.execute("VACUUM;")

        n_ev  = conn.execute("SELECT COUNT(*) FROM events;").fetchone()[0]
        n_vid = conn.execute("SELECT COUNT(DISTINCT visitorid) FROM events;").fetchone()[0]
        n_ip  = conn.execute("SELECT COUNT(*) FROM item_properties;").fetchone()[0]
        print(f"✅ Created {DST}")
        print(f"   Events: {n_ev:,} | Users: {n_vid:,} | Item props: {n_ip:,}")
    finally:
        conn.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["by_time","by_customers"], default="by_time")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--top_n", type=int, default=20000)
    args = ap.parse_args()
    main(args.strategy, args.days, args.top_n)
