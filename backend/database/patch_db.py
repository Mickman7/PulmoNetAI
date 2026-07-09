import os
import sqlite3

# The directory where your db.py sits
DB_DIR = "/Users/mickman/Documents/programs/PulmoNetAI/backend/database"

def apply_migration():
    # Scan the directory for any SQLite database files
    db_files = [f for f in os.listdir(DB_DIR) if f.endswith(('.db', '.sqlite', '.sqlite3'))]
    
    if not db_files:
        print(f"ERROR: No database files found in {DB_DIR}")
        print("Ensure you have run the server at least once so SQLAlchemy can create the file.")
        return

    for db_file in db_files:
        db_path = os.path.join(DB_DIR, db_file)
        print(f"Patching database: {db_path}")
        
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Inject the new column safely
            cursor.execute("ALTER TABLE predictions ADD COLUMN attention_summary TEXT;")
            conn.commit()
            print(f"Successfully patched {db_file} with 'attention_summary' column!")
            
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print(f"The column already exists in {db_file}.")
            else:
                print(f"Operational error on {db_file}: {e}")
        finally:
            if conn is not None:
                conn.close()

if __name__ == "__main__":
    apply_migration()