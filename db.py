import sqlite3

def connect_db():
    conn = sqlite3.connect("career.db")
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = connect_db()
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name TEXT,
        surname TEXT,
        email TEXT UNIQUE,
        password TEXT,
        education TEXT
    )
    """)

    # RESULTS TABLE (VERY IMPORTANT 🔥)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        skills TEXT,
        interests TEXT,
        education TEXT,
        result TEXT
    )
    """)

    # RESET TOKENS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reset_tokens (
        email TEXT PRIMARY KEY,
        token TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def add_user(first_name, surname, email, hashed_password, education):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (first_name, surname, email, password, education) VALUES (?, ?, ?, ?, ?)",
                       (first_name, surname, email, hashed_password, education))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        conn.close()

def get_user(email):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def store_reset_token(email, token):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR REPLACE INTO reset_tokens (email, token) VALUES (?, ?)", (email, token))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error storing reset token: {e}")
        return False
    finally:
        conn.close()

def verify_reset_token(email, token):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reset_tokens WHERE email = ? AND token = ?", (email, token))
    row = cursor.fetchone()
    if row:
        cursor.execute("DELETE FROM reset_tokens WHERE email = ?", (email,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def update_password(email, hashed_password):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, email))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error updating password: {e}")
        return False
    finally:
        conn.close()
    