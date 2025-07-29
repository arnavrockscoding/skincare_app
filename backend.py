from flask import Flask, render_template, request, redirect, flash
import mysql.connector
from hashlib import sha256

app = Flask(name)
app.secret_key = 'change_this_secret'

# MySQL credentials
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'rohan',
    'database': 'flask_users'
}

def init_db():
    conn = mysql.connector.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password']
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS flask_users")
    conn.commit()
    conn.close()

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(100) UNIQUE,
            password_hash CHAR(64)
        );
    """)
    conn.commit()
    conn.close()

def hash_pw(pw):
    return sha256(pw.encode()).hexdigest()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form['username'].strip()
        password = request.form['password']
        if not username or not password:
            flash('Fill in both fields', 'error')
        else:
            pw_hash = hash_pw(password)
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            if action == 'signup':
                try:
                    cursor.execute(
                        "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                        (username, pw_hash)
                    )
                    conn.commit()
                    flash('Sign up successful!', 'success')
                except mysql.connector.IntegrityError:
                    flash('Username already taken', 'error')
            elif action == 'login':
                cursor.execute(
                    "SELECT * FROM users WHERE username=%s AND password_hash=%s",
                    (username, pw_hash)
                )
                if cursor.fetchone():
                    flash(f"Welcome back, {username}!", 'success')
                else:
                    flash('Invalid username or password', 'error')
            conn.close()
        return redirect('/')
    return render_template('index.html')

if name == 'main':
    init_db()
    app.run(debug=True)
