import sqlite3

# Connect to the database
db_path = 'database/products.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the `products` table with a field for feature vectors
cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    image_url TEXT NOT NULL,
    feature_vector TEXT NOT NULL
)
''')

# Create an index to speed up searches
cursor.execute('''CREATE INDEX IF NOT EXISTS idx_price ON products(price)''')

# Save changes and close the connection
conn.commit()
conn.close()

print("Table `products` updated successfully.")
