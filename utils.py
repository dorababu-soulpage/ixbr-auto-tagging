import json
import psycopg2

from decouple import config

db = config("DATABASE_NAME")
host = config("DATABASE_HOST")
username = config("DATABASE_USERNAME")
password = config("DATABASE_PASSWORD")

def get_db_record(file_id):

    db_url = f"postgresql://{username}:{password}@{host}:5432/{db}"
    try:
        # Attempt to connect and execute queries
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM files where id={file_id}")
        row = cursor.fetchone()
        columns = [column[0] for column in cursor.description]
        return dict(zip(columns, row))
    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def update_db_record(file_id, data):
    import psycopg2

    db_url = f"postgresql://{username}:{password}@{host}:5432/{db}"
    try:
        # Attempt to connect and execute queries
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()
        # Convert the new data to a JSON string
        new_json_data = json.dumps(data)

        # SQL query to update the JSON field
       # SQL query to update the JSON field using -> operator
        update_sql = f"""
            UPDATE files
            SET extra = extra || %s::jsonb
            WHERE id = %s
        """
        # Execute the SQL query with parameters
        cursor.execute(update_sql, (new_json_data, file_id))
        # Commit the changes and close the connection
        connection.commit()

    except psycopg2.Error as e:
        print("Error connecting to the database:", e)
    finally:
        # Close cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()