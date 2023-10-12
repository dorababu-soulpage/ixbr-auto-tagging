import json
import psycopg2
import boto3

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

def s3_uploader(name, body):
    # name is s3 file name
    # boyd is io.BytesIO()
    access_key = config("AWS_S3_ACCESS_KEY_ID")
    secret_key = config("AWS_S3_SECRET_ACCESS_KEY")
    region = config("AWS_S3_REGION")
    bucket = config("AWS_S3_BUCKET_NAME")
    

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3 = session.resource("s3")
    s3.Bucket(bucket).put_object(
        Key=name,
        Body=body.getvalue(),
        ACL="public-read",
        ContentType="application/octet-stream",
    )
    location = session.client("s3").get_bucket_location(Bucket=bucket)[
        "LocationConstraint"
    ]
    uploaded_url = f"https://s3-{location}.amazonaws.com/{bucket}/{name}"
    return uploaded_url