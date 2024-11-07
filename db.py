# Create Embeddings of fields of stream and stream_meta tables using Open AI embdding API

from openai import OpenAI
from dotenv import load_dotenv
import psycopg2
import time
import redis
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

load_dotenv()

# Initialize OpenAI Client and Redis
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# r = redis.Redis(host='localhost', port=6379, db=0)

# Connect Database
conn = psycopg2.connect(
    database=os.getenv("POSTGRESQL_DB_NAME"),
    user=os.getenv("POSTGRESQL_USER"),
    password=os.getenv("POSTGRESQL_PASSWORD"),
    host=os.getenv("POSTGRESQL_HOST"),
    port=os.getenv("POSTGRESQL_PORT")
)
cursor = conn.cursor()

# Function to generate embedding using OpenAI
def get_embedding(text):
    response = openai_client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

# Function to get embedding and store
def get_embedding_and_store(stream_id, text):
    # Generate embdding
    embedding = get_embedding(text)

    # Convert embedding to string for storage
    embedding_str = ','.join(map(str, embedding))
    # print(embedding_str)

    # Check if stream_id already exist
    cursor.execute("SELECT embedding_vector FROM stream_embeddings WHERE id = %s", (stream_id,))
    result = cursor.fetchone()

    if not result:
        # Store embedding in PostgreSQL db
        cursor.execute(
            """
            INSERT INTO stream_embeddings (id, embedding_vector)
            VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET embedding_vector = EXCLUDED.embedding_vector;
            """, (stream_id, embedding_str)
        )
        conn.commit()

        # Cache the embedding in Redis
        # r.set(f"embedding:{stream_id}", embedding_str)

        print(f"Embedding for stream_id {stream_id} created and stored.")

# Function to merge fields, generate embedding, and store it
def process_stream_for_embedding(stream_id):
    try:
        # Fetch necessary fields from stream and stream_meta tables
        cursor.execute("""
            SELECT s.source, s.description, s.tags, sm.entity, sm.attribute
            FROM stream AS s
            LEFT JOIN stream_meta AS sm ON s.id = sm.stream_id
            WHERE s.id = %s;
        """, (stream_id,))

        # Retrieve the result
        source, description, tags, entity, attribute = cursor.fetchone()

        # Check if entity and attribute are available
        if entity and attribute:
            # Concatenate all fields
            merged_text = f"Source: {source}. Description: {description}. Tags: {tags}. Entity: {entity}. Attribute: {attribute}."
        else:
            # Concatenate only stream fields
            merged_text = f"Source: {source}. Description: {description}. Tags: {tags}."

        # Call the embedding function with the concatenated text
        get_embedding_and_store(stream_id, merged_text)
        
    except Exception as e:
        print(f"Error processing stream_id {stream_id}: {e}")

def get_cached_embedding(stream_id):
    cursor.execute("SELECT embedding_vector FROM stream_embeddings WHERE id = %s", (stream_id,))
    result = cursor.fetchone()
    if result:
        embedding_str = result[0]
        # r.set(f"embedding:{stream_id}", embedding_str)
        return np.fromstring(embedding_str, sep=',')
    return None
    # # Attempt to retrieve embedding from Redis cache
    # embedding_str = r.get(f"embedding:{stream_id}")
    # if embedding_str:
    #     return np.fromstring(embedding_str.decode('utf-8'), sep=',')
    # else:
    #     # If not cached, fetch from PostgreSQL and cache it
    #     cursor.execute("SELECT embedding_vector FROM stream_embeddings WHERE id = %s", (stream_id,))
    #     result = cursor.fetchone()
    #     if result:
    #         embedding_str = result[0]
    #         r.set(f"embedding:{stream_id}", embedding_str)
    #         return np.fromstring(embedding_str, sep=',')
    # return None

# Function to process new entries in embeddings_queue
def process_new_entries():
    while True:
        # Check for new entries in the embeddings queue
        cursor.execute("SELECT table_name, row_id FROM embeddings_queue WHERE processed = FALSE LIMIT 1;")
        new_entry = cursor.fetchone()

        if new_entry:
            table_name, row_id = new_entry

            # Process based on the table type
            process_stream_for_embedding(row_id)

            # Mark entry as processed
            cursor.execute("UPDATE embeddings_queue SET processed = TRUE WHERE table_name = %s AND row_id = %s;", (table_name, row_id))
            conn.commit()
        else:
            # No new entries, wait before checking again
            time.sleep(10)

# Function to process all streams in batch
def process_all_streams():
    try:
        # Fetch all unique stream_ids from the stream table
        cursor.execute("SELECT id FROM stream;")
        stream_ids = cursor.fetchall()

        for stream_id_tuple in stream_ids:
            stream_id = stream_id_tuple[0]
            process_stream_for_embedding(stream_id)
            # Pause briefly to avoid hitting rate limits
            time.sleep(1)
    except Exception as e:
        print(f"Error processing all streams: {e}")

# Function to handle user query and find closest matching streams
def find_closest_streams(user_query, top_n=5):
    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    # Fetch precomputed stream metadata embeddings from the database
    cursor.execute("SELECT id, stream_id, description, embedding_vector FROM stream_embeddings")
    streams = cursor.fetchall()

    # Calculate the similarity between streams and query
    embeddings = np.array([np.fromstring(row[3], sep=',') for row in streams])
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get top N matches
    top_matches = sorted(zip(similarities, streams), reverse=True)[:top_n]
    
    # Fetch data from stream and stream_meta table using stream_id
    results = []

    for similarity, stream in top_matches:
        stream_id = stream[1]

        cursor.execute("SELECT * FROM stream WHERE id = %s;", {stream_id,})
        stream_data = cursor.fetchone()

        cursor.execute("SELECT * FROM stream_meta WHERE stream_id = %s", (stream_id,))
        stream_meta_data = cursor.fetchall()

        result = {
            "similarity": similarity,
            "stream_data": stream_data,
            "stream_meta_data": stream_meta_data
        }
        results.append(result)
    
    for result in results:
        print(f"Similarity: {result['similarity']}")
        print("Stream Data:", result['stream_data'])
        print("Stream Meta Data:")
        for meta in result['stream_meta_data']:
            print(meta)
        print("\n" + "-"*50 + "\n")
    
    return results

if __name__ == "__main__":
    try:
        mode = input("Choose mode: (1) Process all streams, (2) Process new entries, (3) Query search: ")

        if mode == "1":
            process_all_streams()
        elif mode == "2":
            process_new_entries()
        elif mode == "3":
            user_query = input("Enter your query: ")
            find_closest_streams(user_query)
        else:
            print("Invalid mode selected.")
    
    except KeyboardInterrupt:
        print("Service interrupted.")
    finally:
        cursor.close()
        conn.close()
        print("Database connection closed.")