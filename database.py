import psycopg2
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv
import json
import streamlit as st

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

def save_query_to_db(query_text):
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO research_queries (query_text) VALUES (%s) RETURNING id",
                (query_text,)
            )
            return cur.fetchone()[0]

def save_results_to_db(query_id, results):
    """Save results as JSON in database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Ensure results is a dictionary
                if not isinstance(results, dict):
                    results = {
                        'final_content': str(results),
                        'raw_research': '',
                        'topic': ''
                    }
                
                # Save to database
                cur.execute(
                    "INSERT INTO research_outputs (query_id, title, content) VALUES (%s, %s, %s::jsonb) RETURNING id",
                    (query_id, results.get('topic', ''), json.dumps(results))
                )
                return cur.fetchone()[0]
            except Exception as e:
                print(f"Error saving to database: {e}")
                conn.rollback()
                raise e

def get_filtered_history(search_filter=None, start_date=None, end_date=None, sort_order="Newest First"):
    """Get filtered history from database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT 
                    rq.id,
                    rq.query_text,
                    rq.created_at,
                    CASE 
                        WHEN ro.content IS NULL THEN NULL
                        ELSE ro.content::text
                    END as content,
                    ro.title
                FROM research_queries rq
                LEFT JOIN research_outputs ro ON rq.id = ro.query_id
                WHERE 1=1
            """
            params = []
            
            if search_filter:
                query += " AND (rq.query_text ILIKE %s OR ro.title ILIKE %s)"
                search_term = f"%{search_filter}%"
                params.extend([search_term, search_term])
            
            if start_date:
                query += " AND rq.created_at::date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND rq.created_at::date <= %s"
                params.append(end_date)
            
            if sort_order == "Newest First":
                query += " ORDER BY rq.created_at DESC"
            elif sort_order == "Oldest First":
                query += " ORDER BY rq.created_at ASC"
            else:  # Relevance
                if search_filter:
                    query += " ORDER BY similarity(rq.query_text, %s) DESC"
                    params.append(search_filter)
                else:
                    query += " ORDER BY rq.created_at DESC"
            
            query += " LIMIT 10"
            
            cur.execute(query, params)
            return cur.fetchall()

def get_query_content(query_id):
    """Get content for a specific query"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    ro.content,
                    ro.title,
                    array_agg(json_build_object('url', rs.url, 'title', rs.title)) as sources
                FROM research_outputs ro
                LEFT JOIN research_sources rs ON ro.id = rs.output_id
                WHERE ro.query_id = %s
                GROUP BY ro.id
            """, (query_id,))
            return cur.fetchone()

def test_db():
    try:
        conn = get_db_connection()
        st.success("Database connection successful!")
        conn.close()
    except Exception as e:
        st.error(f"Database connection failed: {e}")

if __name__ == "__main__":
    test_db()