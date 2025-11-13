"""
Text search service for BM25 and transcript-based searching.
"""
import os
import sqlite3
from typing import List, Dict, Optional, Any
from utils.logger import get_logger
from services.database import convert_bm25_syntax, highlight_keywords_in_text
from config.settings import B1_DB_PATH, B2_DB_PATH

logger = get_logger('text_search')


def bm25_search_videos(query: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Search videos using BM25 full-text search.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        List of search results with video_id, snippet, score, timestamp, etc.
    """
    # Validate query
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return []

    # Convert user-friendly syntax to SQLite FTS5 syntax
    converted_query = convert_bm25_syntax(query.strip())
    logger.info(f"Original query: '{query}'")
    logger.info(f"Converted query: '{converted_query}'")

    # Check if converted query is valid
    if not converted_query or not converted_query.strip():
        logger.warning("Converted query is empty")
        return []

    results = []
    original_query = query

    # Search both databases
    for db_name, db_path in [("b1", B1_DB_PATH), ("b2", B2_DB_PATH)]:
        if not os.path.exists(db_path):
            logger.warning(f"Database not found: {db_path}")
            continue

        conn = None
        cur = None
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()

            # BM25 search query
            sql = """
            SELECT video_id,
                   snippet(docs, 2, '<b>', '</b>', '...', 50) as snippet,
                   text as full_text,
                   bm25(docs) as score,
                   t as timestamp
            FROM docs
            WHERE text MATCH ?
            ORDER BY bm25(docs) DESC
            LIMIT ?
            """

            logger.debug(f"Executing query for {db_name}: {converted_query}")
            cur.execute(sql, (converted_query, limit))
            rows = cur.fetchall()

            logger.info(f"Found {len(rows)} results in {db_name}")

            for row in rows:
                video_id, snippet, full_text, score, timestamp = row

                # Ensure all values are properly typed
                video_id = str(video_id) if video_id is not None else ""
                snippet = str(snippet) if snippet is not None else ""
                full_text = str(full_text) if full_text is not None else ""
                timestamp = float(timestamp) if timestamp is not None else 0.0

                # Highlight keywords in full text
                try:
                    highlighted_full_text = highlight_keywords_in_text(full_text, original_query)
                except Exception as e:
                    logger.error(f"Error highlighting text for {video_id}: {e}")
                    highlighted_full_text = full_text

                results.append({
                    "video_id": video_id,
                    "snippet": snippet,
                    "full_text": full_text,
                    "highlighted_full_text": highlighted_full_text,
                    "score": float(score) if score is not None else 0.0,
                    "timestamp": timestamp,
                    "db_source": db_name
                })

        except Exception as e:
            logger.error(f"Error searching {db_name}: {e}", exc_info=True)
        finally:
            # Ensure connections are properly closed
            if cur:
                try:
                    cur.close()
                except Exception as e:
                    logger.error(f"Error closing cursor for {db_name}: {e}")
            if conn:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection for {db_name}: {e}")

    # Sort by score and return top results
    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"Returning {len(results[:limit])} total results")
    return results[:limit]


def deduplicate_bm25_results(results: List[Dict[str, Any]], time_threshold: float = 30.0) -> List[Dict[str, Any]]:
    """
    Deduplicate BM25 search results by removing entries from the same video
    that are within time_threshold seconds of each other.

    Args:
        results: List of search results
        time_threshold: Time threshold in seconds for deduplication

    Returns:
        Deduplicated list of results
    """
    if not results:
        return []

    # Group by video_id
    video_groups: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        video_id = result.get("video_id", "")
        if video_id not in video_groups:
            video_groups[video_id] = []
        video_groups[video_id].append(result)

    deduplicated = []

    # Process each video group
    for video_id, group in video_groups.items():
        # Sort by timestamp
        group.sort(key=lambda x: x.get("timestamp", 0.0))

        # Keep first result and any that are far enough from previous kept results
        kept_results = []
        for result in group:
            current_time = result.get("timestamp", 0.0)

            # Check if this result is far enough from all previously kept results
            should_keep = True
            for kept in kept_results:
                kept_time = kept.get("timestamp", 0.0)
                if abs(current_time - kept_time) < time_threshold:
                    should_keep = False
                    break

            if should_keep:
                kept_results.append(result)

        deduplicated.extend(kept_results)

    # Sort by original score
    deduplicated.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    logger.info(f"Deduplicated {len(results)} results to {len(deduplicated)}")
    return deduplicated


def get_transcript_snippets_with_highlights(
    video_id: str,
    start_time: float,
    end_time: float,
    seeds: Dict[str, List[str]],
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Get transcript snippets for a video within a time range with highlighted keywords.

    Args:
        video_id: Video identifier
        start_time: Start timestamp
        end_time: End timestamp
        seeds: Dictionary of seed keywords (unigrams, phrases)
        top_n: Number of top snippets to return

    Returns:
        List of transcript snippets with highlights
    """
    from utils.helpers import dataset_for_video
    from config.settings import B1_DB_PATH, B2_DB_PATH

    # Get the appropriate database path
    dataset = dataset_for_video(video_id)
    db_path = B1_DB_PATH if dataset == "b1" else B2_DB_PATH

    if not os.path.exists(db_path):
        logger.warning(f"Database not found: {db_path}")
        return []

    # Build query from seeds
    keywords = []
    if isinstance(seeds, dict):
        keywords.extend(seeds.get("unigrams", []))
        keywords.extend(seeds.get("phrases", []))

    if not keywords:
        logger.warning("No keywords provided for transcript search")
        return []

    query = " OR ".join(keywords)

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        sql = """
        SELECT text, t as timestamp, bm25(docs) as score
        FROM docs
        WHERE video_id = ?
          AND t BETWEEN ? AND ?
          AND text MATCH ?
        ORDER BY bm25(docs) DESC
        LIMIT ?
        """

        cur.execute(sql, (video_id, start_time, end_time, query, top_n))
        rows = cur.fetchall()

        snippets = []
        for row in rows:
            text, timestamp, score = row
            highlighted_text = highlight_keywords_in_text(text, query)
            snippets.append({
                "text": text,
                "highlighted_text": highlighted_text,
                "timestamp": float(timestamp),
                "score": float(score)
            })

        conn.close()
        logger.debug(f"Found {len(snippets)} transcript snippets for {video_id}")
        return snippets

    except Exception as e:
        logger.error(f"Error getting transcript snippets: {e}", exc_info=True)
        return []
