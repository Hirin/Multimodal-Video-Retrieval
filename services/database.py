"""
Database service for BM25 FTS5 operations.
"""
import sqlite3
import re
from typing import List, Tuple, Optional, Dict, Any
from utils.logger import get_logger

logger = get_logger('database')


def connect_readonly(db_path: str) -> sqlite3.Connection:
    """
    Create a read-only connection to the database.

    Args:
        db_path: Path to the database file

    Returns:
        SQLite connection object
    """
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        logger.debug(f"Connected to database: {db_path}")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database {db_path}: {e}")
        raise


def count_matches(
    db_path: str,
    video_id: str,
    start_time: float,
    end_time: float,
    query_any: Optional[str],
    query_phrase: Optional[str]
) -> Tuple[str, int, int]:
    """
    Count matches for a given video and time window.

    Args:
        db_path: Path to the database file
        video_id: Video identifier
        start_time: Start timestamp
        end_time: End timestamp
        query_any: ANY query (OR-based)
        query_phrase: Phrase query (exact match)

    Returns:
        Tuple of (video_id, any_count, phrase_count)
    """
    try:
        conn = connect_readonly(db_path)
        cur = conn.cursor()
        sql = "SELECT COUNT(*) FROM docs WHERE video_id=? AND t BETWEEN ? AND ? AND docs MATCH ?"

        h_any = h_phr = 0
        if query_any:
            h_any = cur.execute(sql, (video_id, start_time, end_time, query_any)).fetchone()[0]
        if query_phrase:
            h_phr = cur.execute(sql, (video_id, start_time, end_time, query_phrase)).fetchone()[0]

        conn.close()
        logger.debug(f"Counted matches for {video_id}: any={h_any}, phrase={h_phr}")
        return (video_id, int(h_any), int(h_phr))
    except Exception as e:
        logger.error(f"Error counting matches: {e}")
        return (video_id, 0, 0)


def run_fts5_query(db_path: str, match_query: str, limit: int = 800) -> List[Tuple]:
    """
    Run a full-text search query using SQLite FTS5.

    Args:
        db_path: Path to the database file
        match_query: FTS5 match query
        limit: Maximum number of results

    Returns:
        List of tuples (video_id, frame_id, timestamp, score, text)
    """
    try:
        conn = connect_readonly(db_path)
        cur = conn.cursor()

        sql = """
            SELECT video_id, frame_id, t, bm25(docs) as score, text
            FROM docs
            WHERE docs MATCH ?
            ORDER BY score
            LIMIT ?
        """

        results = cur.execute(sql, (match_query, limit)).fetchall()
        conn.close()

        logger.info(f"FTS5 query returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error running FTS5 query: {e}")
        return []


def convert_bm25_syntax(query: str) -> str:
    """
    Convert user-friendly BM25 syntax to SQLite FTS5 syntax.

    Args:
        query: User query with friendly syntax

    Returns:
        FTS5-compatible query string
    """
    # Replace operators: & -> AND, | -> OR, - -> NOT
    query = re.sub(r'\s*&\s*', ' AND ', query)
    query = re.sub(r'\s*\|\s*', ' OR ', query)
    query = re.sub(r'\s*-\s*', ' NOT ', query)

    # Clean up extra spaces
    query = re.sub(r'\s+', ' ', query).strip()

    logger.debug(f"Converted BM25 syntax: {query}")
    return query


def highlight_keywords_in_text(text: str, query: str) -> str:
    """
    Highlight keywords from BM25 query in the text.

    Args:
        text: Text to highlight
        query: Search query

    Returns:
        Text with highlighted keywords (using <b> tags)
    """
    # Type checking and conversion
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not isinstance(query, str):
        query = str(query) if query is not None else ""

    # If either is empty, return original text
    if not text or not query:
        return text

    try:
        # Extract quoted phrases (e.g., "công an")
        quoted_phrases = re.findall(r'"([^"]+)"', query)
        quoted_phrases = [phrase.strip() for phrase in quoted_phrases if phrase.strip()]

        # Remove quoted phrases from query for further processing
        query_without_quotes = re.sub(r'"[^"]*"', '', query)

        # Split by operators and clean up
        keywords = re.split(r'\s*(?:AND|OR|NOT|\(|\))\s*', query_without_quotes)
        keywords = [k.strip() for k in keywords if k.strip()]

        # Split by spaces and punctuation for individual words
        words = re.findall(r'\b\w+\b', query_without_quotes.lower())

        # Add numbers specifically
        numbers = re.findall(r'\b\d+\b', query_without_quotes)

        # Combine all keywords
        all_keywords = quoted_phrases + keywords + words + numbers

        # Remove duplicates while preserving order
        all_keywords = list(dict.fromkeys(all_keywords))
        all_keywords = [k for k in all_keywords if k.strip()]
        all_keywords.sort(key=len, reverse=True)

        # Create highlighted text
        highlighted_text = text

        for keyword in all_keywords:
            # Create pattern based on keyword type
            if keyword.isdigit():
                # For numbers, use word boundaries but allow flexible matching
                pattern = re.compile(
                    r'(?<![a-zA-Z])' + re.escape(keyword) + r'(?![a-zA-Z])(?![^<]*</b>)',
                    re.IGNORECASE
                )
            elif ' ' in keyword:
                # For phrases, match the exact sequence with word boundaries
                words = keyword.split()
                if len(words) > 1:
                    pattern_parts = [re.escape(word) for word in words]
                    flexible_pattern = r'\s+'.join(pattern_parts)
                    pattern = re.compile(
                        r'\b' + flexible_pattern + r'\b(?![^<]*</b>)',
                        re.IGNORECASE
                    )
                else:
                    pattern = re.compile(
                        r'\b' + re.escape(keyword) + r'\b(?![^<]*</b>)',
                        re.IGNORECASE
                    )
            else:
                # For single words, use word boundaries
                pattern = re.compile(
                    r'\b' + re.escape(keyword) + r'\b(?![^<]*</b>)',
                    re.IGNORECASE
                )

            # Replace all occurrences, preserving original case
            def replace_func(match):
                matched_text = match.group(0)
                return f'<b>{matched_text}</b>'

            highlighted_text = pattern.sub(replace_func, highlighted_text)

        return highlighted_text

    except Exception as e:
        logger.error(f"Error highlighting text: {e}")
        return text


def build_bm25_query_from_seeds(query_vi: str, seeds: Dict[str, Any]) -> str:
    """
    Build a BM25 query from seed keywords.

    Args:
        query_vi: Original Vietnamese query
        seeds: Dictionary containing seed keywords

    Returns:
        BM25 query string
    """
    q_any_parts = []
    q_phr_parts = []

    for k, vals in seeds.items():
        if not vals:
            continue
        q_any_parts.extend(vals)
        q_phr_parts.append(f'"{" ".join(vals)}"')

    q_any = " OR ".join(q_any_parts) if q_any_parts else ""
    q_phr = " OR ".join(q_phr_parts) if q_phr_parts else ""

    if q_any and q_phr:
        return f"({q_phr}) OR ({q_any})"
    return q_phr or q_any or ""
