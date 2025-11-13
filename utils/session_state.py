"""
Session state manager for Streamlit application.
Centralizes session state management to avoid scattered st.session_state usage.
"""
import streamlit as st
from typing import Any, Optional
from utils.logger import get_logger

logger = get_logger('session_state')


class SessionStateManager:
    """
    Manages Streamlit session state with a clean interface.
    """

    @staticmethod
    def initialize() -> None:
        """
        Initialize session state with default values.
        """
        # App initialization flag
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
            logger.info("Session state initialized")

            # Clear all watch mode keys to default to Youtube Online
            keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
            for key in keys_to_reset:
                del st.session_state[key]

        # Video server flag
        if '_local_video_server_started' not in st.session_state:
            st.session_state._local_video_server_started = False

        # Video cache
        if 'video_cache' not in st.session_state:
            st.session_state.video_cache = {}

        # Search results cache
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []

        # Dialog context flag
        if 'dialog_context' not in st.session_state:
            st.session_state.dialog_context = False

    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Get a value from session state.

        Args:
            key: Session state key
            default: Default value if key doesn't exist

        Returns:
            Value from session state or default
        """
        return st.session_state.get(key, default)

    @staticmethod
    def set(key: str, value: Any) -> None:
        """
        Set a value in session state.

        Args:
            key: Session state key
            value: Value to set
        """
        st.session_state[key] = value

    @staticmethod
    def delete(key: str) -> None:
        """
        Delete a key from session state.

        Args:
            key: Session state key to delete
        """
        if key in st.session_state:
            del st.session_state[key]

    @staticmethod
    def exists(key: str) -> bool:
        """
        Check if a key exists in session state.

        Args:
            key: Session state key

        Returns:
            True if key exists, False otherwise
        """
        return key in st.session_state

    @staticmethod
    def clear_watch_modes() -> None:
        """
        Clear all watch mode keys from session state.
        """
        if SessionStateManager.get('dialog_context', False):
            logger.debug("Skipping watch mode reset due to dialog context")
            return

        keys_to_reset = [key for key in st.session_state.keys() if key.startswith("watch_mode_")]
        for key in keys_to_reset:
            del st.session_state[key]
        logger.debug(f"Cleared {len(keys_to_reset)} watch mode keys")

    @staticmethod
    def set_dialog_context(active: bool = True) -> None:
        """
        Set dialog context flag.

        Args:
            active: Whether dialog is active
        """
        st.session_state.dialog_context = active

    @staticmethod
    def is_dialog_active() -> bool:
        """
        Check if a dialog is currently active.

        Returns:
            True if dialog is active, False otherwise
        """
        return SessionStateManager.get('dialog_context', False)

    @staticmethod
    def set_server_started(started: bool = True) -> None:
        """
        Set video server started flag.

        Args:
            started: Whether server is started
        """
        st.session_state._local_video_server_started = started

    @staticmethod
    def is_server_started() -> bool:
        """
        Check if video server is started.

        Returns:
            True if server is started, False otherwise
        """
        return SessionStateManager.get('_local_video_server_started', False)

    @staticmethod
    def cache_video(video_id: str, data: Any) -> None:
        """
        Cache video data.

        Args:
            video_id: Video identifier
            data: Data to cache
        """
        if 'video_cache' not in st.session_state:
            st.session_state.video_cache = {}
        st.session_state.video_cache[video_id] = data

    @staticmethod
    def get_cached_video(video_id: str) -> Optional[Any]:
        """
        Get cached video data.

        Args:
            video_id: Video identifier

        Returns:
            Cached data or None if not found
        """
        cache = SessionStateManager.get('video_cache', {})
        return cache.get(video_id)

    @staticmethod
    def cache_search_results(results: list) -> None:
        """
        Cache search results.

        Args:
            results: Search results to cache
        """
        st.session_state.search_results = results

    @staticmethod
    def get_search_results() -> list:
        """
        Get cached search results.

        Returns:
            Cached search results or empty list
        """
        return SessionStateManager.get('search_results', [])

    @staticmethod
    def clear_cache() -> None:
        """
        Clear all cached data.
        """
        if 'video_cache' in st.session_state:
            st.session_state.video_cache = {}
        if 'search_results' in st.session_state:
            st.session_state.search_results = []
        logger.info("Cleared all cached data")


# Initialize session state on module import
if 'app_initialized' not in st.session_state:
    SessionStateManager.initialize()
