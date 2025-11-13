"""
Fallback service for query translation and aspect creation when Gemini is unavailable.
"""
from typing import List, Dict, Tuple
from deep_translator import GoogleTranslator
from utils.logger import get_logger

logger = get_logger('fallback_service')


def fallback_translate_and_create_aspects(query_vi: str) -> Tuple[List[Dict], Dict]:
    """
    Fallback method: translate query to English and create aspects from each line.

    Args:
        query_vi: Query text in Vietnamese (or any language)

    Returns:
        Tuple of (aspects list, seeds dict)
    """
    try:
        # Split query into lines and translate each to English
        lines = [line.strip() for line in query_vi.split('\n') if line.strip()]
        translated_lines = []

        for line in lines:
            try:
                # Detect language and translate to English if needed
                translator = GoogleTranslator(source='auto', target='en')
                translated = translator.translate(line)
                translated_lines.append(translated)
                logger.debug(f"Translated: '{line}' -> '{translated}'")
            except Exception as e:
                logger.warning(f"Translation error for '{line}': {e}")
                translated_lines.append(line)  # Use original if translation fails

        # Create aspects from translated lines
        aspects = []
        for i, line in enumerate(translated_lines):
            aspect = {
                "name": f"aspect_{i+1}",
                "prompts": [line],  # Use translated line as single prompt
                "weight": 1.0
            }
            aspects.append(aspect)

        # Create minimal seeds (empty for fallback)
        seeds = {
            "unigrams": [],
            "phrases": []
        }

        logger.info(f"Created {len(aspects)} fallback aspects from query")
        return aspects, seeds

    except Exception as e:
        logger.error(f"Fallback translation failed: {e}")
        # Ultimate fallback: single aspect with original query
        aspects = [{
            "name": "fallback_aspect",
            "prompts": [query_vi],
            "weight": 1.0
        }]
        seeds = {"unigrams": [], "phrases": []}
        return aspects, seeds
