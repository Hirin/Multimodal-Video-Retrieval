"""
Gemini AI service for query enhancement and aspect extraction.
"""
import json
from typing import Dict, List, Any, Tuple
from utils.logger import get_logger
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MAX_TOKENS, USE_LLM_ASPECTS

logger = get_logger('gemini_service')


FALLBACK_ASPECTS = [
    {
        "name": "general scene",
        "prompts": ["a general scene or location"],
        "weight": 1.0
    }
]

FALLBACK_SEEDS = {
    "unigrams": [],
    "phrases": []
}


def extract_aspects_and_seeds_with_gemini(
    query_vi: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    model: str = GEMINI_MODEL,
    max_aspects: str = "N"
) -> Dict[str, Any]:
    """
    Extract search aspects and seed keywords from a Vietnamese query using Gemini AI.

    Args:
        query_vi: Query text in Vietnamese
        temperature: Sampling temperature for generation
        top_p: Top-p sampling parameter
        model: Gemini model to use
        max_aspects: Maximum number of aspects to generate

    Returns:
        Dictionary with 'aspects', 'seeds', 'source', and optionally 'error' keys
    """
    if not (USE_LLM_ASPECTS and GEMINI_API_KEY):
        logger.warning("Using FALLBACK aspects/seeds (no API key)")
        return {
            "aspects": FALLBACK_ASPECTS,
            "seeds": FALLBACK_SEEDS,
            "source": "fallback",
            "error": "no_api_key"
        }

    # System instruction for Gemini
    sys_inst = (
        "You are a query builder for VIDEO RETRIEVAL / TRANSCRIPT SEARCH.\n\n"
        "**NOTE**\n"
        "- Avoid symbols like \"/\", \"\\\",... anywhere in the JSON output; they often trigger FST parsing errors.\n"
        "TASK\n"
        "- INPUT: a NATURAL description (Vietnamese or English) of the target video content.\n"
        "- OUTPUT: a single VALID JSON object with two fields: \"aspects\" and \"seeds\".\n"
        "- Do NOT add explanations, comments, or Markdown in the output JSON.\n\n"
        "OUTPUT CONSTRAINTS\n"
        f"- JSON must use standard double quotes (\").\n"
        f"- \"aspects\": an array with at most {max_aspects} items. Each item is:\n"
        "  {\n"
        "    \"name\": \"<aspect name in VIETNAMESE, short, strongly discriminative>\",\n"
        "    \"prompts\": [\"<2–4 prompts in ENGLISH describing visuals/scene/action>\", ...],\n"
        "    \"weight\": <float 0.80–1.30, 2 decimals, reflecting DIAGNOSTIC STRENGTH>\n"
        "  }\n"
        "- \"seeds\": {\n"
        "    \"unigrams\": [≤ 28 VIETNAMESE single-word keywords, no common stopwords, prioritize distinctive terms],\n"
        "    \"phrases\":  [≤ 18 VIETNAMESE phrases (2–5 words), semantically rich/distinctive]\n"
        "  }\n"
        "- Do not exceed count limits. No academic/theoretical wording. Do not mix languages against the rules:\n"
        "  • aspects.name = VI | aspects.prompts = EN  \n"
        "  • seeds.* = VI\n"
        "- Deduplicate, normalize whitespace, keep numbers/units if highly distinctive (e.g., \"211kg\").\n"
        "- Prioritize distinctive features: entities, objects, colors, actions, context, props, icons, rare numbers.\n"
        "- Weights: 1.00 = neutral. >1.15 for highly diagnostic (unique/rare). 0.90–0.99 for auxiliary/uncertain. Never below 0.80 or above 1.30.\n\n"
        "SUMMARY STRATEGY\n"
        "1) Split the scene into 4–7 independent aspects; for each, write 2–4 EN prompts that clearly describe visuals (usable for BM25/CLIP).\n"
        "2) Seeds (VI):\n"
        "   - unigrams: nouns/subjects/colors/events/props (avoid function words); prefer highly discriminative terms.\n"
        "   - phrases: combine 2–5 words into distinctive expressions (e.g., \"người phụ nữ cầm sen\", \"treo trên xe cẩu\").\n"
        "3) Do not invent details beyond the description. If ambiguous, choose safe options; still output fully structured JSON.\n\n"
        "FEW-SHOT FORMAT (reference)\n"
        "- Each example has \"INPUT:\" and \"OUTPUT:\". Learn the structure and level of detail from them.\n"
        "- For a new description, output ONLY the JSON as in \"OUTPUT\" (do not print \"INPUT/OUTPUT\").\n\n"
        "PROCESS FOR NEW INPUTS\n"
        "1) Decompose the description into aspects (subjects, actions, context, colors, props, numbers).\n"
        "2) Create 2–4 EN prompts per aspect, clear and useful for CLIP/Caption retrieval.\n"
        "3) Generate VI seeds:\n"
        "   - unigrams: distinctive nouns/terms; keep numbers/units (e.g., \"211kg\"); avoid function words.\n"
        "   - phrases: 2–5 words; informative combinations (subject + action + context/color/prop).\n"
        "4) Set weight 0.80–1.30 by diagnostic strength/confidence.\n"
        "5) Output ONE VALID JSON, with nothing else.\n"
    )

    payload = {"query_vi": query_vi}

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)

        # Try new API first
        try:
            resp = client.responses.generate(
                model=model,
                input=json.dumps(payload, ensure_ascii=False),
                system_instruction=sys_inst,
                config={
                    "max_output_tokens": GEMINI_MAX_TOKENS,
                    "response_mime_type": "application/json"
                }
            )
            txt = getattr(resp, "output_text", None)
            if not txt:
                raise RuntimeError("Empty output_text")
        except Exception:
            # Fallback to old API
            cfg = genai.types.GenerateContentConfig(
                max_output_tokens=GEMINI_MAX_TOKENS,
                response_mime_type="application/json",
                system_instruction=sys_inst,
                temperature=temperature,
                top_p=top_p
            )
            resp = client.models.generate_content(
                model=model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=cfg
            )
            txt = getattr(resp, "text", None) or getattr(resp, "output_text", None)

        if not txt:
            raise RuntimeError("Empty text")

        data = json.loads(txt)

        # Validate response
        if "aspects" not in data or not isinstance(data["aspects"], list):
            raise ValueError("no aspects in response")
        if "seeds" not in data or not isinstance(data["seeds"], dict):
            data["seeds"] = {"unigrams": [], "phrases": []}

        data["source"] = "gemini"
        logger.info(f"Successfully extracted {len(data['aspects'])} aspects from Gemini")
        return data

    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        logger.info("Using fallback translation method...")
        from services.fallback_service import fallback_translate_and_create_aspects
        aspects, seeds = fallback_translate_and_create_aspects(query_vi)
        return {
            "aspects": aspects,
            "seeds": seeds,
            "source": "fallback",
            "error": str(e)
        }


def create_fallback_enhanced_data(query_vi: str) -> Dict[str, Any]:
    """
    Create fallback enhanced data when Gemini is not available.

    Args:
        query_vi: Query text in Vietnamese

    Returns:
        Dictionary with 'aspects' and 'seeds'
    """
    logger.info("Creating fallback enhanced data")
    from services.fallback_service import fallback_translate_and_create_aspects

    aspects, seeds = fallback_translate_and_create_aspects(query_vi)
    return {
        "aspects": aspects,
        "seeds": seeds,
        "source": "fallback"
    }
