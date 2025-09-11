import google.generativeai as genai

from app.core.config import settings


def analyze_with_gemini(
    prompt: str,
) -> str:
    """
    Safe wrapper for calling Gemini.
    - If GEMINI_API_KEY not set, return-friendly message.
    - Otherwise, call Gemini and return text.
    """

    # gemini integration
    if not settings.is_gemini_available:
        return "AI not configured. Please set GEMINI_API_KEY in your environment."

    try:
        model = genai.GenerativeModel(model_name=settings.gemini_model)
        response = model.generate_content(prompt)
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"AI call failed: {str(e)}"
