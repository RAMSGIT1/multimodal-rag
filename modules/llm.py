# from langchain_groq import ChatGroq
# from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.llms import Ollama


# def get_llm(provider, model, api_key):

#     if provider == "groq":

#         return ChatGroq(
#             api_key=api_key,
#             model=model,
#             temperature=0.3
#         )

#     if provider == "openai":

#         return ChatOpenAI(
#             api_key=api_key,
#             model=model,
#             temperature=0.3
#         )

#     if provider == "gemini":

#         return ChatGoogleGenerativeAI(
#             google_api_key=api_key,
#             model=model,
#             temperature=0.3
#         )

#     if provider == "ollama":

#         return Ollama(
#             model=model
#         )


import requests

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama


# =========================
# GET AVAILABLE MODELS
# =========================

def get_available_models(provider, api_key=None):

    # =========================
    # GROQ
    # =========================

    if provider == "groq":

        if not api_key:
            return [], "API key required"

        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )

            if response.status_code != 200:
                return [], f"Groq API error: {response.status_code}"

            data = response.json()

            models = [
                model["id"]
                for model in data.get("data", [])
                if model.get("active", True)
            ]

            models.sort()

            return models, None

        except Exception as e:
            return [], f"Groq connection error: {str(e)}"


    # =========================
    # OPENAI
    # =========================

    if provider == "openai":

        if not api_key:
            return [], "API key required"

        try:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={
                    "Authorization": f"Bearer {api_key}"
                },
                timeout=10
            )

            if response.status_code != 200:
                return [], f"OpenAI API error: {response.status_code}"

            data = response.json()

            models = [
                model["id"]
                for model in data.get("data", [])
            ]

            # Keep models useful for text generation.
            # Remove obvious non-chat/special-purpose models.
            excluded = (
                "embedding",
                "moderation",
                "whisper",
                "tts",
                "dall-e",
                "search",
                "transcribe",
                "realtime"
            )

            models = [
                model
                for model in models
                if not any(
                    word in model.lower()
                    for word in excluded
                )
            ]

            models.sort()

            return models, None

        except Exception as e:
            return [], f"OpenAI connection error: {str(e)}"


    # =========================
    # GEMINI
    # =========================

    if provider == "gemini":

        if not api_key:
            return [], "API key required"

        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={
                    "key": api_key
                },
                timeout=10
            )

            if response.status_code != 200:
                return [], f"Gemini API error: {response.status_code}"

            data = response.json()

            models = []

            for model in data.get("models", []):

                supported_methods = model.get(
                    "supportedGenerationMethods",
                    []
                )

                if "generateContent" not in supported_methods:
                    continue

                name = model.get("name", "")

                # Gemini returns names such as:
                # models/gemini-2.5-flash
                if name.startswith("models/"):
                    name = name.replace("models/", "", 1)

                if name:
                    models.append(name)

            models.sort()

            return models, None

        except Exception as e:
            return [], f"Gemini connection error: {str(e)}"


    # =========================
    # OLLAMA
    # =========================

    if provider == "ollama":

        try:
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=5
            )

            if response.status_code != 200:
                return [], "Ollama server is not running"

            data = response.json()

            models = [
                model["name"]
                for model in data.get("models", [])
            ]

            models.sort()

            if not models:
                return [], "No Ollama models installed"

            return models, None

        except Exception:
            return [], (
                "Ollama is not running. "
                "Start Ollama and try again."
            )


    return [], "Unsupported provider"


# =========================
# CREATE LLM
# =========================

def get_llm(provider, model, api_key, temperature=0.3):

    if provider == "groq":

        return ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature
        )

    if provider == "openai":

        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature
        )

    if provider == "gemini":

        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature
        )

    if provider == "ollama":

        return Ollama(
            model=model,
            temperature=temperature
        )

    raise ValueError(
        f"Unsupported provider: {provider}"
    )