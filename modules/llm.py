from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama


def get_llm(provider, model, api_key):

    if provider == "groq":

        return ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0.3
        )

    if provider == "openai":

        return ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.3
        )

    if provider == "gemini":

        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0.3
        )

    if provider == "ollama":

        return Ollama(
            model=model
        )