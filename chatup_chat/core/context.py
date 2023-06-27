
import openai
from chatup_chat.core.db_client import DatabaseApiClient
from chatup_chat.core.open_ai_client import get_user_query_embedding
from chatup_chat.core import LLM_MODEL


db_client = DatabaseApiClient()


PROMPT = """
    given this conversation tell us what the latest user request is specifically and in details: \n
    """


def manage_context(messages):
    print(messages)
    response = openai.Completion.create(
        model=LLM_MODEL,
        prompt=f"{PROMPT}{messages}"
    )
    return response
