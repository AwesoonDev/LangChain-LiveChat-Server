

from typing import List
from langchain import ConversationChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from chatup_chat.core.db_client import DatabaseApiClient

from chatup_chat.core.response_handler import ChatUpStreamHandler
db_client = DatabaseApiClient()


def create_conversation_chain(conversation_id, shop_id, temperature=0):
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        streaming=True,
        callbacks=[ChatUpStreamHandler(conversation_id)],
        temperature=temperature
    )
    template: str = db_client.get_prompt()
    negative_keywords: List[str] = db_client.get_negative_keywords(shop_id)
    template = template.format(negativeKeyWords=''.join(negative_keywords), history="{history}", input="{input}")
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    memory = ConversationSummaryBufferMemory(
        llm=chat, input_key="input", max_token_limit=10000, human_prefix="Customer"
    )
    return ConversationChain(prompt=PROMPT, llm=chat, verbose=True, memory=memory)
