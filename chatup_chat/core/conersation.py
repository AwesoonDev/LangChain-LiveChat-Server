

from abc import ABC
from typing import Any
import openai
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from flask_socketio import emit

from chatup_chat.core.db_client import DatabaseApiClient
from chatup_chat.core import CHAT_LLM_MODEL


db_client = DatabaseApiClient()


class ChatUpStreamHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any):
        emit("ai_response", token)


class Conversation(ABC):
    def __init__(self, shop_id=None) -> None:
        self._store_contact_info = None
        self.register_shop(shop_id)
        self._messages = []
        self.model = CHAT_LLM_MODEL

    def register_shop(self, shop_id):
        self._shop_id = shop_id
        if self._shop_id and not self._store_contact_info:
            self._store_contact_info = db_client.get_shop_contact_info(self._shop_id)

    def add_assistant_message(self, assistance: str):
        self._messages.append(
            {"role": "assistant", "content": assistance}
        )

    def create_customer_msg(self, user_query: str):
        return {
            "role": self.role,
            "content": user_query
        }

    def get_latest_user_messages(self):
        return [msg for msg in self._messages if msg["role"] in ["assistant", "user"]]
