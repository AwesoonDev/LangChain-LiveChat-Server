

from dataclasses import dataclass
import os
from typing import Any
from chatup_chat.adapter.analytics_client import ChatAnalyticsApiClient
from chatup_chat.adapter.db_client import DatabaseApiClient
from chatup_chat.core import Bot
from chatup_chat.core.cache import RedisClusterJson
from flask_socketio import emit
from chatup_chat.core.message_enums import MessageType
from chatup_chat.core.quality_bot import CategoryBot, FilterContext, LatestInquiryBot, QualityBot
from chatup_chat.core.util import save_message
from chatup_chat.models.message import Message
from chatup_chat.core.memory import Memory

CONVERSATION_ANALYSIS = (os.getenv("CONVERSATION_ANALYSIS", "True") == "True")

cache = RedisClusterJson()
db_client = DatabaseApiClient()
chat_analytics = ChatAnalyticsApiClient()


@dataclass
class Room:
    is_live: bool = False
    space_id: str = None
    admin_managed: bool = False
    conversation_id: str = None
    admin_session_id: str = None
    occupant_session_id: str = None
    ai_response: Message = None
    bot: Bot = None
    last_activity_time: int = 0

    def to_dict(self):
        return {
            "is_live": self.is_live,
            "admin_managed": self.admin_managed,
            "conversation_id": self.conversation_id,
            "admin_session_id": self.admin_session_id,
            "occupant_session_id": self.occupant_session_id,
            "space_id": self.space_id,
            "last_activity_time": self.last_activity_time
        }

    def admin_joined(self, admin):
        self.admin_managed = True
        self.admin_session_id = admin.session_id
        self.save()
        emit("admin_joined", namespace="/customer", to=self.occupant_session_id)

    def admin_forfeited(self):
        self.admin_managed = False
        self.save()
        emit("admin_forfeited", namespace="/customer", to=self.occupant_session_id)

    def admin_messages_user(self, msg: Message):
        print("admin says: ", msg.message, self.occupant_session_id)
        emit("admin_response", msg.message, namespace="/customer", to=self.occupant_session_id)
        save_message(self, msg)
        self.save()

    def ai_token_call_back(self, token: str, **kwargs: Any):
        if self.ai_response is None:
            self.ai_response = Message(message="", message_type=MessageType.AI.value)
        self.ai_response.message += token
        if not self.admin_managed:
            emit("ai_response", token, namespace="/customer", to=self.occupant_session_id)

    def ai_feedback_finished(self):
        if self.admin_managed:
            emit("ai_suggestion", {
                "message": self.ai_response.message,
                "conversation_id": self.conversation_id
            }, namespace="/admin", to=self.admin_session_id)
        else:
            save_message(self, self.ai_response)

    def user_says(self, message: Message):
        save_message(self, message)
        if self.admin_managed:
            emit("customer_response", {
                "message": message.message,
                "conversation_id": self.conversation_id
            }, namespace="/admin", to=self.admin_session_id)
        self.bot.converse()
        if CONVERSATION_ANALYSIS:
            chat_analytics.submit_conversation_analytics(self.conversation_id)

    def set_bot(self, bot: Bot):
        self.bot = bot
        self.bot.elder_bot = self.bot
        self.bot.quality_bot = QualityBot(memory=Memory(), elder_bot=self.bot)
        self.bot.call_back_handler = self
        self.bot.quality_bot.call_back_handler = self
        self.bot.category_bot = CategoryBot(memory=Memory(), elder_bot=self.bot)
        self.bot.inquiry_bot = LatestInquiryBot(memory=Memory(), elder_bot=self.bot)
        self.bot.filter_context = FilterContext(memory=Memory(), elder_bot=self.bot)

    def save(self):
        cache[f"room_{self.space_id}:{self.occupant_session_id}:{self.conversation_id}"] = self.to_dict()
