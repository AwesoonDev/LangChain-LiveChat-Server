from __future__ import annotations

from flask_socketio import Namespace, emit
from chatup_chat.api.models.customer import CustomerSchema, MessageSchema
from chatup_chat.core.loader import load_chat_bot
from chatup_chat.core.cache import RedisClusterJson
from chatup_chat.core.chat import Chat
from chatup_chat.core.customers import initiate_conversation

customer_schema = CustomerSchema()
message_schema = MessageSchema()
chat = Chat()
cache = RedisClusterJson()


class Customer(Namespace):

    def on_connect(self):
        print("Customer connected")

    def on_disconnect(self):
        print("Customer disconnected")

    def on_init(self, data):
        customer = customer_schema.load(data)
        conversation_id = initiate_conversation(customer)
        emit("init_response", conversation_id)

    def on_message(self, data):
        print("Received another event with data: ", data)
        customer_message = message_schema.load(data)
        customer_bot = load_chat_bot(conversation_id=customer_message["conversation_id"])
        customer_bot.converse(customer_message["message"])
