from __future__ import annotations
from flask import request

from flask_socketio import Namespace, emit
from chatup_chat.api.models.customer import AdminMessageSchema
from chatup_chat.api.util import authorize_admin
from chatup_chat.core.admin.admin_manager import AdminManager
from chatup_chat.core.cache import RedisClusterJson
from chatup_chat.core.loader import load_chat_bot
from chatup_chat.core.message_enums import MessageType
from chatup_chat.core.room.room_manager import RoomManager
from chatup_chat.models.message import Message

message_schema = AdminMessageSchema()
cache = RedisClusterJson()
room_manager = RoomManager()
admin_manager = AdminManager()

room_manager.admin_manager = admin_manager
admin_manager.room_manager = room_manager


class Admin(Namespace):

    def on_connect(self):
        admin = authorize_admin(admin_manager)
        print("Admin connected")
        rooms = room_manager.get_live_rooms(admin.shop_id)
        conversation_ids = [room.conversation_id for room in rooms]

        emit("live_conversations", conversation_ids)

    def on_disconnect(self):
        admin = admin_manager.get_admin_by_session(request.sid)
        admin_manager.checkout_admin(admin)
        print("Admin disconnected")

    def on_get_live_conversations(self):
        admin = admin_manager.get_admin_by_session(request.sid)
        rooms = room_manager.get_live_rooms(admin.shop_id)
        conversation_ids = [room.conversation_id for room in rooms]
        emit("live_conversations", conversation_ids)

    def on_join(self, data):
        conversation_id = data["conversation_id"]
        admin = admin_manager.get_admin_by_session(request.sid)
        print("Admin joining: ", conversation_id)
        room = room_manager.get_room_by_conversation_id(conversation_id)
        room.admin_joined(admin)

    def on_message(self, data):
        admin_message = message_schema.load(data)
        admin = admin_manager.get_admin_by_session(request.sid)
        customer_bot = load_chat_bot(conversation_id=admin_message["conversation_id"])
        print("Received another event with data: ", data)
        room = room_manager.get_room_by_conversation_id(admin_message["conversation_id"])
        # room.set_bot(customer_bot)
        admin.message_user(room, Message(
            message=admin_message["message"],
            message_type=MessageType.USER.value,
            metadata=["admin"]
        ))

    def on_forfeit(self, data):
        conversation_id = data["conversation_id"]
        room = room_manager.get_room_by_conversation_id(conversation_id)
        room.admin_forfeited()
