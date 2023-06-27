
import json
import openai
from chatup_chat.core.conersation import Conversation
from chatup_chat.core.context import manage_context
from chatup_chat.core.db_client import DatabaseApiClient
from chatup_chat.core.open_ai_client import get_user_query_embedding
from flask_socketio import emit


db_client = DatabaseApiClient()

INITIAL_SYSTEM_MSG = (
    "You are an helpful AI customer assistant. You are the store's representative"
    "You are not talkative and provide short answers. "
    "If you dont know the Answer to a question you truthfuly say you do not know "
    "It is important that you do not respond with any extra information that is not being asked of you. "
    "You should only answer with context that is provided to you"
)

CONTEXT_MSG_PREFIX = ""

class StreamCallBackHandler:
    def new_token(self, response):
        choice = response["choices"][0]
        if not choice.get("finish_reason"):
            token = choice["delta"]["content"]
            emit("ai_response", token)
            print(token, end="", flush=True)
            return token
        else:
            return ""
        

class ChatOpenAiCompletion:
    def __init__(self, model, messages, stream, conversation: Conversation) -> None:
        self.model = model
        self.messages = messages
        self.stream_call_back = StreamCallBackHandler()
        self.conversation = conversation
        self.stream = stream

    def create(self):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            stream=self.stream
        )
        assistant_message = ""
        print(">> AI: ", end="", flush=True)
        for r in response:
            assistant_message += self.stream_call_back.new_token(r)
        self.conversation.add_assistant_message(assistant_message)


class CustomerConversation(Conversation):
    def __init__(self, shop_id=None) -> None:
        super().__init__(shop_id=shop_id)
        self.role = "user"
        self.init_sys_message = INITIAL_SYSTEM_MSG
        self.add_store_contact_info()
        self._initial_system_msg = {
            "role": "system",
            "content": INITIAL_SYSTEM_MSG
        }
        self._messages.append(self._initial_system_msg)
        self.message_histories = []

    def converse(self, user_query: str):
        print(f">> Customer: {user_query}")
        # self._messages.append(self.create_system_msg_based_on_new_context(user_query))
        self._messages.append(self.create_customer_msg(user_query))
        user_query_contexted = manage_context(self.get_latest_user_messages())
        self._messages.append(self.create_system_msg_based_on_new_context(user_query_contexted))
        ChatOpenAiCompletion(
            model=self.model,
            messages=self._messages,
            stream=True,
            conversation=self
        ).create()
        print(self._messages)

    def add_store_contact_info(self):
        if self._store_contact_info:
            self.init_sys_message += f". Your store Contact email is {self._store_contact_info}"

    def get_relevant_context(self, user_query: str):
        query_embedding = get_user_query_embedding(user_query)
        context = db_client.get_closest_shop_doc(query_embedding, self._shop_id)
        return context

    def create_system_msg_based_on_new_context(self, user_query: str):
        context = self.get_relevant_context(user_query)
        content = CONTEXT_MSG_PREFIX.format(context=context)
        f = open("verbose-context.txt", "w+")
        result = {
            "role": "system",
            "content": content
        }
        f.write(json.dumps(result))
        f.close()
        return result
