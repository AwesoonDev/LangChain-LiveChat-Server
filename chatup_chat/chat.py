import json

from flask import Flask, request
from langchain import OpenAI
from chatup_chat.core.customer_conversation import CustomerConversation
from chatup_chat.core.db_client import DatabaseApiClient
from flask_socketio import SocketIO, emit
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from typing import Any
from langchain.chains.question_answering import load_qa_chain

from chatup_chat.core.open_ai_client import get_user_query_embedding

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

db_client = DatabaseApiClient()


class ChatUpStreamHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any):
        emit("ai_response", token)


conversations = {}


def create_conversation_chain():
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        streaming=True,
        callbacks=[ChatUpStreamHandler()],
        temperature=0
    )
    template = """The following is a friendly conversation between a human and an AI customer Assistant.
the context helps the AI to answer the customer's question. It is very important that the The AI is not talkative and provides short answers.
If the AI does not know the answer to a question based on the context provided. it truthfully says it
does not know and provides the store contact info and asks them to contact them. While the AI answers
the customers questions based on the contexts provided the AI does not mention the word context or
let the customer know where it is getting his knowledge from. The context provided does not necessarily
relate to the question being asked if its not related refer them to the store contact info. The AI
only gives responses that satisfy the question and not extra unecessary information from the context or anywhere else.
The AI response is very important to be always in html format. the ai response needs to be well formatted with any necessary
indenations like new lines (<br>) and such. It is important that the AI provides actual uselful information and not
placeholders
{history}
Current conversation:
Customer: {input}
AI Assistant:"""

    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    memory = ConversationSummaryBufferMemory(
        llm=chat, input_key="input", max_token_limit=10000, human_prefix="Customer"
    )
    return ConversationChain(prompt=PROMPT, llm=chat, verbose=True, memory=memory)


@socketio.on("connect")
def handle_connect():
    conversations[request.sid] = {"conversation": None}
    print(f"New connection: {request.sid}")


@socketio.on("disconnect")
def handle_disconnect():
    if request.sid in conversations:
        del conversations[request.sid]
    print(f"Disconnected: {request.sid}")


@socketio.on("user_message")
def handle_user_message(data):
    print(data)
    user_input = data["message"]
    shop_id = int(data["shop_id"])
    print(user_input)
    if request.sid in conversations:
        conv_chain = conversations[request.sid]["conversation"]
        if conv_chain is None:
            conversations[request.sid]["conversation"] = CustomerConversation(shop_id=shop_id)
        conversations[request.sid]["conversation"].register_shop(shop_id)
        conversations[request.sid]["conversation"].converse(user_input)
    else:
        print(f"Error: No conversation found for {request.sid}")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=8014)
