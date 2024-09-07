import chainlit as cl
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@cl.on_chat_start
def on_chat_start():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    conversational_rag_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    cl.user_session.set("conversational_rag_chain", conversational_rag_chain)
    cl.user_session.set("session_id", str(uuid.uuid4()))


@cl.on_message
async def main(message: cl.Message):
    conversational_rag_chain: RunnableWithMessageHistory = cl.user_session.get(
        "conversational_rag_chain"
    )
    session_id = cl.user_session.get("session_id")
    user_input = message.content
    answer = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    await cl.Message(content=answer["answer"]).send()
