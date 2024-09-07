import os
import os.path
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS

docs = list()
for file in os.listdir("chapter_html"):
    path = os.path.join("chapter_html", file)
    loader = BSHTMLLoader(path, open_encoding="utf-8")
    docs.extend(loader.load())

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss")
