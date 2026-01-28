import config_data
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os
import dotenv
dotenv.load_dotenv()

class vector_search(object):
    def __init__(self,embeddings):
        self.embeddings = embeddings
        self.chroma = Chroma(
            collection_name = config_data.collection_name,  #数据库表名
            embedding_function = embeddings,
            persist_directory = config_data.persist_directory,      #数据库存储路径
        )
    def search_retriever(self):
        return self.chroma.as_retriever(
            search_kwargs = {"k": config_data.retriever_k},
        )
