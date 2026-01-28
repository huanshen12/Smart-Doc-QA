from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
import os
import dotenv
dotenv.load_dotenv()

md5_file_path = "./md5.txt"
collection_name = "rag"
persist_directory = "./chroma_db"
chunk_size = 500
chunk_overlap = 50
separators = ["\n\n", "\n", " ", "",".",",","?","。","，","？"]
mini_size = 500
retriever_k = 1
llm = ChatOpenAI(
    model = "ep-20251122233041-rpp9j",
    api_key=os.getenv("DeepSeek_api_key"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

DashScopeEmbeddings = DashScopeEmbeddings(
    model = "text-embedding-v4",
    dashscope_api_key=os.getenv("DashScope_api_key")
)
config_session = {
        "configurable":{
            "session_id":"user002",
        }
    }