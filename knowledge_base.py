import config_data
import os
import hashlib
from langchain_chroma import Chroma # 使用Gitee AI Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import dotenv
from langchain_community.embeddings import DashScopeEmbeddings
dotenv.load_dotenv()

def check_md5(md5_str:str):    #检查是否已经存在md5值
    """
    return False 表示没有存md5
    return True 表示已经存在md5
    """
    #检查是否有md5文件
    if not os.path.exists(config_data.md5_file_path):
        open(config_data.md5_file_path,"w",encoding="utf-8").close()
        return False
    else:
        with open(config_data.md5_file_path,"r",encoding="utf-8") as f:
            for line in f:
                if line.strip() == md5_str:
                    return True
            return False

        
def save_md5(md5_str:str):     #保存md5值
    with open(config_data.md5_file_path,"a",encoding="utf-8") as f:
        f.write(md5_str+"\n")

def get_md5(text:str):      #将字符串转换为md5值
    md5 = hashlib.md5()
    md5.update(text.encode("utf-8"))
    return md5.hexdigest()
    
class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config_data.persist_directory, exist_ok=True)  #如果文件不存在则创建，如果存在则跳过
        try:
            print("初始化Embeddings...")
            embeddings = DashScopeEmbeddings(
                model = "text-embedding-v4",
                dashscope_api_key=os.getenv("DashScope_api_key")
            )
            print("Gitee AI Embeddings初始化成功")
            
            self.chroma = Chroma(
                collection_name = config_data.collection_name,  #数据库表名
                embedding_function = embeddings,
                persist_directory = config_data.persist_directory,      #数据库存储路径
            )
            self.spliter = RecursiveCharacterTextSplitter(
                chunk_size = config_data.chunk_size,
                chunk_overlap = config_data.chunk_overlap,
                separators = config_data.separators,  
                length_function = len,              #使用len函数作为长度计算函数
            )
            print("初始化数据库成功")
        except Exception as e:
            print(e)
    def upload_by_str(self,text:str,file_name):
        md5_str = get_md5(text)
        if check_md5(md5_str):
            return "该文本已存在"
        if len(text) < config_data.mini_size:
            docs = [text]
        else:
            docs = self.spliter.split_text(text)
        print(f"长度{len(docs)}")
        metadata = {
            "source":file_name,
            "create_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator":"小李",
        }
        print("这里成功了")
        
        self.chroma.add_texts(
                docs,
                metadatas = [metadata for _ in docs],
            )
        print("存储成功")
        save_md5(md5_str)
        return "上传成功"        

# if __name__ == "__main__":
#     knowledge_base_service = KnowledgeBaseService()
#     res = knowledge_base_service.upload_by_str("去你的","testfile")
#     print(res)

