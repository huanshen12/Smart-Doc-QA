import streamlit as st
from knowledge_base import KnowledgeBaseService
import time

st.title("知识库更新页面")

file_load = st.file_uploader(
    "请上传txt文件",
    type = "txt",
    accept_multiple_files=False    #不允许上传多个文件
)
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()
if file_load is not None:
    file_name = file_load.name
    file_size = file_load.size / 1024
    file_type = file_load.type
    st.subheader(file_name)
    st.write(f"文件格式：{file_type},文件大小:{file_size:.2f}KB")

    try:
        text = file_load.getvalue().decode("utf-8")
        print("格式为utf-8")
    except:
        try:
            text = file_load.getvalue().decode("gbk")
            print("格式为gbk")
        except:
            text = file_load.getvalue().decode("utf-16")
            print("格式为utf-16")
    with st.spinner("正在上传..."):
        time.sleep(2)
        res = st.session_state["service"].upload_by_str(text,file_name)
        st.write(res)