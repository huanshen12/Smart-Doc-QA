import streamlit as st
import time
from rag import RagService
from config_data import config_session



st.title("智能知识客服")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"你好！有什么我可以帮助你的吗？"}]

if "rag" not in st.session_state:
    st.session_state["rag"]= RagService()
user_input = st.chat_input("请输入你的问题")

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role":"user","content":user_input})

    text = st.session_state["rag"].chain.stream({"question":user_input},config_session)
    list_text = []
    def generate_text_stream(text,list_text):
        for chunk in text:
            list_text.append(chunk)
            yield chunk

    with st.spinner("思考中..."):
        # time.sleep(2)
        st.chat_message("assistant").write_stream(generate_text_stream(text,list_text))
        st.session_state.messages.append({"role":"assistant","content":"".join(list_text)})

