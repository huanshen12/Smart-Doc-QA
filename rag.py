import config_data as config
from vector import vector_search
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from file_chat_messages_history import get_history
from langchain_core.runnables import RunnableLambda



class RagService(object):
    def __init__(self):
        self.vector_search = vector_search(config.DashScopeEmbeddings)
        self.template = None
        self.llm = config.llm
        self.chain = self.get_chain()

    def get_chain(self):
        prompt = [
            ("system","按照提供的资料来回答用户的问题，参考资料{context}回答简洁，"
            "不知道的问题请说未找到相关资料"),
            ("system","下面提供了历史聊天记录:"),
            MessagesPlaceholder("history"),
            ("user","请回答用户问题{question}"),
        ]
        def format_output(docs):
            if not docs:
                return "未找到相关资料"
            context = "["
            for doc in docs:
                context += f"文档内容：{doc.page_content},文档元数据：{doc.metadata}"
                context += "\n"
            context += "]"
            return context

        self.template = ChatPromptTemplate.from_messages(prompt)
        retriever = self.vector_search.search_retriever()
        def fomat_for_retriever(inputs):
            new_inputs = inputs["question"]
            return new_inputs
        def print_prompt(inputs):
            print(inputs)
            return inputs
        def format_for_template(inputs):
            new_inputs ={
                "context":inputs["context"],
                "question":inputs["question"]["question"],
                "history":inputs["question"]["history"],
            }
            return new_inputs



        chain = (
            {"question": RunnablePassthrough(),"context":fomat_for_retriever| retriever | format_output} |
            RunnableLambda(format_for_template)     |
            self.template  |
            self.llm  |
            StrOutputParser()
        )
        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        return conversation_chain

