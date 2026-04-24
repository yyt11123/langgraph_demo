import os
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# 代理设置
os.environ["NO_PROXY"] = "dashscope.aliyuncs.com"

# 全局路径和集合名称
PDF_PATH = "Stock_Market_Performance_2024.pdf"
PERSIST_DIRECTORY = r"C:\Vaibhav\LangGraph_Book\LangGraphCourse\Agents"
COLLECTION_NAME = "stock_market"

# 实例化记忆模块
memory = MemorySaver()

# 实例化 LLM
llm = ChatTongyi(model="qwen-plus", temperature=0)

# 实例化 Embedding 模型
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
