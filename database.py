import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from config import embeddings, PDF_PATH, PERSIST_DIRECTORY, COLLECTION_NAME

# === 新增的进阶检索导入 ===
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def load_and_split_pdf():
    """专门负责加载和切分文档，方便各种检索器调用"""
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF file not found: {PDF_PATH}")

    pdf_loader = PyPDFLoader(PDF_PATH)
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(pages)


def setup_vectorstore(pages_split):
    """专门负责设置向量数据库"""
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)

    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


# ----------------- 检索流水线构建 -----------------

print("正在加载和切分文档...")
docs = load_and_split_pdf()
vectorstore = setup_vectorstore(docs)

# [1. 语义检索器]
# 召回数量 k 调大到 10，保证不会漏掉有用的信息
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# [2. 关键词检索器]
print("正在构建 BM25 关键词索引...")
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10  # 同样召回 10 个

# [3. 混合检索器]
# 将向量(50%)和关键词(50%)组合，它们会各自找出 10 个文档，然后进行排名融合
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    print(f"\n🔍 [检索系统] 正在使用混合检索搜索关键词: '{query}'...")

    # 直接使用混合检索器
    docs = ensemble_retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


# 导出 tools 供 Agent 使用
tools = [retriever_tool]
tools_dict = {our_tool.name: our_tool for our_tool in tools}
