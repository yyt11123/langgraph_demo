import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from config import embeddings, PDF_PATH, PERSIST_DIRECTORY, COLLECTION_NAME

# === 基础检索导入 ===
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# === 重排序导入 ===
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever


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
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# [2. 关键词检索器]
print("正在构建 BM25 关键词索引...")
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

# [3. 混合检索器]
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)

# [4. 终极重排序器 (Reranker) ]
print("正在加载 BGE 重排模型")
# 实例化交叉编码器模型
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
# 设置重排规则：从混合检索捞出来的 20 个文档中，精选出得分最高的 Top 3
compressor = CrossEncoderReranker(model=model, top_n=3)

# [5. 组装终极检索器]
# 用压缩器包住混合检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=ensemble_retriever
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """
    print(f"\n🔍 [检索系统] 正在使用 混合检索+重排 搜索关键词: '{query}'...")

    # 使用组装好的终极重排检索器
    docs = compression_retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    results = []
    # 打印出重排后的结果和相关性得分
    for i, doc in enumerate(docs):
        # 别猜键名了，直接打印它所有的 metadata 看看里面有什么！
        print(f"🎯 [Debug] 文档 {i+1} 的元数据: {doc.metadata}")
        score = doc.metadata.get("relevance_score", "未找到分数")
        print(f"🎯 选中文档 {i+1} | 相关度评分: {score}")
        results.append(f"Document {i+1}:\n{doc.page_content}")

    return "\n\n".join(results)


# 导出 tools 供 Agent 使用
tools = [retriever_tool]
tools_dict = {our_tool.name: our_tool for our_tool in tools}
