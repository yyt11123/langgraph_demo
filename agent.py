from typing import TypedDict, Annotated, Sequence
from operator import add as add_messages
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from config import llm, memory
from database import tools, tools_dict

# 绑定工具到大模型
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    optimized_queries: dict


def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state."""
    messages = list(state["messages"])
    # 确保系统提示词在最前面
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_prompt)] + messages

    message = llm_with_tools.invoke(messages)
    return {"messages": [message]}


def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024.
IMPORTANT: The PDF document has ALREADY been processed and loaded into your database. 
You CANNOT "see" the file directly, so you MUST use the `retriever_tool` to search for any information related to the user's query.
Never ask the user to upload a document. If you don't know the answer, use the tool to search first.
Please always cite the specific parts of the documents you use in your answers.
"""


def rewrite_node(state: AgentState) -> AgentState:
    """独立节点：拦截大模型的指令，调用 LLM 重写搜索词并存入 State"""
    tool_calls = state["messages"][-1].tool_calls
    optimized_queries = {}

    # 1. 直接在节点内定义 Prompt 和 Chain
    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的金融文档搜索优化专家。\n"
        "用户/大模型生成的初步搜索词为：【{query}】\n"
        "请将该词扩展、重写为对向量数据库和关键词检索(BM25)最友好的格式。\n"
        "规则：提取核心名词，加上中英文同义词，去掉口语化词汇。直接输出重写后的结果，不要有任何解释或废话。"
    )
    chain = prompt | llm

    # 2. 遍历工具调用，进行洗稿
    for t in tool_calls:
        if t["name"] == "retriever_tool":
            raw_query = t["args"].get("query", "")
            print(f"\n🧠 [查询重写节点] 收到大模型的原始指令: '{raw_query}'")

            # 3. 直接在这里调用 LLM 进行翻译
            optimized = chain.invoke({"query": raw_query}).content.strip()
            print(f"✨ [查询重写节点] 输出优化后的搜索词: '{optimized}'")

            # 4. 以 tool_call_id 为键，存起来
            optimized_queries[t["id"]] = optimized

    # 将这个字典写回 State，传给下一个节点
    return {"optimized_queries": optimized_queries}


def retriever_action(state: AgentState) -> AgentState:
    """独立节点：从 State 拿出优化好的词去执行搜索"""
    tool_calls = state["messages"][-1].tool_calls

    # 拿出上一个节点（rewrite_node）存进来的优化词字典，如果没有就给个空字典
    optimized_queries = state.get("optimized_queries", {})
    results = []

    for t in tool_calls:
        if not t["name"] in tools_dict:
            print(f"Tool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry."
        else:
            # 去字典里拿优化好的词，如果意外没拿到，就用大模型的原始词兜底
            query_to_use = optimized_queries.get(t["id"], t["args"].get("query", ""))
            print(f"🛠️ [检索节点] 正在拿词条去数据库里搜: '{query_to_use}'")

            result = tools_dict[t["name"]].invoke(query_to_use)
            print(f"📄 [检索返回文本长度]: {len(str(result))} 字符")

        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )

    print("✅ 工具执行完毕，资料已带回给大脑！")
    return {"messages": results}


# 构建图的逻辑封装在函数中
def create_rag_agent():
    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", retriever_action)
    graph.add_node("rewrite", rewrite_node)

    graph.add_conditional_edges("llm", should_continue, {True: "rewrite", False: END})
    graph.add_edge("rewrite", "retriever_agent")
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")

    # 返回编译后的 Agent
    return graph.compile(checkpointer=memory)


# 实例化 Agent
rag_agent = create_rag_agent()
