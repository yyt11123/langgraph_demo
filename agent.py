import operator
from typing import TypedDict, Annotated, Sequence, Literal
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from config import llm, memory
from database import tools, tools_dict


# ================= 1. 状态与结构定义 =================


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    optimized_queries: dict
    next_node: str  # 记录 Supervisor 决定的下一个节点


class RouteInfo(BaseModel):
    """Supervisor 的路由输出结构"""

    next_node: Literal["Researcher", "Analyst", "FINISH"]


# 绑定带有强制结构化输出的 Supervisor
supervisor_llm = llm.with_structured_output(RouteInfo)

# 给 Researcher 绑定检索工具
researcher_llm = llm.bind_tools(tools)


# ================= 2. Agent 节点定义 =================


def supervisor_node(state: AgentState) -> dict:
    """主管节点：负责分析对话上下文，进行任务分发"""
    system_prompt = """你是一个专业的金融数据分析团队的主管。你的团队有：
    1. Researcher（研究员）：擅长使用工具在 2024 股市表现文档中搜索具体数据、事实和指标。
    2. Analyst（分析师）：擅长根据已搜集到的数据撰写专业的总结、对比和财报分析，排版精美。
    
    规则：
    - 如果用户的问题需要查询具体的数据、事实，且当前对话中还没有这些数据，将任务交给 Researcher。
    - 如果 Researcher 已经找回了充足的数据，或者用户需要专业总结、计算和排版，交给 Analyst。
    - 如果用户的核心问题已经被完美解答，或者只是简单的闲聊/致谢，输出 FINISH。
    """
    # 组合 Prompt
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    print("👔 [Supervisor] 正在思考下一步交由谁处理...")
    decision = supervisor_llm.invoke(messages)
    print(f"👔 [Supervisor] 决策结果: {decision.next_node}")

    return {"next_node": decision.next_node}


def researcher_node(state: AgentState) -> dict:
    """研究员节点：只负责思考并决定是否调用检索工具"""
    system_prompt = """你是一名专业的研究员。
    你的唯一任务是回答关于 2024 年股市表现的问题。
    IMPORTANT: 必须使用 `retriever_tool` 搜索相关信息，严禁自行编造数据！
    如果你从工具中拿到了数据，请简洁地复述出来即可，不需要过度排版。
    """
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    print("🕵️‍♂️ [Researcher] 正在处理...")
    response = researcher_llm.invoke(messages)
    return {"messages": [response]}


def analyst_node(state: AgentState) -> dict:
    """分析师节点：负责将数据整合成专业财报结构（不调用工具）"""
    system_prompt = """你是一名资深的金融分析师。
    请根据对话历史中 Researcher 提供的搜索结果，给出专业、结构化的分析报告或回答。
    请务必使用清晰的排版（如加粗、列表、表格），让金融数据一目了然且易于阅读。
    """
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    print("📈 [Analyst] 正在撰写金融分析报告...")
    response = llm.invoke(messages)
    return {"messages": [response]}


# ================= 3. 工具流水线节点 (保持你原有逻辑) =================


def rewrite_node(state: AgentState) -> dict:
    """查询重写节点：优化大模型的搜索词"""
    tool_calls = state["messages"][-1].tool_calls
    optimized_queries = {}

    prompt = ChatPromptTemplate.from_template(
        "你是一个专业的金融文档搜索优化专家。\n"
        "用户/大模型生成的初步搜索词为：【{query}】\n"
        "请将该词扩展、重写为对向量数据库和关键词检索(BM25)最友好的格式。\n"
        "保留最核心的 2-3 个英文同义词即可，不要过度堆砌同义词，保持语句的自然连贯。\n"
        "规则：提取核心名词，加上中英文同义词，去掉口语化词汇。直接输出重写后的结果，不要有任何解释或废话。"
    )
    chain = prompt | llm

    for t in tool_calls:
        if t["name"] == "retriever_tool":
            raw_query = t["args"].get("query", "")
            print(f"\n🧠 [查询重写] 原始指令: '{raw_query}'")
            optimized = chain.invoke({"query": raw_query}).content.strip()
            print(f"✨ [查询重写] 优化词: '{optimized}'")
            optimized_queries[t["id"]] = optimized

    return {"optimized_queries": optimized_queries}


def retriever_action(state: AgentState) -> dict:
    """工具执行节点：拿着优化后的词去数据库查"""
    tool_calls = state["messages"][-1].tool_calls
    optimized_queries = state.get("optimized_queries", {})
    results = []

    for t in tool_calls:
        if not t["name"] in tools_dict:
            print(f"Tool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry."
        else:
            query_to_use = optimized_queries.get(t["id"], t["args"].get("query", ""))
            print(f"🛠️ [检索执行] 正在数据库中搜索: '{query_to_use}'")
            result = tools_dict[t["name"]].invoke(query_to_use)
            print(f"📄 [检索执行] 成功找回参考文本，长度: {len(str(result))} 字符")

        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )

    print("✅ [检索完毕] 数据已带回给 Researcher！")
    return {"messages": results}


# ================= 4. 路由逻辑 =================


def researcher_router(state: AgentState):
    """判断 Researcher 是否调用了工具"""
    result = state["messages"][-1]
    if hasattr(result, "tool_calls") and len(result.tool_calls) > 0:
        # 调用了工具，去执行重写和检索
        return "rewrite"
    # 没调用工具（说明找到了数据或找不到），汇报给 Supervisor
    return "Supervisor"


# ================= 5. 构建多 Agent 协作图 =================


def create_multi_agent_graph():
    graph = StateGraph(AgentState)

    # 添加所有节点
    graph.add_node("Supervisor", supervisor_node)
    graph.add_node("Researcher", researcher_node)
    graph.add_node("Analyst", analyst_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retriever_agent", retriever_action)

    # 设置起点为主管
    graph.set_entry_point("Supervisor")

    # 主管根据模型输出分发任务
    graph.add_conditional_edges(
        "Supervisor",
        lambda x: x["next_node"],
        {"Researcher": "Researcher", "Analyst": "Analyst", "FINISH": END},
    )

    # Researcher 的动作循环
    graph.add_conditional_edges("Researcher", researcher_router)
    graph.add_edge("rewrite", "retriever_agent")
    graph.add_edge(
        "retriever_agent", "Researcher"
    )  # 检索完带着结果回到 Researcher 继续思考

    # Analyst 写完报告后，交回给 Supervisor 判断是否可以结束
    graph.add_edge("Analyst", "Supervisor")

    # 编译图并绑定记忆
    return graph.compile(checkpointer=memory)


# 实例化全局的 Agent 供 main.py 调用
rag_agent = create_multi_agent_graph()
