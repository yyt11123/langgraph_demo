import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_core.messages import HumanMessage, AIMessage
from agent import rag_agent


# 对话的记忆配置
run_config = {"configurable": {"thread_id": "group_meeting_demo_01"}}


def ask_agent(question: str):
    print(f"🙋‍♂️ User: {question}")

    # 将用户问题打包并带上配置（记忆）传给 Agent
    messages = [HumanMessage(content=question)]
    result = rag_agent.invoke({"messages": messages}, config=run_config)

    print("\n🤖 AI Assistant:")
    print(result["messages"][-1].content)
    print("=" * 60)
    # 遍历打印所有的 AI 纯文本回复（过滤掉调用工具的中间思考过程）
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            print(m.content)
            print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    # 运行你的测试样例
    ask_agent(
        "2024年美国股市的整体表现如何？纳斯达克指数和标普500指数的涨幅分别是多少？请在总结后，用 Python 画一个精美的柱状图对比一下这两个指数的涨幅。"
    )
