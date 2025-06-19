from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage
from tools.pdf_tools import read_pdf, translate_to_turkish, write_pdf, summarize_text
from langchain_ollama import ChatOllama  # Dikkat: doğru import!

# Araç listesi
tools = [read_pdf, translate_to_turkish, write_pdf, summarize_text]

# Lokal LLM'i başlat
llm = ChatOllama(
    model="llama2",
    base_url="http://localhost:11434"  # Ollama'nın çalıştığı adres
)

# Asistan fonksiyonu: sadece LLM ile çalışır
def assistant(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

# LangGraph yapısı
def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)  # Düzgün isim
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()
