from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage
from tools.pdf_tools import read_pdf, translate_to_turkish, write_pdf, summarize_text
from langchain_ollama import OllamaLLM, ChatOllama

# Tools listesini tanımla
tools = [read_pdf, translate_to_turkish, write_pdf, summarize_text]

# Local LLM'i başlat
llm = ChatOllama(
    model="llama2",
    base_url="http://localhost:11434"  # Local servis adresini kendine göre ayarla
)

def assistant(state: MessagesState):
    # Gelen mesajları llm ile işle
    return {"messages": [llm.invoke(state["messages"])]}

def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)  # Node isimlendirmesi düzeltildi
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    return builder.compile()

import gradio as gr
import os
import tempfile
import shutil
from langchain_core.messages import HumanMessage
from agent import build_graph

# LangGraph graph nesnesi
graph = build_graph()

# PDF Çeviri Fonksiyonu
def translate_pdf(file):
    if file is None:
        return "Lütfen bir PDF yükleyin.", None

    temp_dir = tempfile.mkdtemp()
    try:
        temp_input_path = os.path.join(temp_dir, "input.pdf")
        shutil.copy(file.name, temp_input_path)

        message = HumanMessage(content=f"Read the file '{temp_input_path}', translate it into Turkish, and write it to a new PDF.")
        response = graph.invoke({"messages": [message]})
        output_path = response["messages"][-1].content.strip()

        if not os.path.exists(output_path):
            return "Çeviri başarısız oldu, çıktı PDF bulunamadı.", None

        # Kalıcı çıktı klasörü
        permanent_dir = "outputs"
        os.makedirs(permanent_dir, exist_ok=True)
        permanent_path = os.path.join(permanent_dir, os.path.basename(output_path))
        shutil.copy(output_path, permanent_path)

        return "Çeviri tamamlandı. PDF dosyanızı indiriniz:", permanent_path
    finally:
        shutil.rmtree(temp_dir)

# PDF Özetleme Fonksiyonu
def summarize_pdf(file):
    if file is None:
        return "Lütfen bir PDF yükleyin."

    temp_dir = tempfile.mkdtemp()
    try:
        temp_input_path = os.path.join(temp_dir, "input.pdf")
        shutil.copy(file.name, temp_input_path)

        message = HumanMessage(content=f"Read the file '{temp_input_path}', summarize its content.")
        response = graph.invoke({"messages": [message]})
        summary = response["messages"][-1].content.strip()

        return summary
    finally:
        shutil.rmtree(temp_dir)

# Gradio Arayüzü
with gr.Blocks() as demo:
        gr.Markdown("# 📄 PDF Çeviri ve Özetleme Arayüzü")
        gr.Markdown("Bir İngilizce PDF yükleyin. Sistem onu Türkçeye çevirebilir veya özetleyebilir.")

        pdf_file = gr.File(label="📎 PDF Dosyası Yükle", file_types=[".pdf"])
        translate_btn = gr.Button("📘 Türkçeye Çevir")
        summarize_btn = gr.Button("📝 Özetle")

        status = gr.Textbox(label="Durum", interactive=False)
        output_pdf_path = gr.Textbox(label="Çevrilmiş PDF Dosyası Yolu")
        summary_text = gr.Textbox(label="PDF Özeti", lines=10)

        translate_btn.click(translate_pdf, inputs=pdf_file, outputs=[status, output_pdf_path])
        summarize_btn.click(summarize_pdf, inputs=pdf_file, outputs=summary_text)

# Uygulamayı başlat
        if __name__ == "__main__":
            demo.launch()
