# 📝 PDF Translator & Summarizer with LangGraph

## 📌 Overview

This project is a LangGraph-powered AI Agent that can:

- ✅ Translate English academic PDFs into Turkish
- ✅ Generate a concise summary of the uploaded PDF
- ✅ Save translated content as a new PDF
- ✅ Run via a simple Gradio interface

Built using `LangGraph`, `LangChain`, `Groq LLM`, and `Gradio`.

---

## 🎯 Goal

The core goal is to develop a modular LangGraph Agent capable of performing real-world tasks involving:

- PDF processing
- Multilingual translation
- Summarization
- File writing
- Human-interpretable tool usage

This system is an early step in building a general-purpose AI agent that could perform well on the [GAIA benchmark](https://huggingface.co/spaces/GAIA-benchmark/gaia).

---

## 🛠️ Tech Stack

| Feature             | Stack/Tool                 |
|--------------------|----------------------------|
| Language Model      | Groq LLM (Mixtral 8x7b)    |
| Agent Framework     | LangGraph                  |
| Tool Invocation     | LangChain Tools            |
| UI                  | Gradio                     |
| PDF Handling        | PyPDF2 + fpdf              |

---

## 🧠 Agent Topology

```mermaid
flowchart TD
    Start --> Retriever["PDF Reader"]
    Retriever --> Assistant["LangGraph Agent"]
    Assistant -->|Calls Tool| ToolNode["Tool Executor"]
    ToolNode --> Assistant
    Assistant --> End["Translated or Summarized Output"]
