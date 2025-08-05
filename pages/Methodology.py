import streamlit as st
import graphviz

st.title("Methodology")

st.header("Process Flowchart")

# Create a new directed graph
graph = graphviz.Digraph()

# Add nodes for each step in the process
graph.node("A", "User uploads PDF(s)")
graph.node("B", "Extract text using OCR")
graph.node("C", "Display extracted text")
graph.node("D", "Summarize the text")
graph.node("E", "Display summary")
graph.node("F", "Chat with documents (RAG)")

# Add edges to show the flow
graph.edge("A", "B")
graph.edge("B", "C")
graph.edge("C", "D")
graph.edge("D", "E")
graph.edge("E", "F")

# Display the flowchart
st.graphviz_chart(graph)
