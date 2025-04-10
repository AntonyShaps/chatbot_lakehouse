import gradio as gr
import requests

API_URL = "http://localhost:8000/ask"  # Adjust if running on a different host/port

def query_docs(question):
    try:
        response = requests.post(API_URL, json={"question": question})
        data = response.json()

        if "error" in data:
            return f"❌ Error: {data['error']}"

        answer = data.get("answer", "")
        sources = data.get("sources", [])
        sources_formatted = "\n".join([f"- {url}" for url in sources])

        return f"🧠 **Answer:**\n\n{answer}\n\n📚 **Sources:**\n{sources_formatted}"

    except Exception as e:
        return f"❌ Request failed: {str(e)}"

demo = gr.Interface(
    fn=query_docs,
    inputs=gr.Textbox(label="Ask a question about Bitmovin documentation:", lines=2, placeholder="e.g. How to mute a webhook?"),
    outputs=gr.Markdown(label="Answer"),
    title="📘 Bitmovin Docs Chatbot",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()
