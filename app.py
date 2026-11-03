"""Web chat UI for the Bright Studio assistant.  Run: python app.py  ->  http://127.0.0.1:7860"""
import gradio as gr

from ask import stream_answer

EXAMPLES = [
    "What packages do you offer and how much do they cost?",
    "Who should I talk to about branding?",
    "How do I cancel my contract?",
    "What results did you get for PeakFit Gym?",
    "Do you run Google Ads?",
]

CSS = """
.gradio-container {max-width: 820px !important; margin: auto;}
#header {text-align: center; padding: 8px 0 0;}
#header h1 {margin-bottom: 0;}
#header p {opacity: .7; margin-top: 4px;}
"""


def respond(message, history):
    # Gradio gives history as [{"role","content"}, ...]; keep only plain-text content.
    clean = [{"role": h["role"], "content": h["content"]} for h in history if isinstance(h["content"], str)]
    answer, hits = "", []
    try:
        for answer, hits in stream_answer(message, clean):
            yield answer
    except Exception as e:  # Ollama down, model missing, etc.
        yield f"⚠️ Sorry, I couldn't reach the language model ({type(e).__name__}). Is Ollama running?"
        return
    sources = sorted({m["source"] for _, m, _ in hits})
    if sources:
        yield answer + "\n\n---\n📄 *Sources searched: " + ", ".join(sources) + "*"


with gr.Blocks(title="Bright Studio Assistant") as demo:
    gr.HTML(
        "<div id='header'><h1>✨ Bright Studio Assistant</h1>"
        "<p>Ask about our services, pricing, team, and how we work.</p></div>"
    )
    gr.ChatInterface(
        respond,
        examples=EXAMPLES,
        chatbot=gr.Chatbot(height=480, placeholder="Hi! Ask me anything about Bright Studio."),
        textbox=gr.Textbox(placeholder="Type your question…", container=False, scale=7),
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
