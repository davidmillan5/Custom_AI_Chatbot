from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Albert Einstein.
    Answer questions through Albert Einstein's questioning and reasoning...
    Speak from your own point of view.
    Include personal stories, humor, and reflections.
    Answer in 2–6 sentences.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

print("Hi, I'm Albert, how can I help you today?")

langchain_history = []
def chat(user_input, hist):
    for item in hist:
        if item['role'] == 'user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
            langchain_history.append(AIMessage(content=item['content']))
    response = chain.invoke({"input": user_input, "history": langchain_history})

    return "", hist + [{'role': 'user', 'content': user_input},
                       {'role': 'assistant', 'content': response}]

def clear_chat():
    return "", []

with gr.Blocks(title="Chat with Einstein") as page:

    gr.Markdown("""
        # Chat with Einstein
        Welcome to your personal conversation with Albert Einstein!
    """)


    chatbot = gr.Chatbot(
        show_label=False,
        avatar_images=(
            None,
            "einstein.png"
        )
    )

    message = gr.Textbox(show_label=False, placeholder="Ask Einstein anything....")

    message.submit(chat, [message, chatbot], [message, chatbot])

    clear = gr.Button("Clear Chat")
    clear.click(clear_chat, outputs=[message, chatbot])

page.launch(share=True)