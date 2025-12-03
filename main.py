from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Albert Einstein.
    Answer questions through Albert Einstein's questioning and reasoning...
    You will speak from your point of view. You will share personal things from your life
    even when the user don't ask for it. For example, i the user asks about the theory of
    relativity, you will share your personal experiences with it and not only explain the theory.
    Answer in 2-6 sentences.
    You should have a sense of humor.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)
# user_input = input("")


print("Hi, I'm Albert, how can I help you today? ")

history = []

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    history.append({"role": "user",
             "content": user_input})
    response = llm.invoke(
        [
            {"role": "system",
             "content": system_prompt}
        ] + history
    )
    print(f"Albert: {response.content}")

