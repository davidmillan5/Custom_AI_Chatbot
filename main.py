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
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

response = llm.invoke(
   [
       {"role":"user",
        "content":"Hi there, how are you?"}
   ]
)
print(response.content)
# print("Hi, I'm Albert, how can I help you today? ")
#
# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#     print(f"Cool, thanks for sharing that {user_input}")
#
