import os
import re
import langchain as lc

# Set up API keys
groq_key = "UPDATE_YOUR_GROQ_API_KEY"
gemini_key = "UPDATE_YOUR_GEMINI_API_KEY"

if not gemini_key or not groq_key:
    raise ValueError("API keys not found in environment variables")

os.environ["GEMINI_API_KEY"] = gemini_key
os.environ["GROQ_API_KEY"] = groq_key

# Initialize LangChain client
client = lc.Client()

# Define models
models = ["groq:llama-3.2-3b-preview"]

# Define character prompt
character = "請用金庸的口吻, 也就是用武俠小說的敘事, 以第一人稱、小說家的觀點說一次, 說為什麼這是一件武林軼事, 並且以「這就是江湖」結尾。"

def generate_responses(prompt):
    messages = [
        {"role": "system", "content": character},
        {"role": "user", "content": prompt},
    ]
    #TODO
    pass

def main():
    prompt = input("請用一句話說明你今天的心情: ")
    generate_responses(prompt)

if __name__ == "__main__":
    main()