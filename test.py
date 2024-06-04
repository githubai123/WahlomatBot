import openai
from langchain.llms import Ollama

# Set the API base and key globally for the openai package
openai.api_base = "http://localhost:11434/v1"  # Replace with your server URL
openai.api_key = "TEST"  # Replace with your actual API key

# Initialize the OpenAI class
llm = Ollama(
    model="crewai-llama3:8b"
)

# Define your prompt
prompt = "Explain the concept of machine learning."

# Send the prompt to the model and get the response
response = llm(prompt)

# Print the response from the model
print(response)
