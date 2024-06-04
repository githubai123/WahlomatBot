import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_model_name = os.getenv("OPENAI_MODEL_NAME")

from app.crew import run_crew

st.title('Podcast Discussion Round with CrewAI and Llama 3')

# Input for user query
user_input = st.text_input("Enter your query:")

if user_input:
    # Run the crew
    result = run_crew(user_input)
    st.write("Task Results:")
    st.write(result)

