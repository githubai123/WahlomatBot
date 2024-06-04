import streamlit as st
import os
from dotenv import load_dotenv
import re
import time
import sys

# Load environment variables from .env file
load_dotenv()

# Now you can access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_model_name = os.getenv("OPENAI_MODEL_NAME")
from app.crew import run_crew



# Input for user query
## MUSS ANGEPASST WERDEN Einfach nach https://discuss.streamlit.io/t/crewai-streamlit-ui/64793
#display the console processing on streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Market Research Analyst" in cleaned_data:
            # Apply different color 
            cleaned_data = cleaned_data.replace("Market Research Analyst", f":{self.colors[self.color_index]}[Market Research Analyst]")
        if "Business Development Consultant" in cleaned_data:
            cleaned_data = cleaned_data.replace("Business Development Consultant", f":{self.colors[self.color_index]}[Business Development Consultant]")
        if "Technology Expert" in cleaned_data:
            cleaned_data = cleaned_data.replace("Technology Expert", f":{self.colors[self.color_index]}[Technology Expert]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


st.title('Podcast Discussion Round with CrewAI and Llama 3')

# Input for user query
user_input = st.text_input("Enter your query:")

if user_input:
    # Run the crew

       # Placeholder for stopwatch
        stopwatch_placeholder = st.empty()
        
        # Start the stopwatch
        start_time = time.time()
        with st.expander("Processing!"):
            sys.stdout = StreamToExpander(st)
            with st.spinner("Generating Results"):
                crew_result = run_crew(user_input)

        # Stop the stopwatch
        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")

        st.header("Tasks:")

        st.header("Results:")
        st.markdown(crew_result)
