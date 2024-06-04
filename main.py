from typing import Dict, TypedDict, Optional

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import StateGraph,END
from langchain_openai import ChatOpenAI
import random
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(model_name="crewai-llama3:8b", temperature=0.7)

debate_topic = "Should Data Scientist write backend and API code as well?"

output =llm("I which to have a debate on {}. What would be the fighting sides. Output just the names")
output_parser = CommaSeparatedListOutputParser()
classes = output_parser.parse(output)
print(classes)