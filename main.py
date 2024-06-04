from typing import Dict, TypedDict, Optional

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import StateGraph,END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import SystemMessagePromptTemplate
import random
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatOpenAI(model_name="crewai-llama3:8b", temperature=0.8)

debate_topic = "Should Data Scientist write backend and API code as well?"
output =llm.predict_messages([HumanMessage(content="I wish to have a debate on {}.Do not write anything exept the names of the two fighting parties.".format(debate_topic))])
output_parser = CommaSeparatedListOutputParser()
classes = output_parser.parse(output.content)
#  Not really working with llama3
print(classes)
