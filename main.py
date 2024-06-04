from typing import Dict, TypedDict, Optional

from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langgraph.graph import StateGraph, END
from langchain.llms import Ollama
import openai
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import SystemMessagePromptTemplate
import random
import time
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_base = "http://localhost:11434/v1"
openai.api_key = "YOUR_OLLAMA_API_KEY"

llm = Ollama(
    model="crewai-llama3:8b",
    temperature=0.7
)

debate_topic = "Should Data Scientist write backend and API code as well?"
output = llm(
    "I wish to have a debate on {}. What would be the fighting sides called? Output just the names and nothing else as comma separated list".format(
        debate_topic))
output_parser = CommaSeparatedListOutputParser()

print(output)
classes = output_parser.parse(output)
#  Not really working with llama3


# classes = ['Data Scientists', 'Full-Stack Developers']
class GraphState(TypedDict):
    classification: Optional[str]
    history: Optional[str]
    current_response: Optional[str]
    count: Optional[int]
    result: Optional[str]
    greeting: Optional[str]


workflow = StateGraph(GraphState)

prefix_start = 'You are in support of {}. You are in a debate with {} over the topic: {}. This is the conversation so far:\n{}\n. Put forth your next argument to support {} countering {}. Don’t repeat your previous arguments. Give a short, one line answer.'


def classify(question):
    return llm("classify the sentiment of input as {} or {}. Output just the class. Input:{}".format(
        '_'.join(classes[0].split(' ')), '_'.join(classes[1].split(' ')), question)).strip()


def classify_input_node(state):
    question = state.get('current_response')
    classification = classify(question)  # Assume a function that classifies the input
    return {"classification": classification}


def handle_greeting_node(state):
    return {"greeting": "Hello! Today we will witness the fight between {} vs {}".format(classes[0], classes[1])}


def handle_pro(state):
    summary = state.get('history', '').strip()
    current_response = state.get('current_response', '').strip()
    if summary == 'Nothing':
        prompt = prefix_start.format(classes[0], classes[1], debate_topic, 'Nothing', classes[0], "Nothing")
        argument = classes[0] + ":" + llm(prompt)
        summary = 'START\n'
    else:
        prompt = prefix_start.format(classes[0], classes[1], debate_topic, summary, classes[0], current_response)
        argument = classes[0] + ":" + llm(prompt)
    return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count') + 1}


def handle_opp(state):
    summary = state.get('history', '').strip()
    current_response = state.get('current_response', '').strip()
    prompt = prefix_start.format(classes[1], classes[0], debate_topic, summary, classes[1], current_response)
    argument = classes[1] + ":" + llm(prompt)
    return {"history": summary + '\n' + argument, "current_response": argument, "count": state.get('count') + 1}


def result_node(state):
    summary = state.get('history').strip()
    prompt = "Summarize the conversation and judge who won the debate. No ties are allowed. Conversation:{}".format(
        summary)
    return {"result": llm(prompt)}


workflow.add_node("classify_input", classify_input_node)
workflow.add_node("handle_greeting", handle_greeting_node)
workflow.add_node("handle_pro", handle_pro)
workflow.add_node("handle_opp", handle_opp)
workflow.add_node("result_node", result_node)


def decide_next_node(state):
    return "handle_opp" if state.get('classification') == '_'.join(classes[0].split(' ')) else "handle_pro"


def check_conv_length(state):
    return "result_node" if state.get("count") == 10 else "classify_input"


workflow.add_conditional_edges(
    "classify_input",
    decide_next_node,
    {
        "handle_pro": "handle_pro",
        "handle_opp": "handle_opp"
    }
)

workflow.add_conditional_edges(
    "handle_pro",
    check_conv_length,
    {
        "result_node": "result_node",
        "classify_input": "classify_input"
    }
)

workflow.add_conditional_edges(
    "handle_opp",
    check_conv_length,
    {
        "result_node": "result_node",
        "classify_input": "classify_input"
    }
)

workflow.set_entry_point("handle_greeting")
workflow.add_edge('handle_greeting', "handle_pro")
workflow.add_edge('result_node', END)

app = workflow.compile()
conversation = app.invoke({'count': 0, 'history': 'Nothing', 'current_response': ''})

print(conversation['history'])
print(conversation["result"])
