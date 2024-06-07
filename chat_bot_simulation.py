from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.adapters.openai import convert_message_to_dict
from langchain_openai import ChatOpenAI
import openai
model_chat_bot= ChatOpenAI(
    model_name="crewai-llama3:8b",
    temperature=0.7,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="0"
)
# Define the user model
model_user = ChatOpenAI(
    model_name="crewai-llama3:8b",
    temperature=0.7,
    openai_api_base="http://localhost:11434/v1",
    openai_api_key="0"
)

# Define the chat bot function
def my_chat_bot(messages: List[Dict]) -> Dict:
    system_message = {
        "role": "system",
        "content": "You are a customer support agent for an airline.",
    }
    messages = [system_message] + messages
    completion = model_chat_bot.invoke(input=messages)
    return completion

# Define the system prompt template
system_prompt_template = """You are a customer of an airline company. \
You are interacting with a user who is a customer support person. \

{instructions}

When you are finished with the conversation, respond with a single word 'FINISHED'"""

# Create the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
instructions = """Your name is Harrison. You are trying to get a refund for the trip you took to Alaska. \
You want them to give you ALL the money back. \
This trip happened 5 years ago."""

prompt = prompt.partial(name="Harrison", instructions=instructions)

# Create the simulated user
simulated_user = prompt | model_user

# Define the initial messages
messages = [HumanMessage(content="Hi! How can I help you?")]
print(simulated_user.invoke({"messages": messages}))

# Define the chat bot node
def chat_bot_node(messages):
    # Convert from LangChain format to the OpenAI format, which our chatbot function expects.
    messages = [convert_message_to_dict(m) for m in messages]
    # Call the chat bot
    chat_bot_response = my_chat_bot(messages)
    # Respond with an AI Message
    print(type(chat_bot_response))
    
    return chat_bot_response

# Define the simulated user node
def _swap_roles(messages):
    new_messages = []
    for m in messages:
        if isinstance(m, AIMessage):
            new_messages.append(HumanMessage(content=m.content))
        else:
            new_messages.append(AIMessage(content=m.content))
    return new_messages

def simulated_user_node(messages):
    # Swap roles of messages
    new_messages = _swap_roles(messages)
    # Call the simulated user
    response = simulated_user.invoke({"messages": new_messages})
    # This response is an AI message - we need to flip this to be a human message
    return HumanMessage(content=response.content)

# Define the edge logic
def should_continue(messages):
    if len(messages) > 6:
        return "end"
    elif messages[-1].content == "FINISHED":
        return "end"
    else:
        return "continue"

# Build the graph
from langgraph.graph import END, MessageGraph

graph_builder = MessageGraph()
graph_builder.add_node("user", simulated_user_node)
graph_builder.add_node("chat_bot", chat_bot_node)
# Every response from your chat bot will automatically go to the simulated user
graph_builder.add_edge("chat_bot", "user")
graph_builder.add_conditional_edges(
    "user",
    should_continue,
    # If the finish criteria are met, we will stop the simulation,
    # otherwise, the virtual user's message will be sent to your chat bot
    {
        "end": END,
        "continue": "chat_bot",
    },
)
# The input will first go to your chat bot
graph_builder.set_entry_point("chat_bot")
simulation = graph_builder.compile()

# Run the simulation
for chunk in simulation.stream([]):
    # Print out all events aside from the final end chunk
    if END not in chunk:
        print(chunk)
        print("----")
