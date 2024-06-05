from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from .rag_tools import HelperTools
from langgraph.graph import StateGraph, END
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage

# Define the language model
model = ChatOpenAI(model_name="crewai-llama3:8b", temperature=0.7)

# Define tools
helpers =HelperTools()
search_tool = helpers.get_serper_search_tool()

# define Graph structure to coordinate the discussion
graph = MessageGraph()

pdf_tools = {}
parties = ["CDU", "FDP"]
for party in parties:
    pdf_tools[party] = helpers.get_tool_pdf_rag_party(party)

print(f"Number of tools {len(pdf_tools)}.")









def create_agents(topic):
    moderator = Agent(
        role=f'Discussion host',
        goal=f'Create a fruitful discussion on the topic: {topic} between members of different political parties and come to a common verdict.',
        backstory=f"""You are an experienced discussion host with deep insight in the topic:  {topic}.
        Your goal is to disect the topic and break down the pros and cons and the outline the different political positions.
        You enjoy challenges and strive to make complex concepts accessible.You moderate and even interrupt the participants of this discussion 
        to ensure a well balanced discussion.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=model
    )
    graph.add_node("moderator", moderator.llm)

    politicans = {}
    for party in parties:
        politicans[party] = Agent(
            role='Discussion participant',
            goal=f'Outline and promote the position of the german political party {party} on the topic: {topic}. You can react on the answers of other participants.',
            backstory=f'You are a politician with a strong background in {party}.',
            verbose=True,
            allow_delegation=False,
            tools=[pdf_tools[party]],
            llm=model
        )

    # Create a task for the teacher to answer questions
    introduction = Task(
        description=f"Welcome the audience and welcome the discussion participants. Then give a neutral overview on the topic of todays"
                    f"discussion {topic}. Summarize in <200 words. Also do it in a friendly and funny way.",
        expected_output="Introduction and welcome",
        agent=moderator
    )
    position_statements = {}
    for party in parties:
        position_statements[party] = Task(
            description=f"Give a short but precise summary on your opinion on the topic {topic}",
            expected_output=f"opinion on the topic {topic}",
            agent=politicans[party]
        )

    summarize = Task(
        description=f"Summarise  the different positions in less than 500 words. And ask the participants on their counter or pro aruments to the different positions",
        expected_output=f"concise summary of the different position",
        agent=moderator
    )

    reactions = {}
    for party in parties:
        reactions[party] = Task(
            description=f"Respond with the {party} politician perspective on {topic}",
            expected_output=f"A response promoting the {party} politicians arguments reaction and attacking the position of his opponent",
            agent=politicans[party]
        )

    conclude_discussion = Task(
        description=f"Summarise  the different positions in less than 500 words. And ask the participants on their counter or pro aruments to the different positions",
        expected_output=f"Verdict on the topic of the discussion and summary of the main positions and their differences.",

        agent=moderator
    )

    return([moderator] + list(politicans.values()), [introduction] + list(position_statements.values()) + [summarize] + list(reactions.values()) + [conclude_discussion])

def perform_task_with_tools(agent, task_description, conversation_history):
    # Use the PDFSearchTool to retrieve relevant information
    search_results = agent.tools[0].search(task_description)
    
    # Combine the backstory and conversation history with the search results
    context_str = f"Backstory: {agent.backstory}\n\nConversation History:\n{conversation_history}\n\nSearch Results:\n" + "\n".join(search_results)
    prompt = f"Using the following context, respond to the task: {task_description}\n\nContext:\n{context_str}\n\nResponse:"
    response = agent.llm.generate(prompt)
    
    return response

def run_crew(discussion_topic):
    agents,tasks = create_agents(discussion_topic)
    # Instantiate the crew with the updated tasks
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=2  # Set verbosity level for logging
    )
    

    conversation_history = ""
# Run the crew and integrate with LangGraph
    for task in crew.tasks:
        if task.agent.role == 'Discussion participant':
            response = perform_task_with_tools(task.agent, task.description, conversation_history)
            conversation_history += f"{task.agent.backstory}: {response}\n"
            print(f"{task.agent.backstory}: {response}")
        elif task.agent.role == 'Discussion host':
            # Simplified logic for deciding the winner
            response1 = perform_task_with_tools(task1.agent, task1.description, conversation_history)
            response2 = perform_task_with_tools(task2.agent, task2.description, conversation_history)
            
            if "better argument" in response1:  # Placeholder for actual evaluation logic
                print(f"Observer: The advocate had the better arguments.")
            else:
                print(f"Observer: The opponent had the better arguments.")



     # Run the crew
    
    #result = crew.kickoff()
    return result

