from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool, PDFSearchTool

from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

# Define the language model
model = ChatOpenAI(model_name="crewai-llama3:8b", temperature=0.7)

# Define tools
search_tool = SerperDevTool()


# Function to read PDF content
def read_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PdfReader(f)
        content = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            content += page.extract_text()
    return content


def create_agents(topic):
    moderator = Agent(
        role=f'Discussion host on the topic: {topic} ',
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

    cdu_politician = Agent(
        role='Discussion participant',
        goal=f'Outline and promote '
             f'the position of the german political party CDU  on the topic :  {topic}. You can react on the answers of other participants.',
        backstory="""You are an politician with a strong economic background. You have a degree in law and are member of the german
        conservative political party CDU. """,
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],  # No specific tools for now
        llm=model
    )

    gruene_politician = Agent(
        role='Discussion participant',
        goal=f'Outline and promote '
             f'the position of the german political party "Bündnis 90 die Grünen"  on the topic :  {topic}. You can react on the answers of other participants.',
        backstory="""You are an politician with a strong ecological background. Your are a convinced environmentalist. """,
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=model
    )

    fdp_politician = Agent(
        role='Discussion participant',
        goal=f'Outline and promote '
             f'the position of the german political party "FDP"  on the topic :  {topic}. You can react on the answers of other participants.',
        backstory="""You are an politician with a strong background in finance and law.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=model
    )

    # Create a task for the teacher to answer questions
    introduction = Task(
        description=f"Welcome the audience and welcome the discussion participants. Then give a neutral overview on the topic of todays"
                    f"discussion {topic}. Summarize in <200 words. Also do it in a friendly and funny way.",
        expected_output="Introduction and welcome",
        agent=moderator
    )
    present_cdu_position = Task(
        description=f"Give a short but precise summary on your opinion on the topic {topic}",
        expected_output=f"opinion on the topic {topic}",
        agent=cdu_politician
    )

    present_fdp_position = Task(
        description=f"Give a short but precise summary on your opinion on the topic {topic}",
        expected_output=f"opinion on the topic {topic}",
        agent=fdp_politician
    )

    present_gruene_position = Task(
        description=f"Give a short but precise summary on your opinion on the topic {topic}",
        expected_output=f"opinion on the topic {topic}",
        agent=gruene_politician
    )
    summarize = Task(
        description=f"Summarise  the different positions in less than 500 words. And ask the participants on their counter or pro aruments to the different positions",
        expected_output=f"consice summary of the different position",

        agent=moderator
    )
    present_cdu_position_react = Task(
        description=f"React on the other participants arguments",
        expected_output=f"consice reaction on the  over positions",

        agent=cdu_politician
    )

    present_fdp_position_react = Task(
        description=f"React on the other participants arguments",
        expected_output=f"consice reaction on the  over positions",

        agent=fdp_politician
    )

    present_gruene_position_react= Task(
        description=f"React on the other participants arguments",
        expected_output=f"consice reaction on the  over positions",

        agent=gruene_politician
    )
    conclude_discussion = Task(
        description=f"Summarise  the different positions in less than 500 words. And ask the participants on their counter or pro aruments to the different positions",
        expected_output=f"Verdict on the topic of the discussion and summary of the main positions and their differences.",

        agent=moderator
    )

    return [moderator, cdu_politician, fdp_politician, gruene_politician],[introduction,present_fdp_position,present_gruene_position,present_cdu_position,summarize,present_fdp_position_react,present_cdu_position_react,present_gruene_position_react,conclude_discussion]



def run_crew(discussion_topic):
    agents,tasks = create_agents(discussion_topic)
    # Instantiate the crew with the updated tasks
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=2  # Set verbosity level for logging
    )
    # Run the crew
    result = crew.kickoff()
    return result
