from crewai import Agent
from crewai_tools import SerperDevTool,PDFSearchTool

from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader

# Define the language model
llama_model = ChatOpenAI(model_name="crewai-llama3:8b", temperature=0.7)

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

# Define the data sources (PDF files)
data_peter_parker = read_pdf("data/Gruene.pdf")
data_hulk = read_pdf("data/FDP.pdf")
data_dr_doom = read_pdf("data/CDU.pdf")



tool_peter = PDFSearchTool("data/Gruene.pdf")
tool_doom = PDFSearchTool("data/CDU.pdf")
tool_hulk = PDFSearchTool("data/FDP.pdf")

#tool_peter = PDFSearchTool("data/Test.pdf")
#tool_doom = PDFSearchTool("data/Test.pdf")
#tool_hulk = PDFSearchTool("data/Test.pdf")

# Define agents
peter_parker = Agent(
    role='Environmental Policy Expert',
    goal='Discuss environmental policies and green initiatives',
    backstory="""Peter Parker is an environmental policy expert. He is passionate about environmental protection and sustainable development.""",
    verbose=True,
    allow_delegation=False,
    llm=llama_model,
    tools=[search_tool],
    context=data_peter_parker
)

hulk = Agent(
    role='Economic Policy Analyst',
    goal='Discuss conservative policies and economic growth',
    backstory="""Hulk is an economic policy analyst. He focuses on economic policies and maintaining law and order.""",
    verbose=True,
    allow_delegation=False,
    llm=llama_model,
    tools=[search_tool],
    context=data_hulk
)

dr_doom = Agent(
    role='Healthcare Policy Specialist',
    goal='Discuss social policies and healthcare',
    backstory="""Dr. Doom is a healthcare policy specialist. He is an expert in social policies and healthcare reform.""",
    verbose=True,
    allow_delegation=False,
    llm=llama_model,
    tools=[search_tool],
    context=data_dr_doom
)




host = Agent(
    role='Podcast Host',
    goal='Moderate the discussion and ensure a balanced conversation',
    backstory="""The host is a seasoned podcast presenter, known for their impartiality and ability to facilitate engaging discussions.""",
    verbose=True,
    allow_delegation=False,
    llm=llama_model
)

