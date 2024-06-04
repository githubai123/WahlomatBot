from crewai import Task
from .agents import peter_parker, hulk, dr_doom, host

# Define tasks
task1 = Task(
    description="""Discuss the latest environmental policies and green initiatives with Peter Parker.""",
    expected_output="Detailed discussion points and insights",
    agent=peter_parker
)

task2 = Task(
    description="""Discuss conservative policies and economic growth with Hulk.""",
    expected_output="Detailed discussion points and insights",
    agent=hulk
)

task3 = Task(
    description="""Discuss social policies and healthcare reform with Dr. Doom.""",
    expected_output="Detailed discussion points and insights",
    agent=dr_doom
)

task4 = Task(
    description="""Moderate the discussion and ensure a balanced conversation.""",
    expected_output="Summary of the discussion and key takeaways",
    agent=host
)

