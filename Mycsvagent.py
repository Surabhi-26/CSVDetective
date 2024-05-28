import os
import tkinter as tk
from tkinter import filedialog
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import CSVSearchTool

os.environ["OPENAI_API_KEY"] = "NA"


def select_file():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    print(f"Selected file: {file_path}")
    return file_path


csv_file_path = select_file()
query=input("Enter your query: ")

llm = ChatOpenAI(
    model="crewai-llama3",
    base_url="http://localhost:11434/v1"
)

toolcsv = CSVSearchTool(
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3",
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="sentence-transformers/msmarco-distilbert-base-v4"
            ),
        ),
    ),
    csv=csv_file_path  
)

agent1 = Agent(
    role="CSV Data Analyst",
    goal="Provide the solutions to the queries asked by the user on the CSV data",
    backstory="You are an excellent CSV data summarizer that provides insights into the data by answering the queries",
    allow_delegation=False,
    verbose=True,
    tools=[toolcsv],
    llm=llm
)

agent2 = Agent(
    role="Report generator",
    goal="Provide me detailed report on the insights given by CSV Data Analyst",
    backstory="You are a skilled report generator which gives detailed report on the insights",
    verbose=True,
    llm=llm
)

task1 = Task(
    description=f"{query}",
    agent=agent1,
    expected_output=f"A summary on {query}"
)

task2 = Task(
    description=f"Write a detailed report on {query} based on the CSV Data Analyst's insights",
    agent=agent2,
    expected_output="A paragraph with clear understanding of all statistics"
)

crew = Crew(
    tasks=[task1],
    agents=[agent1],
    verbose=2
)

result = crew.kickoff()
print(result)
