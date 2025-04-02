from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv

load_dotenv()

endpoint=os.getenv("ENDPOINT_URL", "YOUR_END_POINT")  
deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4")  
subscription_key=os.getenv("AZURE_OPENAI_API_KEY") 

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=subscription_key,
    api_version="2024-05-01-preview"

)

# Define Tools (Functions) for Function Calling
@tool
def add_numbers(a: float, b: float) -> float:
    """Adds two numbers and returns the result."""
    return a + b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers and returns the result."""
    return a * b

# Bind Tools to the Chat Model
tools = [add_numbers, multiply_numbers]
llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49?"

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)



messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add_numbers": add_numbers, "multiply_numbers": multiply_numbers}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)


output=llm_with_tools.invoke(messages)

print(output.content)