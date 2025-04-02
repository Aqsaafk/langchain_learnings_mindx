from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict, Annotated
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI API Configuration
endpoint = os.getenv("ENDPOINT_URL", "YOUR_ENDPOINT_KEY")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY") 

# Initialize LLM
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=subscription_key,
    api_version="2024-08-01-preview"
)

# Define Structured Output Format
class BookSummary(TypedDict):
    """Structured book summary output."""

    title: Annotated[str, ..., "Title of the book"]
    author: Annotated[str, ..., "Author of the book"]
    summary: Annotated[str, ..., "A brief summary of the book"]
    key_takeaways: Annotated[list[str], ..., "Important lessons from the book"]
    target_audience: Annotated[str, ..., "Who should read this book"]
    read_time: Annotated[str, ..., "Estimated time required to finish the book"]

# Enable structured output for the model
structured_llm = llm.with_structured_output(BookSummary)

# User query
query = "Summarize 'Atomic Habits' by James Clear."

# Invoke model
output = structured_llm.invoke(query)

# Print structured output
print(output)
