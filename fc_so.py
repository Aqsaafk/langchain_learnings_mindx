from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from typing import Optional, List
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Azure OpenAI
endpoint = os.getenv("ENDPOINT_URL", "YOUR_ENDPOINT_KEY")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY") 

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=subscription_key,
    api_version="2024-08-01-preview"
)

# Define Structured Output
class Product(TypedDict):
    name: Annotated[str, ..., "The name of the product"]
    price: Annotated[float, ..., "The price of the product in USD"]
    rating: Annotated[float, ..., "The average customer rating out of 5"]
    link: Annotated[str, ..., "A link to purchase the product"]

class ShoppingResponse(TypedDict):
    products: List[Product]

# Define Tools (Function Calling)
@tool
def find_products(category: str, max_price: float) -> List[Product]:
    """Finds and returns a list of products based on category and price limit."""
    dummy_products = [
        {"name": "Logitech K380 Wireless Keyboard", "price": 45.99, "rating": 4.5, "link": "https://example.com/k380"},
        {"name": "Anker Wireless Keyboard", "price": 39.99, "rating": 4.2, "link": "https://example.com/anker"},
        {"name": "Redragon K552 Mechanical Keyboard", "price": 49.99, "rating": 4.6, "link": "https://example.com/redragon"}
    ]
    return [p for p in dummy_products if p["price"] <= max_price]

# Bind Tools
tools = [find_products]
llm_with_tools = llm.bind_tools(tools)

# Query the Assistant
query = "Find me a good wireless keyboard under 50$."
messages = [HumanMessage(query)]

# Invoke AI with function calling
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

# Process tool calls
for tool_call in ai_msg.tool_calls:
    if tool_call["name"] == "find_products":
        tool_msg = find_products.invoke(tool_call)
        messages.append(tool_msg)

# Get final structured response
structured_llm = llm.with_structured_output(ShoppingResponse)
output = structured_llm.invoke(messages)

# Print the structured JSON output
print(output)
