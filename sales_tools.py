from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from typing import List
from typing_extensions import TypedDict, Annotated
import os
from dotenv import load_dotenv
import mysql.connector

# Load environment variables
load_dotenv()

# Set up Azure OpenAI
endpoint = os.getenv("ENDPOINT_URL", "YOUR_ENDPOINT_KEY")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY") 

tools = []  # Placeholder for function tools

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=subscription_key,
    api_version="2024-08-01-preview"
)

# Connect to MySQL database
def connect_db():
    """Establishes a connection to MySQL."""
    return mysql.connector.connect(
        host="YOUR_HOSTNAME",
        user="USERNAME",
        password=os.getenv("MYSQL_PASSWORD"),
        database="MINDX_INVENTORY_DEV"
    )

# Define Structured Output
class Product(TypedDict):
    name: Annotated[str, ..., "The name of the product"]
    price: Annotated[float, ..., "The price of the product in USD"]
    total_sales: Annotated[int, ..., "Total sales count"]

class SalesResponse(TypedDict):
    products: List[Product]

# Define Tools (Function Calling)
@tool
def find_top_selling_products(max_price: float) -> List[Product]:
    """Finds top-selling products under a given price."""
    db = connect_db()
    cursor = db.cursor(dictionary=True)
    
    query = """
    SELECT p.name, p.price, SUM(sor.quantity) AS total_sales
    FROM sales_order_row sor
    JOIN product p ON sor.item_id = p.id
    WHERE p.price <= %s
    GROUP BY p.id
    ORDER BY total_sales DESC
    LIMIT 5;
    """
    cursor.execute(query, (max_price,))
    results = cursor.fetchall()
    db.close()
    
    return results if results else {"error": "No products found"}

# Bind Tools
tools.append(find_top_selling_products)
llm_with_tools = llm.bind_tools(tools)

# Query the Assistant
query = "Find me the top-selling products under 100$."
messages = [HumanMessage(query)]

# Invoke AI with function calling
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

# Process tool calls
for tool_call in ai_msg.tool_calls:
    if tool_call["name"] == "find_top_selling_products":
        tool_msg = find_top_selling_products.invoke(tool_call)
        messages.append(tool_msg)

# Get final structured response
structured_llm = llm.with_structured_output(SalesResponse)
output = structured_llm.invoke(messages)

# Print the structured JSON output
print(output)
