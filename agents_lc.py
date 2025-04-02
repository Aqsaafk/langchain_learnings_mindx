import os
import mysql.connector
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# ✅ Load environment variables
load_dotenv()

# ✅ Set up Azure OpenAI
endpoint = os.getenv("ENDPOINT_URL", "YOUR_ENDPOINT_KEY")  
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key = os.getenv("AZURE_OPENAI_API_KEY") 

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=subscription_key,
    api_version="2024-08-01-preview"
)

# ✅ Database Connection Function
def connect_db():
    """Connects to MySQL database."""
    return mysql.connector.connect(
        host="YOUR_DATABASE_HOST",
        user="USER",
        password=os.getenv("MYSQL_PASSWORD"),
        database="MINDX_INVENTORY_DEV"
    )

# ✅ Define the Tool for Database Querying
def fetch_sales_data(month: str):
    """Fetches sales data for a given month from the database."""
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    query = """
    SELECT 
        DATE_FORMAT(i.created_at, '%Y-%m') AS month,
        SUM(i.total) AS total_sales,
        COUNT(DISTINCT i.id) AS total_orders
    FROM invoices i
    WHERE i.created_at BETWEEN %s AND LAST_DAY(%s)
    GROUP BY month;
    """
    
    first_day = f"{month}-01"
    cursor.execute(query, (first_day, first_day))
    result = cursor.fetchone()
    db.close()

    return result if result else {"error": "No sales data found."}

# ✅ Register the Tool
sales_tool = Tool(
    name="fetch_sales_data",
    func=fetch_sales_data,
    description="Fetches total sales and order count for a given month (format: YYYY-MM)."
)

# ✅ Create an Agent
tools = [sales_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Type of agent
    verbose=True
)

# ✅ Ask the Agent a Query
query = "Get me the sales report for January 2025."
response = agent.run(query)

# ✅ Print the Response
print(response)
