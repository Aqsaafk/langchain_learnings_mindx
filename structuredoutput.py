from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from typing import Optional
from pydantic import BaseModel, Field
from typing import Optional, Union
from typing_extensions import Annotated, TypedDict
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate 

load_dotenv()

endpoint=os.getenv("ENDPOINT_URL", "YOUR_ENDPOINT_KEY")  
deployment=os.getenv("DEPLOYMENT_NAME", "gpt-4o")  
subscription_key=os.getenv("AZURE_OPENAI_API_KEY") 

llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=subscription_key,
    api_version="2024-08-01-preview"

)

class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


class ConversationalResponse(TypedDict):
    """Respond in a conversational manner. Be kind and helpful."""

    response: Annotated[str, ..., "A conversational response to the user's query"]


class FinalResponse(TypedDict):
    final_output: Union[Joke, ConversationalResponse]


structured_llm = llm.with_structured_output(FinalResponse)

output=structured_llm.invoke("Tell me a joke about cats")

print(output)

