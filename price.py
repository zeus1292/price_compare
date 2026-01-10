#Code Structure : 
#1. Identify various workflows.
#2. Build corresponding agents against each workflow. Leverage MCP servers or build them wherever possible.
#3. Build a central agent that can be used to orchestrate the other agents.
#4. Build a central UI that can be used to monitor and control the agents.
#5. Build a central API that can be used to interact with the agents.
#6. Build a central database that can be used to store the data.
#7. Build a central logging system that can be used to log the activities.
#8. Build a central error handling system that can be used to handle the errors.
#9. Build a central monitoring system that can be used to monitor the agents.
#10. Build a central reporting system that can be used to report the activities.


from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from pprint import pprint
from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool

#Create Agent State
from langchain.agents import AgentState

class ProductState(AgentState):
    product_id: str
    product_description: str
    product_price: float
    product_availability: str
    target_seller: str


@tool
def update_latest_gtin()->str:
    """Update the latest GTIN for a category of products
        #This function will identify a suitable source 
        # of data and download all publicly available GTINs."""


@tool
def product_identifier_search(product_id:str)->str:
    """Search for the product based on its identifier"""

@tool
def product_description_search(product_description:str)->str:
    """Search for the product based on its description"""

@tool
def retailer_product_search(target_seller:str)->str:
    """Search for the product in the target retailer"""

@tool
def identify_retailer(product_id:str)->str:
    """Identify the retailer for the product"""

@tool
def marketplace_product_search(target_seller:str)->str:
    """Search for the product in the target retailer"""

@tool
def identify_marketplace(product_id:str)->str:
    """Identify the retailer for the product"""
