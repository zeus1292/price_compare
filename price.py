from dotenv import load_dotenv
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from pprint import pprint

load dotenv()

@tool
def get_available_gtins(gtin: [str]) -> [str]:
    """Get the available GTINs for a category of products"""
    


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

