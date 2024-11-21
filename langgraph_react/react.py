from langchain import hub
from langchain.agents import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import Tool, tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI



react_prompt: PromptTemplate = hub.pull('hwchase17/react')


@tool
def triple(num: float) -> float:
    """ Multiply the input number by 3. """
    return float(num) * 3

tools = [TavilySearchResults(max_results=1), triple]

llm = ChatOpenAI(model='gpt-3.5-turbo-1106')

react_agent_runnable = create_react_agent(llm, tools, react_prompt)