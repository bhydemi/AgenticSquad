from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, MessageGraph, END
from chains import generate_chain, reflection_chain
from typing import Sequence, List
import os


load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = "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"

REFLECT = 'reflect'
GENERATE = 'generate'


def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({'messages': state})

def reflection_node(message: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({'messages': message})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


if __name__ == '__main__':
    inputs = HumanMessage(content='''Make this tweet better:'
                          Arsenal just signed one of the most brillant youngsters in our time.
                          He is very good with the head, both feet and also controls the ball very well'''
                          )

    response = graph.invoke(inputs)