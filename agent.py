
import os
import requests

from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt


# =====================================
# 1. Gift Wishlist Storage
# =====================================

gift_wishlist = []


# =====================================
# 2. Gift Finder Tools
# =====================================

@tool
def find_gifts(interests: str, budget: float, occasion: str) -> str:
    """
    Suggest gift ideas based on interests, budget and occasion.
    """

    return (
        f"Suggest gifts for someone interested in {interests}. "
        f"Budget: {budget} GEL. Occasion: {occasion}. "
        "Provide creative and personalized gift recommendations."
    )


@tool
def save_gift(gift_name: str, price: float, recipient: str) -> str:
    """
    Save a selected gift idea to the wishlist.
    """

    gift = {
        "gift_name": gift_name,
        "price": price,
        "recipient": recipient
    }

    gift_wishlist.append(gift)

    return f"{gift_name} successfully saved to wishlist!"


@tool
def list_gifts() -> str:
    """
    Display all gift ideas saved in the wishlist.
    """

    if not gift_wishlist:
        return "Your wishlist is empty."

    return str(gift_wishlist)


@tool
def delete_gift(gift_name: str) -> str:
    """
    Delete a saved gift only after user confirmation.
    """

    confirmation = interrupt({
        "action": "delete_gift",
        "gift_name": gift_name,
        "message": f"Do you approve deleting {gift_name}?"
    })

    if confirmation is not True:
        return f"Deletion of {gift_name} was cancelled."

    for gift in gift_wishlist:
        if gift["gift_name"].lower() == gift_name.lower():
            gift_wishlist.remove(gift)
            return f"{gift_name} deleted successfully."

    return f"{gift_name} was not found."


@tool
def search_products(query: str) -> str:
    """
    Search sample products and prices using DummyJSON API.
    """

    try:
        response = requests.get(
            "https://dummyjson.com/products/search",
            params={"q": query, "limit": 5},
            timeout=20
        )

        response.raise_for_status()

        products = response.json()["products"]

        if not products:
            return "No products found."

        results = []

        for product in products:
            results.append({
                "name": product["title"],
                "price": product["price"],
                "currency": "USD",
                "description": product["description"]
            })

        return str(results)

    except requests.RequestException as error:
        return f"Product search failed: {error}"


# =====================================
# 3. Agent State
# =====================================

class GiftFinderState(TypedDict):
    messages: Annotated[list, add_messages]


# =====================================
# 4. Build AI Agent
# =====================================

def build_graph():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is missing!")

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=1000,
        api_key=api_key
    )

    gift_tools = [
        find_gifts,
        save_gift,
        list_gifts,
        delete_gift,
        search_products
    ]

    llm_with_tools = llm.bind_tools(gift_tools)

    def assistant(state: GiftFinderState):
        response = llm_with_tools.invoke(state["messages"])

        return {"messages": [response]}

    builder = StateGraph(GiftFinderState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(gift_tools))

    builder.add_edge(START, "assistant")

    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )

    builder.add_edge("tools", "assistant")

    memory = InMemorySaver()

    graph = builder.compile(checkpointer=memory)

    return graph


print("Gift Finder AI agent code loaded successfully!")
