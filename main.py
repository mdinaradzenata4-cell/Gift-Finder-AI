import os
from agent import build_graph
from langgraph.types import Command

if not os.environ.get("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Please add your API key.")

graph = build_graph()
config = {"configurable": {"thread_id": "gift-finder-cli"}}

print("\n🎁 Welcome to Gift Finder AI!")
print("Ask me for gift ideas, search products or manage your wishlist.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "გამოსვლა"]:
        print("Goodbye! 🎁")
        break

    result = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config
    )

    while result.get("__interrupt__"):
        interrupt_data = result["__interrupt__"][0].value
        print("\n⚠️ Confirmation required:")
        print(interrupt_data.get("message", "Do you approve this action?"))

        answer = input("Approve? (yes/no): ").strip().lower()
        approved = answer in ["yes", "y", "დიახ", "კი"]

        result = graph.invoke(
            Command(resume=approved),
            config
        )

    for message in reversed(result["messages"]):
        if message.type == "ai" and message.content:
            print("\nGift Finder AI:", message.content)
            break
