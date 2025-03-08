
"""Run manually the Graph."""

# Import packages and modules
from dotenv import load_dotenv
import warnings

from src.graph import graph

warnings.filterwarnings("ignore")
load_dotenv("local/.env")

if __name__ == "__main__":
    mem = {"configurable": {"thread_id": "1"}}
    question = input("Enter your question: ")
    res = graph.invoke(
        {"question": question},
        mem
    )
    print("**************************************GENERATED RESPONSE**************************************")
    print(res.get("generation"))
    print("**************************************CONFIDENCE SCORE**************************************")
    print(res.get("confidence_score"))
    print("**************************************GENERATION ITERATIONS**************************************")
    print(res.get("iterations"))
    print("**************************************RETRIEVED DOCUMENTS**************************************")
    print(res.get("documents"))