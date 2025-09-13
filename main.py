from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-helloworld!")

    information ="""
    Elon Reeve Musk[a] (born June 28, 1971) is an international businessman and entrepreneur known for his leadership of Tesla, SpaceX, X (formerly Twitter), and the Department of Government Efficiency (DOGE). Musk has been the wealthiest person in the world since 2021; as of May 2025, Forbes estimates his net worth to be US$424.7 billion
    """

    summary_template = """
    Summarize this {information} in a concise manner and provide two interesting facts about the topic.
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"],
        template = summary_template
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content) 

if __name__ == "__main__":
    main()
