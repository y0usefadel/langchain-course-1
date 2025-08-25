from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")

    text = """
Lionel Andrés "Leo" Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi] ⓘ; born 24 June 1987) is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. Widely regarded as one of the greatest players in history, Messi has set numerous records for individual accolades won throughout his professional footballing career, including eight Ballons d'Or, six European Golden Shoes, and eight times being named the world's best player by FIFA.[note 2] In 2025, he was named the All Time Men's World Best Player by the IFFHS. He is the most decorated player in the history of professional football having won 45 team trophies.[note 3] Messi's records include most goals in a calendar year (91), most goals for a single club (672 for Barcelona), most goals in La Liga (474), most goal contributions in the FIFA World Cup (21), and most goal contributions in the Copa América (32). A prolific goalscorer and creative playmaker, Messi has scored more than 870 senior career goals and has provided more than 380 assists for club and country.
"""

    summary_template = """
    You are a helpful assistant that can summarize text.
    Here is the text: {text}
    Please summarize the text in a few sentences.
    """

    summary_template = PromptTemplate(
        input_variables=["text"],
        template=summary_template,
    )

    llm  = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_template | llm

    response = chain.invoke({"text": text})
    print(response.content)




if __name__ == "__main__":
    main()
