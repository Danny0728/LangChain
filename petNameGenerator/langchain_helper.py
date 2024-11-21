from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain import hub

#agents
# from langchain.agents import load_tools this is deprecated 
from langchain_community.agent_toolkits.load_tools import load_tools
# from langchain.agents import initialize_agent  deprecated
from langchain.agents import  create_react_agent, AgentExecutor
from langchain.agents.agent_types import AgentType

#
from dotenv import load_dotenv

load_dotenv()

# temperature 0 no risks accuracy kinda increases
# higher the temp more chances of not so accurate response but great creativity(randomness)
def generate_pet_name(animal_type, pet_color):
    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
    prompt_template = PromptTemplate(
        input_variables=['animal_type','pet_color'],
        template = 'I have a {animal_type} pet of {pet_color} color and I want a cool name for it. \
        Suggest me five cool names for my pet.'
    )
    # print("PromptTemplate: ",prompt_template,"---")
    name_chain = prompt_template | llm
    # print("nameChain: ",name_chain,"-----")
    response = name_chain.invoke({'animal_type':animal_type, 'pet_color':pet_color})
    return response.content

def langchain_agent():
    llm = ChatOpenAI(temperature=0.5)
    tools = load_tools(tool_names=['wikipedia','llm-math'],
                       llm=llm)
    
    prompt_template = hub.pull("hwchase17/react")
    # prompt_template = PromptTemplate(
    #     input_variables=['input','tool_names','tools','agent_scratchpad'],
    #     template="""
    #     You have access to the following tools: {tool_names}.
    #     Use them to answer questions or perform tasks as necessary.

    #     {agent_scratchpad}

    #     Question: {input}
    #     Response:""" 
    # )
    # need to try tthe above prompt cause cant every time pull for hc repo

    agent = create_react_agent(llm,tools,prompt_template)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    # agent = initialize_agent(tools=tools,
    #                         llm=llm,
    #                         # prompt=prompt_template
    #                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #                         verbose=False
    #                          )
    result = agent_executor.invoke(
        {"input":"What is the average lifespan of a dog? Multiply the lifespan by 3"}
        #input key is needed here
    )
    print(result)

if __name__=="__main__":
    langchain_agent()
    # print(generate_pet_name("dog","yellow"))
