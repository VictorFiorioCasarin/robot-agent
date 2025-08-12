from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from src.robot_agent.robot_tools import robot_tools
import json
import yaml

# 1. Configurar a LLM Principal (Gemma3B via Ollama)
main_llm = ChatOllama(model="gemma3:4b", temperature=0.3)

# 2. Carregar o Prompt do Agente a partir do arquivo YAML
with open('Prompts/main_prompt.yaml', 'r') as file:
    agent_prompt = yaml.safe_load(file)['prompt']

# Definir o Prompt do Agente (ReAct/ZeroShotAgent)
# Este prompt guia o Gemma3B sobre como pensar e usar as ferramentas.
# Ele DEVE explicar claramente as ferramentas e o processo de pensamento esperado.
agent_prompt = PromptTemplate.from_template(agent_prompt)

# 3. Criar o Agente
# create_react_agent usa o padrão ReAct para raciocínio.
agent = create_react_agent(main_llm, robot_tools, agent_prompt)

# 4. Criar o Executor do Agente
agent_executor = AgentExecutor(
    agent=agent,
    tools=robot_tools,
    verbose=True, # mostra o raciocínio do agente
    handle_parsing_errors=True # captura erros de parsing
)

# 5. Loop de Interação
if __name__ == "__main__":
    print("Robot: Hello! How can I help you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Robot: Goodbye!")
            break
        
        try:
            # Invocar o agente com a entrada do usuário
            response = agent_executor.invoke({"input": user_input})
            print(f"Robot: {response['output']}")
        except Exception as e:
            print(f"Robot: An error occurred while processing your request: {e}")
            print("Please try again or rephrase your sentence.")
