from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import yaml
import json
import re

# Importar os agentes existentes e novos
from main_robot_agent import agent_executor as command_agent
from conversation_agent import process_conversation

# Configurar a LLM para o router
router_llm = ChatOllama(model="gemma3:4b", temperature=0.3)

# Carregar o prompt do router
with open('Prompts/router_prompt.yaml', 'r') as file:
    router_prompt = yaml.safe_load(file)['prompt']

# Criar o prompt template
router_prompt_template = PromptTemplate.from_template(router_prompt)

# Função para determinar o tipo de input
def determine_input_type(user_input: str) -> str:
    """
    Determine if the user input is a command or a conversation.
    Return 'command' or 'conversation'.
    """
    try:
        # Usar a LLM para interpretar o input
        response = router_llm.invoke(
            router_prompt_template.format(input=user_input)
        )
        
        # Tentar extrair o JSON da resposta
        content = response.content
        # Procura por um padrão JSON na resposta
        json_match = re.search(r'\{.*\}', content)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result.get('type', 'conversation')
            except json.JSONDecodeError:
                pass
        
        # Se não conseguir parsear como JSON, verifica o contexto da frase
        input_lower = user_input.lower()
        
        # Palavras que indicam uma solicitação de informação ou ajuda
        info_words = ['help', 'explain', 'what', 'how', 'why', 'when', 'where', 'can you', 'could you', 'would you']
        if any(word in input_lower for word in info_words):
            return 'conversation'
            
        # Palavras que indicam um comando físico
        command_words = ['pick up', 'go to', 'bring', 'take', 'move', 'get', 'deliver']
        if any(word in input_lower for word in command_words):
            # Verifica se é um comando físico real ou uma metáfora/conversação
            if any(word in input_lower for word in ['help me', 'explain', 'understand', 'learn', 'teach']):
                return 'conversation'
            return 'command'
            
        return 'conversation'
            
    except Exception as e:
        # Se houver erro, assume que é uma conversação
        return 'conversation'

# Função principal do router
def route_input(user_input: str) -> str:
    """
    Route the input to the appropriate agent.
    """
    input_type = determine_input_type(user_input)
    
    if input_type == 'command':
        # Usar o agente de comandos existente
        response = command_agent.invoke({"input": user_input})
        return response['output']
    else:
        # Usar o novo agente de conversação
        response = process_conversation(user_input)
        
        # Se a resposta indica que é um comando, verifica novamente o contexto
        if response == "__COMMAND_MODE__":
            # Verifica se é realmente um comando físico ou uma metáfora/conversação
            if any(word in user_input.lower() for word in ['help me', 'explain', 'understand', 'learn', 'teach']):
                return "I apologize, but I am a household assistant robot. I can help you with physical tasks like picking up objects, navigating rooms, and delivering items. I cannot help with academic subjects or explanations."
            command_response = command_agent.invoke({"input": user_input})
            return command_response['output']
            
        return response

# Loop principal de interação
if __name__ == "__main__":
    print("Robot: Hello! I'm your household assistant robot. How can I help you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Robot: Goodbye!")
            break
        
        try:
            response = route_input(user_input)
            print(f"Robot: {response}")
        except Exception as e:
            print(f"Robot: An error occurred while processing your request: {e}")
            print("Please try again or rephrase your sentence.")
