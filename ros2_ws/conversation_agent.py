from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import yaml

# Configurar a LLM para a conversação
conversation_llm = ChatOllama(model="gemma3:4b", temperature=0.7)

# Carregar o prompt da conversação
with open('Prompts/conversation_prompt.yaml', 'r') as file:
    conversation_prompt = yaml.safe_load(file)['prompt']

# Criar o prompt template
conversation_prompt_template = PromptTemplate.from_template(conversation_prompt)


def is_within_domain(user_input: str) -> bool:
    """
    Check if the user input is within the robot's domain.
    Use your natural language processing capabilities to interpret the user input
    Returns True if the input is related to household tasks or robot capabilities.
    If the input is not related to household tasks or robot capabilities, return False.
    If the input is ambiguous, use your natural language processing capabilities to interpret the user input, and if the user ask you to do something that you can do, return True.
    For example for ambiguous topics:
    User: Help me with my homework
    You: I'm a household assistant robot. I can help you with household tasks like picking up objects, navigating rooms, and delivering items. Do you need help with something like that?
    User: Pick up the pencil
    You: I understand you want me to perform a command. Let me switch to command mode.
    """
    try:
        # Use the LLM to analyze if the input is within domain
        response = conversation_llm.invoke(
            conversation_prompt_template.format(input=user_input)
        )
        
        # Check if the response indicates it's a command
        if "I understand you want me to perform a command" in response.content:
            return True
            
        # Check if the response indicates it's outside domain
        if "I apologize, but I am a household assistant robot and cannot provide information about topics outside my domain" in response.content:
            return False
            
        # For ambiguous cases, check if the response suggests household-related tasks
        if any(phrase in response.content.lower() for phrase in [
            "household tasks",
            "picking up objects",
            "navigating rooms",
            "delivering items",
            "help you with"
        ]):
            return True
            
        # Default to True for other cases, as the LLM will handle the response appropriately
        return True
        
    except Exception as e:
        # In case of any error, default to True to allow the conversation to continue
        return True
    return True

# Função para processar a conversa
def process_conversation(user_input: str) -> str:
    """
    Process the user input in conversation.
    Return "__COMMAND_MODE__" if a command is detected, or the normal response otherwise.
    """
    try:
        # First check if the topic is within domain
        if not is_within_domain(user_input):
            return "I apologize, but I am a household assistant robot and cannot provide information about topics outside my domain. I am designed to help with household tasks like picking up objects, navigating rooms, and delivering items."
            
        response = conversation_llm.invoke(
            conversation_prompt_template.format(input=user_input)
        )
        
        # Verifica se a resposta indica que é um comando
        if "I understand you want me to perform a command" in response.content:
            return "__COMMAND_MODE__"
            
        return response.content
    except Exception as e:
        return f"Sorry, I had a problem processing your message: {e}"
