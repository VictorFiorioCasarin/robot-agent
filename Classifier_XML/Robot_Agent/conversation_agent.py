from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import yaml

# Configurar a LLM para a conversação
conversation_llm = ChatOllama(model="gemma3:4b", temperature=0.7)

# Carregar o prompt da conversação
with open('Classifier_XML/Robot_Agent/Prompts/conversation_prompt.yaml', 'r') as file:
    conversation_prompt = yaml.safe_load(file)['prompt']

# Criar o prompt template
conversation_prompt_template = PromptTemplate.from_template(conversation_prompt)

# Função para processar a conversa
def process_conversation(user_input: str) -> str:
    """
    Processa o input do usuário na conversa.
    Retorna "__COMMAND_MODE__" se detectar um comando, ou a resposta normal caso contrário.
    """
    try:
        response = conversation_llm.invoke(
            conversation_prompt_template.format(input=user_input)
        )
        
        # Verifica se a resposta indica que é um comando
        if "I understand you want me to perform a command" in response.content:
            return "__COMMAND_MODE__"
            
        return response.content
    except Exception as e:
        return f"Desculpe, tive um problema ao processar sua mensagem: {e}"
