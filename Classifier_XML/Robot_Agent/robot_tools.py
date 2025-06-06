import requests
from langchain.tools import tool
import json
import yaml

# Importando o parser JSON que o LangChain usa internamente para ser mais robusto
from langchain.output_parsers import json as json_parser_lc # Importa o módulo json do langchain.output_parsers

# Carregar o prompt do classificador a partir do arquivo YAML
with open('Classifier_XML/Robot_Agent/Prompts/classifier_prompt.yaml', 'r') as file:
    CLASSIFIER_PROMPT = yaml.safe_load(file)['prompt']

known_objects = ["cup", "mug", "bowl", "dish", "spoon", "fork", "knife", "napkin", "tray", "basket", "trash bag", "book", "CD", "DVD", "BluRay", "cereal box", "milk carton", "bag", "coat", "apple", "paper", "teabag", "pen", "remote control", "chocolate egg", "refrigerator bottle", "newspaper", "umbrella"]
known_rooms = ["bedroom", "kitchen", "living room", "dining room", "bathroom", "hall", "laundry room"]

@tool
def classify_sentence_semantic(sentence: str) -> str:
    """
    Classifies each word of a sentence into semantic categories (action, object, location, room, direction, other)
    using Gemma3B via Ollama. Returns a JSON string of a dictionary summarizing the semantic tokens.
    """
    system_prompt_with_sentence = CLASSIFIER_PROMPT + f"\nSentence: {sentence}\nOutput:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": system_prompt_with_sentence,
                "stream": False,
                "options": {"temperature": 0.0} # Importante para classificação consistente
            }
        )
        response.raise_for_status() # Lança um HTTPError para respostas ruins (4xx ou 5xx)
        gemma_output_str = response.json()["response"].strip()

        # Tenta analisar a saída string em uma lista de tuplas Python
        try:
            parsed_output = eval(gemma_output_str)
            if not isinstance(parsed_output, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in parsed_output):
                raise ValueError("Parsed output is not in the expected list of (word, label) format.")
        except (SyntaxError, ValueError) as e:
            return f"Error parsing Gemma3 output: {e}. Raw output: {gemma_output_str}"

        # Converte a lista de (word, label) em um dicionário resumido para consumo mais fácil pela LLM principal
        semantic_dict = {}
        for word, label in parsed_output:
            if label != "other": # Ignora "other"
                # Lida com casos onde um rótulo pode aparecer várias vezes (por exemplo, múltiplos objetos)
                if label in semantic_dict:
                    if not isinstance(semantic_dict[label], list):
                        semantic_dict[label] = [semantic_dict[label]]
                    semantic_dict[label].append(word)
                else:
                    semantic_dict[label] = word
        
        return json.dumps(semantic_dict)
        
    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama: {e}"
    
@tool
def rewrite_sentence(input_str: str) -> str:
    """
    Rewrite the sentence command with more information that you get from ask_user tool.
    Use your own knowledge to rewrite the sentence

    Here is an example:
    [User] "Take the keys"
    [Robot] used ask_user tool and get new informations about where the keys are.
    [User] "They are on the kitchen table"
    [Robot] Add the new informations to the sentence like this: "Take the keys from the kitchen table"
    [Robot] Return the new sentence: "Take the keys from the kitchen table"
    [Robot] Restart his workflow with the new sentence using the `classify_sentence_semantic` tool

    """
    try:
        # Tenta remover as aspas simples externas se existirem
        if input_str.startswith("'") and input_str.endswith("'"):
            input_str = input_str[1:-1]
        
        # Tenta analisar como JSON primeiro
        try:
            parsed_input = json.loads(input_str)
            # Se for JSON, retorna o texto formatado
            return f"Rewritten sentence: {input_str}"
        except json.JSONDecodeError:
            # Se não for JSON, trata como string normal
            return f"Rewritten sentence: {input_str}"
            
    except Exception as e:
        return f"An unexpected error occurred in rewrite_sentence: {e}"

@tool
def navigate_to(input_str: str) -> str:
    """
    Navigates the robot to a specific room in the house (for example: kitchen, living room, bedroom).
    Returns a success or failure message.
    The input should be a JSON string with a 'room' key, for example: '{"room": "kitchen"}'.
    """
    try:
        # Tenta remover as aspas simples externas se existirem
        if input_str.startswith("'") and input_str.endswith("'"):
            input_str = input_str[1:-1]
        
        parsed_input = json.loads(input_str)
        room = parsed_input.get("room")
        if not room:
            return "Error: 'room' key not found in input JSON for navigate_to."
        
        print(f"[ROBOT ACTION] Navigating to room: {room}")
        # Simula a chamada à API de navegação do robô
        if room.lower() in known_rooms:
            return f"Robot arrived at {room}."
        else:
            # adiciona um novo cômodo a lista de cômodos conhecidos
            known_rooms.append(room)
            print(f"Room '{room}' added to known_rooms list.")
            return f"Robot arrived at {room}."
        
        '''
        else:
            return f"Cannot navigate to '{room}'. Unknown or inaccessible room."
        '''
    except json.JSONDecodeError:
        return f"Error: Invalid JSON input for navigate_to. Ensure it uses double quotes: {input_str}"
    except Exception as e:
        return f"An unexpected error occurred in navigate_to: {e}"

@tool
def pick_up_object(input_str: str) -> str:
    """
    Attempts to pick up a specific object in the robot's field of view. The robot must be near the object.
    Returns a success or failure message.
    The input should be a JSON string with an 'object_name' key, for example: '{"object_name": "box"}'.
    """
    try:
        # Tenta remover as aspas simples externas se existirem
        if input_str.startswith("'") and input_str.endswith("'"):
            input_str = input_str[1:-1]
        
        parsed_input = json.loads(input_str)
        object_name = parsed_input.get("object_name")
        if not object_name:
            return "Error: 'object_name' key not found in input JSON for pick_up_object."

        print(f"[ROBOT ACTION] Attempting to pick up: {object_name}")
        # Simula a chamada à API de manipulação do robô
        if object_name.lower() in known_objects:
            return f"Object '{object_name}' picked up successfully."
        else:
            # adiciona um novo objeto a lista de objetos conhecidos
            known_objects.append(object_name)
            print(f"Object '{object_name}' added to known_objects list.")
            return f"Object '{object_name}' picked up successfully."

        '''
        else:
            return f"Could not find or pick up the object '{object_name}'."
        '''

    except json.JSONDecodeError:
        return f"Error: Invalid JSON input for pick_up_object. Ensure it uses double quotes: {input_str}"
    except Exception as e:
        return f"An unexpected error occurred in pick_up_object: {e}"

@tool
def deliver_object(input_str: str) -> str:
    """
    Delivers an object the robot is currently holding to a specified location or person.
    Returns a success or failure message.
    The input should be a JSON string with an 'object_name' key and an optional 'target_location' key,
    for example: '{"object_name": "box", "target_location": "user"}' or '{"object_name": "book"}'.
    """
    try:
        # Tenta remover as aspas simples externas se existirem
        if input_str.startswith("'") and input_str.endswith("'"):
            input_str = input_str[1:-1]
        
        parsed_input = json.loads(input_str)
        object_name = parsed_input.get("object_name")
        target_location = parsed_input.get("target_location", "user")

        if not object_name:
            return "Error: 'object_name' key not found in input JSON for deliver_object."

        print(f"[ROBOT ACTION] Delivering '{object_name}' to '{target_location}'.")
        # Simula a chamada à API de entrega do robô
        return f"Object '{object_name}' delivered to '{target_location}'."
    except json.JSONDecodeError:
        return f"Error: Invalid JSON input for deliver_object. Ensure it uses double quotes: {input_str}"
    except Exception as e:
        return f"An unexpected error occurred in deliver_object: {e}"

from langchain.tools import tool

@tool
def ask_user(input_str: str) -> str:
    """
    Displays a question to the user and waits for an answer.
    If the user says 'I don't know', returns a specific keyword to trigger fallback behavior.
    """
    user_response = input(f"\n[Robot]: {input_str}\n[You]: ")
    if user_response.strip().lower() in {"i don't know", "dont know", "no idea", "not sure", "não sei"}:
        return "__UNKNOWN_LOCATION__"
    return user_response



# Define a lista robot_tools depois de todas as ferramentas serem definidas.
robot_tools = [classify_sentence_semantic, navigate_to, pick_up_object, deliver_object, ask_user, rewrite_sentence]