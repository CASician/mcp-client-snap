import json
import requests
import logging
from llama4.token_manager import TokenManager
import re

# ========== LOGGING ==========
logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="logs/llm_output.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)

# ========== READ FROM JSON FILES ==========
def load_json(path, required_keys):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in '{path}': {missing}")
    return data

# ========== REGEX ========== 
JSON_START_PATTERN = r'\{\s*\\?["\']function_call["\']\s*:\s*'

# For FUNCTION RECOGNITION in LLM response. 
def find_start_regex(text):
    match = re.search(JSON_START_PATTERN, text)
    if match:
        # Return the index of the first character of the match ('{')
        return match.start()
    return -1

class LabLLM:
    def __init__(self):
        self.username = None 
        self.password = None 
        self.api_base_url = None 
        self.endpoint = None
        self.access_token = None
        self.headers = None
        self._login()
        self._authenticate()

    def _login(self):
        """ 
        Load username and password for token generation.
        Load ClearML configuration. 
        """
        # == User credentials ==
        creds = load_json(
            "llama4/user_credentials.json",
            required_keys=["username", "password"]  
        )
        self.username = creds["username"]
        self.password = creds["password"]

        # == Load ClearML config ==
        cfg = load_json(
            "llama4/clearml_config.json",
            required_keys=["clearml_ondemand_api_base_url", "clearml_llm_endpoint"]
        )
        self.api_base_url = cfg["clearml_ondemand_api_base_url"]
        self.endpoint = cfg["clearml_llm_endpoint"]

    def _authenticate(self):
        """
        Generate token and headers. 
        """
        tm = TokenManager(self.username, self.password) 
        self.access_token = tm.get_token()
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def chat_completion(self, messages, functions=None, function_call="auto", max_tokens=500):
        """
        This is where magic happens. This function mimics OpenAI chat.completion.create() function. 
        It receives the entire conversation, it auth to Snap4, and ask the LLM an answer. 
        In the future it should support other args. For reference: 
        https://platform.openai.com/docs/api-reference/chat/create?lang=python
       
        Workflow:
        1. Prepare body as wanted by Snap4 llm. Include ALL messages in prompt. 
        2. Invoke LLM
        3. Parse the answer for the relevant part. 
            - It's needed because the model answers with both prompt and answer. But we only need answer.
            - {"prompt": "my_prompt", "answer": "llm_answer"} -> we only need "llm_answer"
        4. If function_call is enabled, look for the function inside the answer text. 
            a. JSON
            b. Text + JSON
            c. Text + JSON + Text
            - all extra text is logged as MODEL REASONING
            - a/b/c should not be needed if LLM listened to instructions. But eh. 
        5. Return message: a JSON object with proper function_call (openai style) if found. 
        
        Note: the arg `functions` does nothing. In OpenAI style, the tools are passed to the llm with every call. I find that redundant. Once with the SYSTEM_MESSAGE is enough. Look inside the client code where the server is initialized. 
        """
        body = {
            "access_token": self.access_token,
            "endpoint": self.endpoint,
            "params": {"prompt": messages}
        }

        # ========== INVOKE LLM COMPLETION ==========  
        response = requests.post(
            self.api_base_url,
            data=json.dumps(body),
            headers=self.headers
        )

        if response.status_code != 200:
            logger.error("LabLLM API error: %s", response.text)
            raise Exception(f"LabLLM error 1: {response.status_code}")

        # ========== GET RELEVANT PART: ANSWER ========== 
        # `data` comprehends both previous messages AND the answer
        # {"prompt": "my_prompt", "answer": "llm_answer"}
        data = response.json()
        if isinstance(data, dict):
            answer = data.get("answer", "")
        else:
            raise Exception(f"LabLLM error 2:\n{data}")
        
        parsed_function_call = None
        reasoning_text = None
        
        # ========== FUNCTION_CALL IF ENABLED ==========  
        if function_call != "none":
            
            # ========== FIND FUNCTION IN ANSWER ==========
            # Divide LLM answer in `parsed_function_call` = JSON and `reasoning_text` = Text

            # 1. Look for combined (Text + JSON) response. Not (Text + JSON + Text TODO!)
            json_start = find_start_regex(answer) 
            
            if json_start != -1:
                json_string = answer[json_start:].strip()
                
                try:
                    parsed_json = json.loads(json_string)
                    
                    if isinstance(parsed_json, dict) and "function_call" in parsed_json:
                        parsed_function_call = parsed_json["function_call"]
                        reasoning_text = answer[:json_start].strip()
                        logger.info("PARSE 1: Separate REASONING from FUNCTION CALL")

                except (json.JSONDecodeError, TypeError):
                    # Fall through to the next check if parsing combined JSON fails
                    logger.info("PARSE 1 FAIL: Separate REASONING from FUNCTION CALL")
                    pass

            # 2. Check for Pure JSON response
            if not parsed_function_call:
                try:
                    parsed_pure = json.loads(answer)
                    
                    if isinstance(parsed_pure, dict) and "function_call" in parsed_pure:
                        parsed_function_call = parsed_pure["function_call"]
                        logger.info("PARSE 2: llm invoked correctly a function")

                except (json.JSONDecodeError, TypeError):
                    # No valid JSON found, treat everything as plain text.
                    logger.info("PARSE 2 FAIL: llm DID NOT invoke correctly a function")
                    pass

            # 3. TODO Check for combined answer (Text + JSON + Text) and reconsider order. 
                
            # ========== BUILD JSON-FUNCTION_CALL OPENAI STYLE ========== 
            if parsed_function_call:
                message = {
                    "role": "assistant",
                    "content": None,
                    "function_call": parsed_function_call
                }
            
            # Log the text reasoning if found
            if reasoning_text:
                logger.info("MODEL REASONING: %s", reasoning_text)
                
        else:
            # No function found: return LLM answer. 
            message = {"role": "assistant", "content": answer}
            
        return {"choices": [{"message": message}]}

