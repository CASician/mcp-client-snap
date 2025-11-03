import json
import requests
import logging
from token_manager import TokenManager

# TODO add loggin logic throghout the file
logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="connecting_lab2mcp.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    encoding="utf-8",
)

def load_json(path, required_keys):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in '{path}': {missing}")
    return data

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
        creds = load_json(
            "user_credentials.json",
            required_keys=["username", "password"]  
        )
        self.username = creds["username"]
        self.password = creds["password"]
        # == Load ClearML config ==
        cfg = load_json(
            "clearml_config.json",
            required_keys=["clearml_ondemand_api_base_url", "clearml_llm_endpoint"]
        )
        self.api_base_url = cfg["clearml_ondemand_api_base_url"]
        self.endpoint = cfg["clearml_llm_endpoint"]

    def _authenticate(self):
        tm = TokenManager(self.username, self.password) 
        self.access_token = tm.get_token()
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }
    def chat_completion(self, messages, functions=None, function_call="auto", max_tokens=500):
        # Convert OpenAI-like messages into a single prompt for the custom Lab API
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        body = {
            "access_token": self.access_token,
            "endpoint": self.endpoint,
            "params": {"prompt": prompt}
        }
        # logger.info("BODY: %s", body["params"])

        # Calls the custom API
        response = requests.post(
            self.api_base_url,
            data=json.dumps(body),
            headers=self.headers
        )

        if response.status_code != 200:
            logger.error("LabLLM API error: %s", response.text)
            raise Exception(f"LabLLM error: {response.status_code}")

        data = response.json()
        answer = data.get("answer", "")
        
        # --- Custom Parsing and Message Structuring ---
        
        parsed_function_call = None
        reasoning_text = None
        
        # 1. Attempt to find and parse function calls (if allowed)
        if function_call != "none":
            
            # Priority Check: Look for combined (Text + JSON) response
            json_start = answer.find('{"function_call":')
            
            if json_start != -1:
                json_string = answer[json_start:].strip()
                
                try:
                    parsed_json = json.loads(json_string)
                    
                    if isinstance(parsed_json, dict) and "function_call" in parsed_json:
                        parsed_function_call = parsed_json["function_call"]
                        reasoning_text = answer[:json_start].strip()

                except (json.JSONDecodeError, TypeError):
                    # Fall through to the next check if parsing combined JSON fails
                    pass

            # Secondary Check: If no call was found yet, check for Pure JSON response
            if not parsed_function_call:
                try:
                    parsed_pure = json.loads(answer)
                    
                    if isinstance(parsed_pure, dict) and "function_call" in parsed_pure:
                        parsed_function_call = parsed_pure["function_call"]

                except (json.JSONDecodeError, TypeError):
                    # No valid JSON found, treat everything as plain text.
                    pass
                
        # --- 2. Build the final message object based on parsing result ---
        
        if parsed_function_call:
            # Struttura richiesta per la tool call
            message = {
                "role": "assistant",
                "content": None,
                "function_call": parsed_function_call
            }
            
            # Log the text reasoning if it was found, but keep the structure clean.
            if reasoning_text:
                logger.info("MODEL REASONING (TEXT BEFORE JSON): %s", reasoning_text)
                
        else:
            # Struttura richiesta per la risposta testuale
            message = {"role": "assistant", "content": answer}
            
        return {"choices": [{"message": message}]}
