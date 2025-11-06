# ========== COLORS FOR BASH ========== 
RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
NC = '\033[0m'

class Snap4Prompts:
    def __init__(self):
        self.prompts = [] 

    def print_prompts(self, prompts):
        print()
        print("\n".join([f"{i}. {p.name}" for i, p in enumerate(prompts, 1)]))
        print()
        return ""

    def _confirmation(self):
        conf = input(f"{BLUE}You want to use this prompt? {NC}\n{GREEN} >>> Type 'yes'/'y' or 'no'/'n':{NC} ")
        conf_words = ['yes', 'y', 'yea', 'yup', 'ye']
        if conf in conf_words:
            return True
        else:
            print('='*50)
            return False 

    def _ask_user_args(self, prompt):
        user_inputs_dict = {}
        arguments_list = prompt.arguments
        print()
        for arg in arguments_list:
            arg_name = arg.name
            if arg_name:
                user_value = input(f" >>> {GREEN}Insert a value for arg '{arg_name}': {NC}")
                user_inputs_dict[arg_name] = user_value
        return user_inputs_dict
        

    def _choose_number(self):
        while True: 
            try:
                nr = int(input(f" {GREEN}>>> Select a number (int) for details: {NC}"))
                break 
            except ValueError: # Catch the specific error for bad int conversion
                print(f"{RED}It has to be only a number, no other characters are allowed {NC}\n")
            except Exception as e: # Catch generic error. 
                print(f"An unexpected error occurred: {e}")
        return nr

    def start(self, prompts):
        confirm = False
        while not confirm:
            print(f"{BLUE}Choose what prompt you want, or select 0 to go back to chat: {NC}{self.print_prompts(prompts)}")
            nr = self._choose_number()
            if nr == 0:
                return None, None
            print("Selected prompt: " + prompts[nr-1].name)
            print(prompts[nr-1].description)
            confirm = self._confirmation()
        user_args = self._ask_user_args(prompts[nr-1]) 
        return prompts[nr-1], user_args
