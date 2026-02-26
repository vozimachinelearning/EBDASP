import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from typing import List, Optional

class LLMEngine:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        print(f"Loading model from {model_path} on {device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                local_files_only=True,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu":
                self.model.to(device)
            
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    def decompose_task(self, task_description: str) -> List[str]:
        """
        Decomposes a complex task into sub-tasks using the LLM.
        Returns a list of sub-task descriptions.
        """
        prompt = f"""
You are a task decomposition expert. Break down the following complex task into a list of independent sub-tasks that can be executed in parallel or sequence.
Task: {task_description}

Format your output as a bulleted list of sub-tasks. Do not include any other text.
- Sub-task 1
- Sub-task 2
...
"""
        response = self.generate(prompt, max_new_tokens=512, temperature=0.3)
        
        # Parse the response into a list
        sub_tasks = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                sub_tasks.append(line[2:].strip())
            elif line and line[0].isdigit() and '. ' in line: # Handle numbered lists
                parts = line.split('. ', 1)
                if len(parts) > 1:
                    sub_tasks.append(parts[1].strip())
                    
        return sub_tasks

    def assign_roles(self, sub_tasks: List[str]) -> List[dict]:
        """
        Assigns roles to each sub-task.
        Returns a list of dicts: {'task': str, 'role': str}
        """
        prompt = f"""
Assign a specific role to each of the following tasks. The roles should be descriptive (e.g., 'Researcher', 'Coder', 'Reviewer', 'Planner').

Tasks:
{chr(10).join([f"- {t}" for t in sub_tasks])}

Format your output as:
Task: [Task Description] | Role: [Role Name]
"""
        response = self.generate(prompt, max_new_tokens=512, temperature=0.3)
        
        assignments = []
        for line in response.split('\n'):
            if " | Role: " in line:
                parts = line.split(" | Role: ")
                task_part = parts[0].replace("Task: ", "").strip()
                role_part = parts[1].strip()
                assignments.append({'task': task_part, 'role': role_part})
        
        # Fallback if parsing fails, just assign 'Generalist'
        if not assignments and sub_tasks:
             for task in sub_tasks:
                 assignments.append({'task': task, 'role': 'Generalist'})
                 
        return assignments
