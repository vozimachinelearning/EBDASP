import torch
import json
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

    def _safe_json(self, text: str):
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    def enhance_query(self, query: str, max_topics: int = 6, max_probes: int = 8) -> dict:
        prompt = f"""
You are enhancing a user query for a multi-agent evidence pipeline.
Return a JSON object with keys: enhanced_query, topics, probing_queries.
- enhanced_query: rewrite the query to be precise and complete.
- topics: short noun-phrase topics covering the query (max {max_topics}).
- probing_queries: specific questions to retrieve evidence (max {max_probes}).
Return JSON only.
Query: {query}
"""
        response = self.generate(prompt, max_new_tokens=512, temperature=0.2)
        parsed = self._safe_json(response)
        if isinstance(parsed, dict):
            enhanced_query = str(parsed.get("enhanced_query") or "").strip() or query
            topics = parsed.get("topics") if isinstance(parsed.get("topics"), list) else []
            probes = parsed.get("probing_queries") if isinstance(parsed.get("probing_queries"), list) else []
            topics = [str(item).strip() for item in topics if str(item).strip()][:max_topics]
            probes = [str(item).strip() for item in probes if str(item).strip()][:max_probes]
            return {
                "enhanced_query": enhanced_query,
                "topics": topics,
                "probing_queries": probes,
            }
        probes = self.generate_probing_queries(query, query, max_items=max_probes)
        return {"enhanced_query": query, "topics": [query], "probing_queries": probes}

    def generate_probing_queries(self, query: str, context: str, max_items: int = 5) -> List[str]:
        prompt = f"""
Given the main query and current context, generate up to {max_items} probing questions that directly help answer the main query.
Main Query: {query}
Current Context: {context}
Return a JSON array of strings only.
"""
        response = self.generate(prompt, max_new_tokens=256, temperature=0.4)
        parsed = self._safe_json(response)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()][:max_items]
        lines = [line.strip("- ").strip() for line in response.split("\n") if line.strip()]
        return lines[:max_items]

    def decompose_task(self, task_description: str, global_goal: Optional[str] = None) -> List[str]:
        """
        Decomposes a complex task into sub-tasks using the LLM.
        Returns a list of sub-task descriptions.
        """
        prompt = f"""
You are a task decomposition expert. Break down the task into sub-tasks that directly contribute to the global goal.
Global Goal: {global_goal or task_description}
Task Context: {task_description}

Return a JSON array of sub-task strings only.
"""
        response = self.generate(prompt, max_new_tokens=512, temperature=0.3)
        
        parsed = self._safe_json(response)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]

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

    def assign_roles(self, sub_tasks: List[str], global_goal: Optional[str] = None) -> List[dict]:
        """
        Assigns roles to each sub-task.
        Returns a list of dicts: {'task': str, 'role': str}
        """
        prompt = f"""
Assign a specific role to each task, ensuring all roles align to the global goal.
Global Goal: {global_goal or ""}

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
