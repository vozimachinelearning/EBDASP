from typing import Any, Dict, List, Optional, TYPE_CHECKING
import uuid
import time
import threading

from .coordinator import Coordinator
from .messages import QueryRequest, QueryResponse, RouteRequest, Task, TaskAssignment, TaskResult
from .transport import Transport

if TYPE_CHECKING:
    from .llm_engine import LLMEngine

class Orchestrator:
    def __init__(self, coordinator: Coordinator, transport: Transport, llm_engine: Optional['LLMEngine'] = None) -> None:
        self.coordinator = coordinator
        self.transport = transport
        self.llm_engine = llm_engine

    def distribute(
        self,
        question: str,
        domain: Optional[str] = None,
        recursion_budget: int = 2,
        constraints: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
    ) -> List[QueryResponse]:
        request = self.coordinator.build_query(
            question=question,
            domain=domain,
            recursion_budget=recursion_budget,
            constraints=constraints,
        )
        route_request = RouteRequest(query_id=request.query_id, domain=domain, limit=max_workers)
        route_response = self.transport.route(route_request)
        return [self.transport.send_query(node_id, request) for node_id in route_response.node_ids]

    def decompose_and_distribute(self, question: str, max_cycles: int = 2) -> Dict[str, Any]:
        """
        Implements ComoRAG-inspired task decomposition and distributed execution with iterative cycles.
        """
        if not self.llm_engine:
            raise RuntimeError("LLMEngine is required for task decomposition.")

        print(f"Starting decomposition for: {question}")
        
        current_context = question
        all_results = []
        final_answer = ""
        
        for cycle in range(max_cycles):
            print(f"--- Cycle {cycle + 1}/{max_cycles} ---")
            
            # 1. Decompose
            sub_tasks_desc = self.llm_engine.decompose_task(current_context)
            if not sub_tasks_desc:
                print("No sub-tasks generated. Stopping cycles.")
                break
                
            # 2. Assign Roles
            assignments_data = self.llm_engine.assign_roles(sub_tasks_desc)
            
            # 3. Create Task Objects
            tasks = []
            print(f"[Orchestrator] Generated {len(assignments_data)} sub-tasks:")
            for item in assignments_data:
                task = Task(
                    task_id=str(uuid.uuid4()),
                    description=item['task'],
                    role=item['role'],
                    status="pending"
                )
                tasks.append(task)
                print(f"  - Task: {task.description[:50]}... (Role: {task.role})")
            
            # 4. Distribute to Workers
            available_nodes = self.transport.available_nodes()
            
            # Ensure local workers are included if not already
            local_workers = list(self.transport._workers.keys())
            for worker_id in local_workers:
                if worker_id not in available_nodes:
                    available_nodes.append(worker_id)
            
            if not available_nodes:
                 print("Critical: No workers (local or remote) available to process tasks.")
                 break
            
            print(f"Distributing tasks to {len(available_nodes)} nodes: {available_nodes}")

            cycle_results: List[TaskResult] = []
            threads = []
            results_lock = threading.Lock()

            def dispatch(node_id: str, assignment_msg: TaskAssignment):
                try:
                    # Determine sender hash safely
                    sender_hash = getattr(self.transport, '_destination_hash_hex', None)
                    
                    # Update assignment with sender info
                    assignment_msg.sender_node_id = self.transport.node_id
                    assignment_msg.sender_hash = sender_hash

                    result = self.transport.send_task(node_id, assignment_msg)
                    with results_lock:
                        cycle_results.append(result)
                    print(f"Received result from {node_id}: {result.result[:100]}...")
                except Exception as e:
                    print(f"Error executing task on {node_id}: {e}")

            if available_nodes:
                for i, task in enumerate(tasks):
                    node_id = available_nodes[i % len(available_nodes)]
                    
                    is_local = node_id in self.transport._workers
                    location = "LOCAL" if is_local else "REMOTE"
                    print(f"[Orchestrator] Dispatching task {task.task_id} to {node_id} ({location})")

                    assignment_msg = TaskAssignment(
                        assignment_id=str(uuid.uuid4()),
                        task=task,
                        assigned_to_node=node_id,
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        sender_node_id=self.transport.node_id,
                        sender_hash=getattr(self.transport, '_destination_hash_hex', None)
                    )
                    
                    t = threading.Thread(target=dispatch, args=(node_id, assignment_msg))
                    threads.append(t)
                    t.start()
                    
                print(f"[Orchestrator] Waiting for {len(tasks)} tasks to complete...")
                for t in threads:
                    t.join()
            else:
                 print("No available workers to distribute tasks to.")
                 break
            
            all_results.extend(cycle_results)
            
            # 5. Consolidate & Check for Next Cycle
            print(f"[Orchestrator] All tasks completed. Synthesizing results from {len(cycle_results)} tasks...")
            cycle_summary = "\n".join([f"Task: {r.task_id} | Result: {r.result}" for r in cycle_results])
            
            consolidation_prompt = f"""
You are a project manager. Review the results of the current cycle.
Original Request: {question}
Current Cycle Context: {current_context}
Results:
{cycle_summary}

Determine if the original request is fully satisfied.
If YES, provide the Final Answer.
If NO, provide a refined description of what remains to be done (this will be the input for the next cycle).

Format:
Status: [DONE/CONTINUE]
Content: [Final Answer or Next Steps]
"""
            response = self.llm_engine.generate(consolidation_prompt)
            
            status_line = next((line for line in response.split('\n') if line.startswith("Status:")), "Status: DONE")
            print(f"[Orchestrator] Cycle {cycle + 1} Analysis: {status_line}")
            content_start = response.find("Content:")
            content = response[content_start + 8:].strip() if content_start != -1 else response

            if "DONE" in status_line:
                final_answer = content
                print(f"Task completed. Final Answer: {final_answer[:200]}...")
                break
            else:
                current_context = content
                print(f"Continuing to next cycle with context: {current_context[:50]}...")

        if not final_answer:
            final_answer = "Task ended without explicit completion."

        return {
            "original_request": question,
            "sub_tasks_count": len(all_results),
            "results": [r.to_dict() for r in all_results],
            "final_answer": final_answer
        }
