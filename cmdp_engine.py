"""
Cascading Markov Decision Process (CMDP) Engine
================================================
This is the core reference implementation of the CMDP architecture.
It demonstrates the MDP loop, AST-based context compression, episodic memory,
Git-based state rollback, and the multi-agent critic system.

Note: External dependencies (like actual LLM API calls or Tree-sitter binaries) 
are abstracted into mock classes to ensure this file is structurally complete 
and runnable for architectural demonstration.
"""

import ast
import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
import shutil

# ==============================================================================
# 1. DATA MODELS (The State Space)
# ==============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    description: str
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    file_path: str = ""

@dataclass
class MemoryEntry:
    task_id: str
    action_taken: str
    outcome: str
    lesson_learned: str

@dataclass
class MDPState:
    """Represents the current state (S_t) of the Markov Decision Process."""
    original_prompt: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    codebase: Dict[str, str] = field(default_factory=dict) # file_path -> code
    memory_scratchpad: List[MemoryEntry] = field(default_factory=list)
    current_task_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

# ==============================================================================
# 2. AST ENGINE (Context Compression & Validation)
# ==============================================================================

class ASTEngine:
    """
    Handles Abstract Syntax Tree operations. 
    In production, this would wrap Tree-sitter for multi-language support.
    Here, we use Python's built-in `ast` module for demonstration.
    """
    
    @staticmethod
    def generate_repo_map(code: str) -> str:
        """
        Compresses code into a 'Repo Map' by stripping function bodies.
        This allows the LLM to understand the API without token bloat.
        """
        try:
            tree = ast.parse(code)
            repo_map_lines = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    repo_map_lines.append(f"class {node.name}:")
                elif isinstance(node, ast.FunctionDef):
                    # Extract arguments for the signature
                    args = [arg.arg for arg in node.args.args]
                    repo_map_lines.append(f"    def {node.name}({', '.join(args)}): ...")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        repo_map_lines.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    names = [n.name for n in node.names]
                    repo_map_lines.append(f"from {node.module} import {', '.join(names)}")
                    
            return "\n".join(repo_map_lines)
        except SyntaxError:
            return "# [AST Error: Could not parse code for Repo Map]"

    @staticmethod
    def validate_snippet(snippet: str, expected_imports: List[str]) -> Tuple[bool, str]:
        """
        The AST Critic: Validates structural integrity before execution.
        """
        try:
            tree = ast.parse(snippet)
        except SyntaxError as e:
            return False, f"Syntax Error at line {e.lineno}: {e.msg}"

        # Check imports
        actual_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                actual_imports.extend([n.name for n in node.names])
            elif isinstance(node, ast.ImportFrom):
                actual_imports.append(node.module)

        missing = [imp for imp in actual_imports if imp not in expected_imports and imp is not None]
        if missing:
            return False, f"AST Critic: Attempted to import non-existent modules: {missing}"

        return True, "AST Validation Passed."

# ==============================================================================
# 3. AGENTS (The Actors in the MDP)
# ==============================================================================

class LLMClient:
    """Mock LLM Client. Replace with OpenAI/Anthropic API calls in production."""
    def generate(self, prompt: str) -> str:
        # Mocking a successful code generation
        return 'import math\n\ndef calculate_area(radius):\n    return math.pi * (radius ** 2)'

class ArchitectAgent:
    def create_plan(self, prompt: str) -> Dict[str, Task]:
        print(f"[Architect] Analyzing prompt: '{prompt}'")
        # Mocking a dependency graph creation
        return {
            "task_1": Task(id="task_1", description="Create math utility", dependencies=[], file_path="math_utils.py"),
            "task_2": Task(id="task_2", description="Create main execution", dependencies=["task_1"], file_path="main.py")
        }

class CoderAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_snippet(self, task: Task, state: MDPState) -> str:
        print(f"[Coder] Generating snippet for: {task.description}")
        
        # 1. Build Context (Repo Map + Memory)
        context = "EXISTING CODEBASE (REPO MAP):\n"
        for path, code in state.codebase.items():
            context += f"--- {path} ---\n{ASTEngine.generate_repo_map(code)}\n\n"
            
        memory_context = "LESSONS LEARNED (SCRATCHPAD):\n"
        for entry in state.memory_scratchpad:
            memory_context += f"- {entry.lesson_learned}\n"

        prompt = f"""
        {context}
        {memory_context}
        TASK: {task.description}
        Generate the code for {task.file_path}.
        """
        
        return self.llm.generate(prompt)

class CriticEnsemble:
    """Evaluates the Reward (R) for the MDP."""
    
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(self, snippet: str, task: Task, state: MDPState) -> Tuple[bool, str]:
        print(f"[Critic Ensemble] Evaluating snippet for: {task.description}")
        
        # 1. AST Critic (Structural)
        is_valid, msg = ASTEngine.validate_snippet(snippet, expected_imports=["math", "os"])
        if not is_valid:
            return False, msg

        # 2. Red Team Agent (Adversarial - Mocked)
        print("[Red Team] Testing for edge cases... (Mock: Passed)")
        
        # 3. LLM-as-a-Judge (Semantic - Mocked)
        print("[Judge] Checking semantic alignment with prompt... (Mock: Passed)")

        return True, "All critics passed."

# ==============================================================================
# 4. THE CMDP ENGINE (The State Machine & Loop)
# ==============================================================================

class CascadingMDP:
    def __init__(self, workspace_dir: str = "./cmdp_workspace"):
        self.workspace = workspace_dir
        self.llm = LLMClient()
        self.architect = ArchitectAgent()
        self.coder = CoderAgent(self.llm)
        self.critics = CriticEnsemble(self.llm)
        
        # Initialize workspace and Git
        self._init_workspace()

    def _init_workspace(self):
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)
        os.makedirs(self.workspace)
        subprocess.run(["git", "init"], cwd=self.workspace, capture_output=True)
        subprocess.run(["git", "config", "user.email", "cmdp@ai.com"], cwd=self.workspace, capture_output=True)
        subprocess.run(["git", "config", "user.name", "CMDP"], cwd=self.workspace, capture_output=True)

    def _commit_state(self, task_id: str):
        """Transitions the state by committing to Git."""
        subprocess.run(["git", "add", "."], cwd=self.workspace, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"CMDP: Complete {task_id}"], 
                       cwd=self.workspace, capture_output=True)

    def _rollback_state(self):
        """Penalty transition: Reverts to the last known good state."""
        print("[MDP] ROLLBACK: Reverting to last checkpoint...")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], 
                       cwd=self.workspace, capture_output=True)

    def _get_next_task(self, state: MDPState) -> Optional[Task]:
        """Determines the next action based on the dependency graph."""
        for task in state.tasks.values():
            if task.status == TaskStatus.PENDING:
                # Check if dependencies are met
                deps_met = all(state.tasks[dep].status == TaskStatus.COMPLETED 
                               for dep in task.dependencies)
                if deps_met:
                    return task
        return None

    def run(self, initial_prompt: str):
        """The main MDP Loop."""
        print("="*50)
        print("STARTING CASCADING MDP")
        print("="*50)

        # 1. Initialize State (S_0)
        state = MDPState(original_prompt=initial_prompt)
        state.tasks = self.architect.create_plan(initial_prompt)
        self._commit_state("initial_plan") # Initial checkpoint

        # 2. The Cascade Loop
        while True:
            task = self._get_next_task(state)
            
            # Terminal State Check
            if not task:
                print("\n[MDP] TERMINAL STATE REACHED. All tasks completed.")
                break

            state.current_task_id = task.id
            task.status = TaskStatus.IN_PROGRESS
            print(f"\n--- Processing Task: {task.id} ({task.description}) ---")

            # Execute Action (A_t)
            snippet = self.coder.generate_snippet(task, state)

            # Evaluate Reward (R)
            is_valid, feedback = self.critics.evaluate(snippet, task, state)

            if is_valid:
                # Positive Reward: Transition to S_t+1
                print(f"[MDP] REWARD: Positive. Transitioning state.")
                
                # Write to file system
                file_path = os.path.join(self.workspace, task.file_path)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w") as f:
                    f.write(snippet)
                
                # Update State
                state.codebase[task.file_path] = snippet
                task.status = TaskStatus.COMPLETED
                state.retry_count = 0
                
                # Commit Checkpoint
                self._commit_state(task.id)
                
                # Log to Episodic Memory
                state.memory_scratchpad.append(MemoryEntry(
                    task_id=task.id,
                    action_taken="Generated and validated successfully",
                    outcome="Success",
                    lesson_learned=f"Successfully implemented {task.description}."
                ))
            else:
                # Negative Reward: Penalty & Potential Rollback
                print(f"[MDP] REWARD: Negative. Feedback: {feedback}")
                state.retry_count += 1
                
                if state.retry_count >= state.max_retries:
                    print(f"[MDP] MAX RETRIES EXCEEDED. Rolling back state.")
                    self._rollback_state()
                    
                    # Update memory so we don't try the same bad approach
                    state.memory_scratchpad.append(MemoryEntry(
                        task_id=task.id,
                        action_taken="Failed after multiple retries",
                        outcome="Failure",
                        lesson_learned=f"Failed to implement {task.description}. Error: {feedback}. Try a different approach."
                    ))
                    
                    # Reset task to pending so it can be re-evaluated or re-planned
                    task.status = TaskStatus.PENDING
                    state.retry_count = 0
                    
                    # In a full implementation, we would trigger the Architect 
                    # here to dynamically re-plan the dependency graph.
                else:
                    print(f"[MDP] Retrying... ({state.retry_count}/{state.max_retries})")

        print("\n" + "="*50)
        print("CMDP EXECUTION FINISHED")
        print("="*50)

# ==============================================================================
# 5. ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    user_prompt = "Create a Python application that calculates the area of a circle and prints it."
    
    engine = CascadingMDP(workspace_dir="./my_generated_app")
    engine.run(user_prompt)