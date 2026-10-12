#!/usr/bin/env python3
"""
SPICE Agent - Main Orchestration Module

Coordinates the planning, acting, observation, evaluation, and reflection
cycle for iterative SPICE netlist generation.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .planner import CircuitPlanner, CircuitPlan
from .actor import NetlistActor
from .observer import SimulationObserver, SimulationResult
from .evaluator import RewardEvaluator, EvaluationResult
from .reflector import FailureReflector, Diagnosis
from .memory import AgentMemory, EpisodeMemory

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """Output from agent generation"""
    netlist: str
    full_response: str
    reward: float
    iterations: int
    success: bool
    final_reason: str
    episode_memory: Optional[EpisodeMemory]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "netlist": self.netlist,
            "full_response": self.full_response,
            "reward": self.reward,
            "iterations": self.iterations,
            "success": self.success,
            "final_reason": self.final_reason,
        }


class SpiceAgent:
    """
    Main agent for iterative SPICE netlist generation.
    
    The agent follows a PLAN-ACT-OBSERVE-EVALUATE-REFLECT-REFINE loop:
    
    1. PLAN: Analyze requirements and create circuit plan
    2. ACT: Generate netlist using LLM
    3. OBSERVE: Run simulation
    4. EVALUATE: Calculate reward
    5. REFLECT: Diagnose failures
    6. REFINE: Update plan and retry if needed
    
    The loop continues until:
    - Reward >= success_threshold, OR
    - Max iterations reached
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-4b:free",
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        success_threshold: float = 0.8,
        temperature: float = 0.2,
        run_simulation: bool = True,
        ngspice_path: str = "ngspice",
        verbose: bool = True,
        fallback_models: Optional[List[str]] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the SPICE agent.

        Args:
            model: Model identifier (OpenRouter or local)
            api_key: API key (or set OPENROUTER_API_KEY env var)
            max_iterations: Maximum refinement iterations
            success_threshold: Reward threshold for success (0.0-1.0)
            temperature: LLM sampling temperature
            run_simulation: Whether to run full simulation
            ngspice_path: Path to ngspice executable
            verbose: Print progress information
            fallback_models: List of fallback models if primary fails
            base_url: Custom API base URL (for Ollama or other providers)
        """
        self.max_iterations = max_iterations
        self.success_threshold = success_threshold
        self.verbose = verbose
        self.fallback_models = fallback_models or [
            "qwen/qwen3-4b:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "google/gemma-3-4b-it:free",
        ]
        self.current_model = model
        self.base_url = base_url

        # Initialize components
        self.planner = CircuitPlanner()
        self.actor = NetlistActor(
            model=self.current_model,
            api_key=api_key,
            temperature=temperature,
            base_url=self.base_url,
        )
        self.observer = SimulationObserver(ngspice_path=ngspice_path)
        self.evaluator = RewardEvaluator(run_simulation=run_simulation)
        self.reflector = FailureReflector()
        self.memory = AgentMemory()

        # State
        self.current_plan: Optional[CircuitPlan] = None

    def _try_fallback_models(self, api_key: Optional[str], temperature: float) -> bool:
        """Try fallback models if current model fails"""
        for model in self.fallback_models:
            if self.verbose:
                print(f"  Trying fallback model: {model}")
            try:
                self.actor = NetlistActor(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    base_url=self.base_url,
                )
                self.current_model = model
                return True
            except Exception:
                continue
        return False

    def generate(self, prompt: str) -> AgentOutput:
        """
        Generate a SPICE netlist through iterative refinement.
        
        Args:
            prompt: Natural language circuit specification
            
        Returns:
            AgentOutput with final netlist and metadata
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting agent for prompt: {prompt[:80]}...")
            print(f"{'='*60}")
        
        # PHASE 1: PLAN
        if self.verbose:
            print("\n[PHASE 1] Planning...")
        
        plan = self.planner.analyze(prompt)
        self.current_plan = plan
        
        # Start episode memory
        self.memory.start_episode(
            prompt=prompt,
            circuit_type=plan.circuit_type,
            specifications=plan.specifications,
        )
        
        if self.verbose:
            print(f"  Circuit Type: {plan.circuit_type}")
            print(f"  Topology: {plan.topology}")
            print(f"  Specifications: {plan.specifications}")
        
        # Iteration loop
        best_netlist = ""
        best_reward = 0.0
        best_response = ""
        final_reason = ""
        
        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n[ITERATION {iteration}/{self.max_iterations}]")
            
            # PHASE 2: ACT
            if self.verbose:
                print("  [PHASE 2] Generating netlist...")
            
            # Get previous context for refinement
            previous_response = best_response if iteration > 1 else None
            feedback = None
            
            if iteration > 1:
                # Get feedback from previous iteration
                feedback = self._get_feedback_for_iteration(iteration - 1)
            
            response = self.actor.generate(
                plan=plan,
                previous_response=previous_response,
                feedback=feedback,
            )
            
            # Extract netlist
            netlist = self.actor.extract_netlist(response)
            
            if self.verbose:
                has_block = "Yes" if netlist else "No"
                print(f"  Netlist extracted: {has_block}")
            
            # PHASE 3: OBSERVE
            if self.verbose:
                print("  [PHASE 3] Running simulation...")
            
            sim_result = self.observer.run(
                netlist=netlist if netlist else "* Empty netlist",
                circuit_type=plan.circuit_type,
                params=plan.specifications,
            )
            
            if self.verbose:
                sim_status = "Success" if sim_result.success else f"Failed: {sim_result.error_message[:50]}"
                print(f"  Simulation: {sim_status}")
            
            # PHASE 4: EVALUATE
            if self.verbose:
                print("  [PHASE 4] Evaluating...")
            
            eval_result = self.evaluator.evaluate(
                response=response,
                prompt=prompt,
                circuit_type=plan.circuit_type,
                params=plan.specifications,
                sim_result=sim_result,
            )
            
            if self.verbose:
                print(f"  Reward: {eval_result.reward:.2f} (Tier {eval_result.tier})")
                print(f"  Reason: {eval_result.reason[:60]}")
            
            # Track in memory
            self.memory.add_iteration(
                iteration=iteration,
                response=response,
                netlist=netlist,
                reward=eval_result.reward,
                reason=eval_result.reason,
                failure_mode=eval_result.details.get("failure_mode", ""),
                error_message=sim_result.error_message if not sim_result.success else "",
            )
            
            # Update best
            if eval_result.reward > best_reward:
                best_reward = eval_result.reward
                best_netlist = netlist
                best_response = response
            
            # Check for success
            if eval_result.reward >= self.success_threshold:
                if self.verbose:
                    print(f"\n✓ Success! Reward {eval_result.reward:.2f} >= {self.success_threshold}")
                
                self.memory.mark_success()
                final_reason = eval_result.reason
                break
            
            # PHASE 5: REFLECT
            if self.verbose:
                print("  [PHASE 5] Reflecting...")
            
            diagnosis = self.reflector.diagnose(
                eval_result=eval_result,
                sim_result=sim_result,
                circuit_type=plan.circuit_type,
            )
            
            if self.verbose:
                print(f"  Failure Mode: {diagnosis.failure_mode}")
                print(f"  Confidence: {diagnosis.confidence:.2f}")
                if diagnosis.suggested_fixes:
                    print(f"  Top Fix: {diagnosis.suggested_fixes[0][:60]}")
            
            # PHASE 6: REFINE
            if iteration < self.max_iterations:
                if self.verbose:
                    print("  [PHASE 6] Refining plan...")
                
                plan = self.planner.refine(
                    previous_plan=plan,
                    diagnosis=diagnosis.to_dict(),
                    iteration=iteration,
                )
            
            final_reason = eval_result.reason
        
        # End episode
        episode = self.memory.end_episode()
        
        # Create output
        output = AgentOutput(
            netlist=best_netlist,
            full_response=best_response,
            reward=best_reward,
            iterations=len(episode.iterations) if episode else 0,
            success=best_reward >= self.success_threshold,
            final_reason=final_reason,
            episode_memory=episode,
        )
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Agent completed: {output.iterations} iterations, reward={output.reward:.2f}")
            print(f"{'='*60}")
        
        return output

    def _get_feedback_for_iteration(self, previous_iteration: int) -> Dict[str, Any]:
        """Get feedback from a previous iteration for refinement"""
        if self.memory.current_episode is None:
            return {}
        
        episode = self.memory.current_episode
        if previous_iteration > len(episode.iterations):
            return {}
        
        record = episode.iterations[previous_iteration - 1]
        
        # Create diagnosis from record
        from .evaluator import EvaluationResult
        eval_result = EvaluationResult(
            reward=record.reward,
            tier=0,
            reason=record.reason,
            has_netlist_block=bool(record.netlist),
            netlist_valid=bool(record.netlist),
            compiles=record.failure_mode not in ["syntax_error", "empty_netlist"],
            simulates=record.failure_mode == "flat_output" or record.reward > 0.5,
            details={"failure_mode": record.failure_mode},
        )
        
        diagnosis = self.reflector.diagnose(
            eval_result=eval_result,
            sim_result=None,
            circuit_type=self.current_plan.circuit_type if self.current_plan else "",
        )
        feedback = self.reflector.generate_feedback(diagnosis, previous_iteration)
        
        return feedback

    def generate_batch(
        self,
        prompts: list,
        progress_callback=None,
    ) -> list:
        """
        Generate netlists for multiple prompts.
        
        Args:
            prompts: List of prompt dictionaries with 'prompt' key
            progress_callback: Optional callback(current, total)
            
        Returns:
            List of AgentOutput objects
        """
        results = []
        total = len(prompts)
        
        for i, item in enumerate(prompts):
            prompt = item.get("prompt", "") if isinstance(item, dict) else item
            
            if self.verbose:
                print(f"\n\n>>> Processing prompt {i+1}/{total}")
            
            output = self.generate(prompt)
            results.append(output)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics from memory"""
        return self.memory.get_statistics()

    def export_results(self, filepath: str) -> None:
        """Export all episode results to JSON"""
        self.memory.export_to_json(filepath)
        if self.verbose:
            print(f"Results exported to {filepath}")

    def set_model(self, model: str) -> None:
        """Change the LLM model"""
        self.actor.model = model
        if self.verbose:
            print(f"Model changed to: {model}")

    def set_max_iterations(self, max_iterations: int) -> None:
        """Change maximum iterations"""
        self.max_iterations = max_iterations
        if self.verbose:
            print(f"Max iterations set to: {max_iterations}")

    def set_success_threshold(self, threshold: float) -> None:
        """Change success threshold"""
        self.success_threshold = max(0.0, min(1.0, threshold))
        if self.verbose:
            print(f"Success threshold set to: {threshold}")
