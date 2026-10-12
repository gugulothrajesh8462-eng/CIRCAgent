#!/usr/bin/env python3
"""
Agent Memory Module

Tracks iteration history, prevents repeating mistakes,
and maintains context across refinement cycles.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IterationRecord:
    """Record of a single iteration"""
    iteration: int
    prompt: str
    response: str
    netlist: str
    reward: float
    reason: str
    failure_mode: str
    error_message: str
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "iteration": self.iteration,
            "prompt": self.prompt,
            "response": self.response,
            "netlist": self.netlist,
            "reward": self.reward,
            "reason": self.reason,
            "failure_mode": self.failure_mode,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IterationRecord":
        """Create from dictionary"""
        return cls(
            iteration=data.get("iteration", 0),
            prompt=data.get("prompt", ""),
            response=data.get("response", ""),
            netlist=data.get("netlist", ""),
            reward=data.get("reward", 0.0),
            reason=data.get("reason", ""),
            failure_mode=data.get("failure_mode", ""),
            error_message=data.get("error_message", ""),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class EpisodeMemory:
    """Complete memory for one prompt episode"""
    prompt: str
    circuit_type: str
    specifications: Dict[str, Any]
    iterations: List[IterationRecord] = field(default_factory=list)
    best_netlist: str = ""
    best_reward: float = 0.0
    success: bool = False
    total_iterations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prompt": self.prompt,
            "circuit_type": self.circuit_type,
            "specifications": self.specifications,
            "iterations": [i.to_dict() for i in self.iterations],
            "best_netlist": self.best_netlist,
            "best_reward": self.best_reward,
            "success": self.success,
            "total_iterations": self.total_iterations,
        }


class AgentMemory:
    """
    Manages memory for the SPICE generation agent.
    
    Features:
    1. Tracks iteration history per prompt
    2. Stores best netlist and reward
    3. Detects repeated mistakes
    4. Provides context for refinement
    5. Exports/imports for analysis
    """

    def __init__(self, max_history_per_episode: int = 10):
        """
        Initialize agent memory.
        
        Args:
            max_history_per_episode: Maximum iterations to store per episode
        """
        self.max_history = max_history_per_episode
        
        # Current episode memory
        self.current_episode: Optional[EpisodeMemory] = None
        
        # Completed episodes
        self.episode_history: List[EpisodeMemory] = []
        
        # Statistics
        self.total_prompts = 0
        self.total_successful = 0
        self.total_iterations = 0

    def start_episode(
        self,
        prompt: str,
        circuit_type: str,
        specifications: Dict[str, Any],
    ) -> None:
        """
        Start a new episode for a prompt.
        
        Args:
            prompt: User prompt
            circuit_type: Type of circuit
            specifications: Circuit parameters
        """
        self.current_episode = EpisodeMemory(
            prompt=prompt,
            circuit_type=circuit_type,
            specifications=specifications,
        )

    def add_iteration(
        self,
        iteration: int,
        response: str,
        netlist: str,
        reward: float,
        reason: str,
        failure_mode: str = "",
        error_message: str = "",
    ) -> None:
        """
        Add an iteration record to current episode.
        
        Args:
            iteration: Iteration number (1-indexed)
            response: Full LLM response
            netlist: Extracted netlist
            reward: Reward score
            reason: Reward reason
            failure_mode: Categorized failure mode
            error_message: Simulation error if any
        """
        if self.current_episode is None:
            raise ValueError("No active episode. Call start_episode() first.")
        
        # Create record
        record = IterationRecord(
            iteration=iteration,
            prompt=self.current_episode.prompt,
            response=response,
            netlist=netlist,
            reward=reward,
            reason=reason,
            failure_mode=failure_mode,
            error_message=error_message,
        )
        
        # Add to episode
        if len(self.current_episode.iterations) < self.max_history:
            self.current_episode.iterations.append(record)
        
        # Update best if improved
        if reward > self.current_episode.best_reward:
            self.current_episode.best_reward = reward
            self.current_episode.best_netlist = netlist
        
        # Update total iterations
        self.current_episode.total_iterations = iteration

    def mark_success(self) -> None:
        """Mark current episode as successful"""
        if self.current_episode:
            self.current_episode.success = True
            self.total_successful += 1

    def end_episode(self) -> Optional[EpisodeMemory]:
        """
        End current episode and store in history.
        
        Returns:
            Completed episode memory
        """
        if self.current_episode is None:
            return None
        
        # Store in history
        self.episode_history.append(self.current_episode)
        self.total_prompts += 1
        self.total_iterations += self.current_episode.total_iterations
        
        # Return and clear
        episode = self.current_episode
        self.current_episode = None
        return episode

    def get_context_for_refinement(self) -> Dict[str, Any]:
        """
        Get context for the next refinement iteration.
        
        Returns:
            Context dictionary with previous attempts and patterns
        """
        if self.current_episode is None:
            return {"error": "No active episode"}
        
        episode = self.current_episode
        
        # Get previous attempts
        previous_attempts = []
        for record in episode.iterations:
            previous_attempts.append({
                "iteration": record.iteration,
                "netlist": record.netlist,
                "reward": record.reward,
                "reason": record.reason,
                "failure_mode": record.failure_mode,
            })
        
        # Detect patterns
        patterns = self._detect_patterns(episode.iterations)
        
        return {
            "prompt": episode.prompt,
            "circuit_type": episode.circuit_type,
            "specifications": episode.specifications,
            "current_iteration": len(episode.iterations),
            "best_reward_so_far": episode.best_reward,
            "previous_attempts": previous_attempts,
            "patterns": patterns,
            "mistakes_to_avoid": self._get_mistakes_to_avoid(episode.iterations),
        }

    def _detect_patterns(
        self,
        iterations: List[IterationRecord],
    ) -> Dict[str, Any]:
        """Detect patterns in iteration history"""
        if len(iterations) < 2:
            return {}
        
        # Count failure modes
        failure_counts = {}
        for record in iterations:
            mode = record.failure_mode
            failure_counts[mode] = failure_counts.get(mode, 0) + 1
        
        # Check reward trend
        rewards = [r.reward for r in iterations]
        improving = all(rewards[i] <= rewards[i+1] for i in range(len(rewards)-1))
        
        # Check for repeated same failure
        repeated_failure = None
        if len(iterations) >= 2:
            last_mode = iterations[-1].failure_mode
            if all(r.failure_mode == last_mode for r in iterations[-2:]):
                repeated_failure = last_mode
        
        return {
            "failure_counts": failure_counts,
            "reward_trend": rewards,
            "improving": improving,
            "repeated_failure": repeated_failure,
        }

    def _get_mistakes_to_avoid(
        self,
        iterations: List[IterationRecord],
    ) -> List[str]:
        """Get list of mistakes to avoid based on history"""
        mistakes = []
        
        # Check for repeated failures
        failure_modes = [r.failure_mode for r in iterations]
        
        if failure_modes.count("missing_code_block") > 1:
            mistakes.append("Always include ```spice code block")
        
        if failure_modes.count("syntax_error") > 1:
            mistakes.append("Double-check SPICE syntax before outputting")
        
        if failure_modes.count("simulation_error") > 1:
            mistakes.append("Ensure Vin source and ground connection are present")
        
        if failure_modes.count("flat_output") > 1:
            mistakes.append("Verify signal path from input to output")
        
        # Check for specific error patterns
        error_messages = [r.error_message.lower() for r in iterations if r.error_message]
        
        if any("undefined node" in e for e in error_messages):
            mistakes.append("Define all nodes - no floating connections")
        
        if any("singular matrix" in e for e in error_messages):
            mistakes.append("Avoid floating nodes - add ground reference")
        
        return mistakes

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        # Calculate average iterations
        avg_iterations = (
            self.total_iterations / self.total_prompts
            if self.total_prompts > 0 else 0
        )
        
        # Calculate success rate
        success_rate = (
            self.total_successful / self.total_prompts
            if self.total_prompts > 0 else 0
        )
        
        # Get reward distribution from episode history
        rewards = [e.best_reward for e in self.episode_history]
        
        return {
            "total_prompts": self.total_prompts,
            "total_successful": self.total_successful,
            "total_iterations": self.total_iterations,
            "average_iterations": avg_iterations,
            "success_rate": success_rate,
            "average_reward": sum(rewards) / len(rewards) if rewards else 0,
            "best_reward": max(rewards) if rewards else 0,
            "worst_reward": min(rewards) if rewards else 0,
        }

    def export_to_json(self, filepath: str) -> None:
        """Export all episode history to JSON"""
        data = {
            "statistics": self.get_statistics(),
            "episodes": [e.to_dict() for e in self.episode_history],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, filepath: str) -> None:
        """Import episode history from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.episode_history = [
            EpisodeMemory(
                prompt=e["prompt"],
                circuit_type=e["circuit_type"],
                specifications=e["specifications"],
                iterations=[
                    IterationRecord.from_dict(i)
                    for i in e.get("iterations", [])
                ],
                best_netlist=e.get("best_netlist", ""),
                best_reward=e.get("best_reward", 0),
                success=e.get("success", False),
                total_iterations=e.get("total_iterations", 0),
            )
            for e in data.get("episodes", [])
        ]
        
        # Update statistics
        stats = data.get("statistics", {})
        self.total_prompts = stats.get("total_prompts", len(self.episode_history))
        self.total_successful = stats.get("total_successful", 0)
        self.total_iterations = stats.get("total_iterations", 0)

    def get_episode_summary(self, episode: EpisodeMemory) -> str:
        """Generate human-readable summary of an episode"""
        lines = [
            f"Prompt: {episode.prompt[:100]}...",
            f"Circuit Type: {episode.circuit_type}",
            f"Success: {'Yes' if episode.success else 'No'}",
            f"Iterations: {episode.total_iterations}",
            f"Best Reward: {episode.best_reward:.2f}",
            "",
            "Iteration History:",
        ]
        
        for record in episode.iterations:
            lines.append(
                f"  {record.iteration}. Reward={record.reward:.2f}, "
                f"Mode={record.failure_mode}, Reason={record.reason[:50]}"
            )
        
        return "\n".join(lines)
