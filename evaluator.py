#!/usr/bin/env python3
"""
Reward Evaluator Module

Calculates tiered rewards for generated SPICE netlists.
The evaluator uses the same reward system as the training pipeline.
"""

from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from circuit_utils import (
    extract_netlist,
    has_netlist_block,
    check_netlist_valid,
    calculate_tiered_reward,
)
from .observer import SimulationResult


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    reward: float
    tier: int
    reason: str
    has_netlist_block: bool
    netlist_valid: bool
    compiles: bool
    simulates: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "reward": self.reward,
            "tier": self.tier,
            "reason": self.reason,
            "has_netlist_block": self.has_netlist_block,
            "netlist_valid": self.netlist_valid,
            "compiles": self.compiles,
            "simulates": self.simulates,
            "details": self.details,
        }


class RewardEvaluator:
    """
    Evaluates generated netlists and calculates tiered rewards.
    
    Reward Tiers:
        Tier 0 (0.00): No netlist block found
        Tier 1 (0.05): Netlist block found but invalid (empty/only comments)
        Tier 2 (0.50): Netlist compiles successfully
        Tier 3 (0.50+): Simulation runs and produces valid output
    
    The evaluator provides:
    1. Tiered reward calculation
    2. Detailed failure analysis
    3. Simulation result integration
    """

    def __init__(self, run_simulation: bool = True):
        """
        Initialize the reward evaluator.
        
        Args:
            run_simulation: Whether to run full simulation for evaluation
        """
        self.run_simulation = run_simulation

    def evaluate(
        self,
        response: str,
        prompt: str,
        circuit_type: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        sim_result: Optional[SimulationResult] = None,
    ) -> EvaluationResult:
        """
        Evaluate a generated netlist.
        
        Args:
            response: Full LLM response (may contain text + code block)
            prompt: Original user prompt
            circuit_type: Circuit type (optional, will be inferred)
            params: Circuit parameters
            sim_result: Pre-computed simulation result (optional)
            
        Returns:
            EvaluationResult with reward and analysis
        """
        # Parse specs from prompt if not provided
        if circuit_type is None or params is None:
            parsed = self._parse_specs_from_prompt(prompt)
            circuit_type = circuit_type or parsed.get('circuit_type', 'filter')
            params = params or parsed.get('parameters', {})
        
        # Check for netlist block
        has_block = has_netlist_block(response)
        
        if not has_block:
            return EvaluationResult(
                reward=0.0,
                tier=0,
                reason="No netlist block found",
                has_netlist_block=False,
                netlist_valid=False,
                compiles=False,
                simulates=False,
                details={"error": "Missing ```spice code block"},
            )
        
        # Extract netlist
        netlist = extract_netlist(response)
        
        # Check if netlist is valid (has components)
        is_valid = check_netlist_valid(netlist)
        
        if not is_valid:
            return EvaluationResult(
                reward=0.05,
                tier=1,
                reason="Netlist block present but invalid or empty",
                has_netlist_block=True,
                netlist_valid=False,
                compiles=False,
                simulates=False,
                details={"error": "Netlist contains no components"},
            )
        
        # Use tiered reward system
        reward, reason = calculate_tiered_reward(
            response,
            circuit_type,
            params,
            run_sim=self.run_simulation,
        )
        
        # Determine tier from reward
        tier = self._reward_to_tier(reward, reason)
        
        # Parse detailed status
        details = self._parse_reward_details(reason, sim_result)
        
        return EvaluationResult(
            reward=reward,
            tier=tier,
            reason=reason,
            has_netlist_block=True,
            netlist_valid=True,
            compiles=tier >= 2,
            simulates=tier >= 3,
            details=details,
        )

    def _reward_to_tier(self, reward: float, reason: str) -> int:
        """Convert reward value to tier number"""
        if reward == 0.0:
            return 0
        elif reward <= 0.1:
            return 1
        elif reward <= 0.5:
            return 2
        else:
            return 3

    def _parse_specs_from_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Heuristic parser to recover specs from the natural language prompt.
        Matches the patterns in generator.py and train.py
        """
        import re
        
        specs = {
            'circuit_type': None,
            'parameters': {}
        }
        
        prompt_lower = prompt.lower()
        
        # Identify Type
        if 'low-pass' in prompt_lower or 'lowpass' in prompt_lower:
            specs['circuit_type'] = 'filter'
            specs['parameters']['filter_type'] = 'low'
        elif 'high-pass' in prompt_lower or 'highpass' in prompt_lower:
            specs['circuit_type'] = 'filter'
            specs['parameters']['filter_type'] = 'high'
        elif 'band-pass' in prompt_lower or 'bandpass' in prompt_lower:
            specs['circuit_type'] = 'filter'
            specs['parameters']['filter_type'] = 'band'
        elif 'amplifier' in prompt_lower:
            specs['circuit_type'] = 'amplifier'
        elif 'oscillator' in prompt_lower:
            specs['circuit_type'] = 'oscillator'
        elif 'clipper' in prompt_lower:
            specs['circuit_type'] = 'clipper'
        elif 'clamper' in prompt_lower:
            specs['circuit_type'] = 'clamper'
        
        # Frequency extraction
        freq_match = re.search(r'([\d\.]+)\s*(k|m|g)?hz', prompt_lower)
        if freq_match:
            val = float(freq_match.group(1))
            unit = freq_match.group(2)
            if unit == 'k':
                val *= 1e3
            if unit == 'm':
                val *= 1e6
            if unit == 'g':
                val *= 1e9
            
            if specs['circuit_type'] == 'filter':
                specs['parameters']['cutoff_freq'] = val
            if specs['circuit_type'] == 'oscillator':
                specs['parameters']['freq_hz'] = val
        
        # Gain extraction
        gain_db_match = re.search(r'([\d\.]+)\s*db', prompt_lower)
        if gain_db_match:
            val = float(gain_db_match.group(1))
            if specs['circuit_type'] == 'filter':
                specs['parameters']['passband_gain_db'] = val
            if specs['circuit_type'] == 'amplifier':
                specs['parameters']['gain_db'] = val
        
        # Defaults
        if specs['circuit_type'] == 'filter':
            specs['parameters'].setdefault('cutoff_freq', 1000)
            specs['parameters'].setdefault('passband_gain_db', 0)
        if specs['circuit_type'] == 'amplifier':
            specs['parameters'].setdefault('gain_db', 20)
        if specs['circuit_type'] == 'oscillator':
            specs['parameters'].setdefault('freq_hz', 1000)
            specs['parameters'].setdefault('v_peak', 1.0)
        if specs['circuit_type'] == 'clipper':
            specs['parameters'].setdefault('clip_voltage', 5.0)
            specs['parameters'].setdefault('direction', 'positive')
        if specs['circuit_type'] == 'clamper':
            specs['parameters'].setdefault('v_in_peak', 5.0)
            specs['parameters'].setdefault('clamp_level', 0.0)
            specs['parameters'].setdefault('direction', 'positive')
        
        return specs

    def _parse_reward_details(
        self,
        reason: str,
        sim_result: Optional[SimulationResult],
    ) -> Dict[str, Any]:
        """Parse detailed information from reward reason"""
        details = {
            "reason": reason,
        }
        
        # Add simulation details if available
        if sim_result:
            details["simulation"] = sim_result.to_dict()
        
        # Parse specific failure modes
        reason_lower = reason.lower()
        
        if "no netlist" in reason_lower:
            details["failure_mode"] = "missing_code_block"
        elif "invalid" in reason_lower or "empty" in reason_lower:
            details["failure_mode"] = "empty_netlist"
        elif "compile" in reason_lower:
            details["failure_mode"] = "syntax_error"
        elif "simulation" in reason_lower or "failed" in reason_lower:
            details["failure_mode"] = "simulation_error"
        elif "flat" in reason_lower:
            details["failure_mode"] = "flat_output"
        else:
            details["failure_mode"] = "unknown"
        
        return details

    def quick_evaluate(
        self,
        response: str,
        prompt: str,
    ) -> Tuple[float, str]:
        """
        Quick evaluation without simulation (for speed).
        
        Returns:
            (reward, reason) tuple
        """
        # Temporarily disable simulation
        original_setting = self.run_simulation
        self.run_simulation = False
        
        result = self.evaluate(response, prompt)
        
        # Restore setting
        self.run_simulation = original_setting
        
        return result.reward, result.reason

    def get_success_metrics(self, eval_result: EvaluationResult) -> Dict[str, bool]:
        """
        Get success metrics for different thresholds.
        
        Returns dict with:
            - has_netlist: Has code block
            - is_valid: Has components
            - compiles: Syntax valid
            - simulates: Produces output
            - is_good: Reward >= 0.5
            - is_excellent: Reward >= 0.8
        """
        return {
            "has_netlist": eval_result.has_netlist_block,
            "is_valid": eval_result.netlist_valid,
            "compiles": eval_result.compiles,
            "simulates": eval_result.simulates,
            "is_good": eval_result.reward >= 0.5,
            "is_excellent": eval_result.reward >= 0.8,
        }
