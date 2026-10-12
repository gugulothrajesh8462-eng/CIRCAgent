#!/usr/bin/env python3
"""
Failure Reflector Module

Analyzes simulation failures and generates diagnostic feedback
for iterative refinement. The reflector provides actionable guidance
to help the actor fix issues in subsequent iterations.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .evaluator import EvaluationResult
from .observer import SimulationResult


@dataclass
class Diagnosis:
    """Container for failure diagnosis"""
    reward: float
    reason: str
    failure_mode: str
    error_message: str
    suggested_fixes: List[str]
    confidence: float
    circuit_type: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "reward": self.reward,
            "reason": self.reason,
            "failure_mode": self.failure_mode,
            "error_message": self.error_message,
            "suggested_fixes": self.suggested_fixes,
            "confidence": self.confidence,
            "circuit_type": self.circuit_type,
        }


class FailureReflector:
    """
    Analyzes failures and generates diagnostic feedback.
    
    The reflector:
    1. Categorizes the failure mode
    2. Identifies likely causes
    3. Suggests specific fixes
    4. Provides confidence level
    """

    # Failure mode definitions
    FAILURE_MODES = {
        "missing_code_block": {
            "description": "No ```spice code block in response",
            "causes": [
                "Model output conversational text instead of code",
                "Code block formatting incorrect",
            ],
            "fixes": [
                "Output the netlist inside a ```spice code block",
                "Start with ```spice and end with ```",
                "Do not include explanations outside the code block",
            ],
        },
        "empty_netlist": {
            "description": "Code block present but contains no components",
            "causes": [
                "Only comments in netlist",
                "Incomplete response",
                "Model refused to generate circuit",
            ],
            "fixes": [
                "Include actual SPICE components (R, C, L, V, etc.)",
                "Remove excessive comments",
                "Generate a complete circuit topology",
            ],
        },
        "syntax_error": {
            "description": "SPICE syntax errors prevent compilation",
            "causes": [
                "Invalid component names",
                "Missing node connections",
                "Incorrect value formats",
                "Missing .end statement",
            ],
            "fixes": [
                "Component names must start with R, C, L, V, I, E, etc.",
                "Each component needs: Name Node1 Node2 Value",
                "Use proper value suffixes: k=1000, m=0.001, u=1e-6, n=1e-9, p=1e-12",
                "End the netlist with .end",
                "Check for typos in node names",
            ],
        },
        "simulation_error": {
            "description": "Netlist compiles but simulation fails",
            "causes": [
                "Missing input source (Vin)",
                "Floating nodes (not connected to anything)",
                "Missing ground connection (node 0)",
                "Unrealistic component values",
                "Op-amp without power supply",
            ],
            "fixes": [
                "Add an input voltage source: Vin vin 0 AC 1",
                "Ensure every node connects to at least two components",
                "Connect circuit ground to node 0",
                "Use realistic values: R=1k-1M, C=1p-100u",
                "For op-amps, ensure proper biasing",
            ],
        },
        "flat_output": {
            "description": "Simulation runs but output is constant/flat",
            "causes": [
                "Input not connected to circuit",
                "Output node not properly defined",
                "Op-amp saturated or not biased",
                "Component values too extreme",
                "Short circuit in signal path",
            ],
            "fixes": [
                "Verify input source connects to circuit input",
                "Check output node is at correct location",
                "For op-amps, use: E1 out 0 +in -in 1e6",
                "Ensure feedback network is properly connected",
                "Check for accidental shorts to ground",
            ],
        },
        "unknown": {
            "description": "Unrecognized failure mode",
            "causes": ["Could not categorize the error"],
            "fixes": [
                "Review the complete error message",
                "Check SPICE syntax carefully",
                "Verify all connections are valid",
            ],
        },
    }

    # Circuit-specific checks
    CIRCUIT_CHECKS = {
        "filter": [
            "Check RC time constant: fc = 1/(2*pi*R*C)",
            "Verify input source is AC type for frequency analysis",
            "For active filters, ensure op-amp is properly configured",
        ],
        "amplifier": [
            "Verify gain formula: Av = -Rf/Rin (inverting) or 1+Rf/Rg (non-inverting)",
            "Check op-amp has feedback path",
            "Ensure input connects to correct op-amp terminal",
        ],
        "oscillator": [
            "Verify frequency-determining components (R, C or L, C)",
            "Check loop gain >= 1 for oscillation",
            "Include amplitude stabilization if needed",
            "Oscillators don't need Vin - they generate their own signal",
        ],
        "clipper": [
            "Diode orientation determines clipping direction",
            "Include current limiting resistor",
            "Add Vin with SIN source for transient analysis",
        ],
        "clamper": [
            "Series capacitor C1 goes from vin to vout (the output node)",
            "Load resistor RL must connect from vout to ground (0) — never between two internal nodes",
            "Positive clamper diode: D1 0 vout DGENERIC (anode=0, cathode=vout)",
            "Negative clamper diode: D1 vout 0 DGENERIC (anode=vout, cathode=0)",
            "Time constant C1*RL >> signal period (use C1=10u, RL=100k for 1kHz input)",
            "Add Vin with SIN source for transient analysis",
        ],
        "general": [
            "The primary output node MUST be named 'vout' — the evaluator probes v(vout); using 'out', 'vo', 'vload', etc. causes silent zero output",
            "All nodes must connect to at least two components — no floating nodes",
            "Node 0 is the ground reference — connect circuit ground to node 0",
            "Include an input voltage source (e.g., Vin vin 0 SIN(0 5 1k)) if your circuit needs one",
            "Component names must start with the correct letter: R=resistor, C=capacitor, D=diode, Q=BJT, M=MOSFET, V=voltage source",
            "End the netlist with .end",
            "Use .model statements for any diodes (DGENERIC D) or transistors",
        ],
    }

    def __init__(self):
        """Initialize the failure reflector"""
        pass

    def diagnose(
        self,
        eval_result: EvaluationResult,
        sim_result: Optional[SimulationResult] = None,
        circuit_type: str = "",
    ) -> Diagnosis:
        """
        Analyze evaluation result and generate diagnosis.
        
        Args:
            eval_result: Result from RewardEvaluator
            sim_result: Optional simulation result with error details
            circuit_type: Type of circuit being analyzed
            
        Returns:
            Diagnosis with failure analysis and suggested fixes
        """
        # Determine failure mode
        failure_mode = eval_result.details.get(
            "failure_mode",
            self._infer_failure_mode(eval_result)
        )
        
        # Get failure mode info
        mode_info = self.FAILURE_MODES.get(failure_mode, self.FAILURE_MODES["unknown"])
        
        # Extract error message
        error_msg = ""
        if sim_result and sim_result.error_message:
            error_msg = sim_result.error_message
        elif eval_result.details.get("error"):
            error_msg = eval_result.details["error"]
        
        # Get suggested fixes
        fixes = mode_info["fixes"].copy()
        
        # Add circuit-specific fixes
        if circuit_type and circuit_type in self.CIRCUIT_CHECKS:
            fixes.extend(self.CIRCUIT_CHECKS[circuit_type])
        
        # Add error-specific fixes
        if error_msg:
            error_fixes = self._error_to_fixes(error_msg)
            fixes.extend(error_fixes)
        
        # Remove duplicates while preserving order
        fixes = list(dict.fromkeys(fixes))
        
        # Calculate confidence
        confidence = self._calculate_confidence(eval_result, sim_result, failure_mode)
        
        return Diagnosis(
            reward=eval_result.reward,
            reason=eval_result.reason,
            failure_mode=failure_mode,
            error_message=error_msg,
            suggested_fixes=fixes,
            confidence=confidence,
            circuit_type=circuit_type,
        )

    def _infer_failure_mode(self, eval_result: EvaluationResult) -> str:
        """Infer failure mode from evaluation result"""
        if not eval_result.has_netlist_block:
            return "missing_code_block"
        elif not eval_result.netlist_valid:
            return "empty_netlist"
        elif not eval_result.compiles:
            return "syntax_error"
        elif not eval_result.simulates:
            return "simulation_error"
        elif eval_result.reward < 0.8:
            return "flat_output"
        return "unknown"

    def _error_to_fixes(self, error_msg: str) -> List[str]:
        """Convert error message to specific fixes"""
        fixes = []
        error_lower = error_msg.lower()
        
        if "undefined node" in error_lower:
            fixes.append(f"Define all nodes - '{error_msg}' suggests a node is referenced but not connected")
        
        if "singular matrix" in error_lower:
            fixes.append("Add a high-value resistor (1G) to ground to prevent floating nodes")
        
        if "time step too small" in error_lower:
            fixes.append("Increase simulation timestep or add initial conditions (.ic)")
        
        if "vin" in error_lower or "source" in error_lower:
            fixes.append("Ensure Vin is properly defined and connected")
        
        if "ground" in error_lower or "node 0" in error_lower:
            fixes.append("Connect circuit ground to node 0")
        
        if "opamp" in error_lower or "op-amp" in error_lower or "op amp" in error_lower:
            fixes.append("Check op-amp connections: +input, -input, output, and power")
        
        return fixes

    def _calculate_confidence(
        self,
        eval_result: EvaluationResult,
        sim_result: Optional[SimulationResult],
        failure_mode: str,
    ) -> float:
        """Calculate confidence in diagnosis (0.0 to 1.0)"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for clear failure modes
        if failure_mode in ["missing_code_block", "empty_netlist"]:
            confidence += 0.3  # Very clear
        
        # Higher confidence with simulation error details
        if sim_result and sim_result.error_message:
            confidence += 0.2
        
        # Higher confidence for low rewards
        if eval_result.reward < 0.1:
            confidence += 0.1
        
        return min(1.0, confidence)

    def generate_feedback(
        self,
        diagnosis: Diagnosis,
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Generate feedback dictionary for the actor.
        
        Args:
            diagnosis: Diagnosis from analyze()
            iteration: Current iteration number
            
        Returns:
            Feedback dictionary for actor consumption
        """
        # Create prioritized fix list
        top_fixes = diagnosis.suggested_fixes[:3]  # Top 3 fixes
        
        # Generate feedback message
        feedback = {
            "iteration": iteration,
            "reward": diagnosis.reward,
            "reason": diagnosis.reason,
            "failure_mode": diagnosis.failure_mode,
            "error_message": diagnosis.error_message,
            "top_fixes": top_fixes,
            "all_fixes": diagnosis.suggested_fixes,
            "confidence": diagnosis.confidence,
            "circuit_type": diagnosis.circuit_type,
        }
        
        return feedback

    def get_iteration_guidance(
        self,
        diagnosis: Diagnosis,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """
        Generate natural language guidance for the next iteration.
        
        Args:
            diagnosis: Current diagnosis
            iteration: Current iteration (1-indexed)
            max_iterations: Maximum allowed iterations
            
        Returns:
            Guidance string for the actor
        """
        remaining = max_iterations - iteration
        
        # Build guidance message
        parts = []
        
        # Current status
        parts.append(f"Iteration {iteration}/{max_iterations}")
        parts.append(f"Current score: {diagnosis.reward:.2f}")
        parts.append(f"Issue: {diagnosis.reason}")
        
        # Add error if available
        if diagnosis.error_message:
            parts.append(f"Error: {diagnosis.error_message}")
        
        # Add top fixes
        if diagnosis.suggested_fixes:
            parts.append("\nSuggested fixes:")
            for i, fix in enumerate(diagnosis.suggested_fixes[:3], 1):
                parts.append(f"  {i}. {fix}")
        
        # Add urgency if running out of iterations
        if remaining <= 1:
            parts.append("\n⚠️ This is the last iteration - make it count!")
        elif remaining <= 2:
            parts.append(f"\n⚡ {remaining} iterations remaining - focus on critical fixes")
        
        return "\n".join(parts)

    def reflect_on_history(
        self,
        diagnoses: List[Diagnosis],
    ) -> Dict[str, Any]:
        """
        Analyze multiple iterations to find patterns.
        
        Args:
            diagnoses: List of diagnoses from previous iterations
            
        Returns:
            Meta-analysis with patterns and recommendations
        """
        if not diagnoses:
            return {"error": "No diagnoses to analyze"}
        
        # Check for repeated failure modes
        failure_counts = {}
        for d in diagnoses:
            mode = d.failure_mode
            failure_counts[mode] = failure_counts.get(mode, 0) + 1
        
        # Find most common failure
        most_common = max(failure_counts, key=failure_counts.get) if failure_counts else "unknown"
        
        # Check for improvement
        rewards = [d.reward for d in diagnoses]
        improving = rewards[-1] > rewards[0] if len(rewards) > 1 else False
        
        # Generate recommendations
        recommendations = []
        
        if failure_counts.get("missing_code_block", 0) > 1:
            recommendations.append("Model consistently fails to output code blocks - try a different prompt format")
        
        if failure_counts.get("syntax_error", 0) > 2:
            recommendations.append("Repeated syntax errors - consider using a simpler circuit topology")
        
        if failure_counts.get("simulation_error", 0) > 2:
            recommendations.append("Persistent simulation failures - ensure input source and ground are included")
        
        if not improving and len(rewards) > 2:
            recommendations.append("No improvement trend - consider restarting with a different approach")
        
        return {
            "iterations": len(diagnoses),
            "failure_mode_counts": failure_counts,
            "most_common_failure": most_common,
            "reward_trend": rewards,
            "improving": improving,
            "final_reward": rewards[-1] if rewards else 0,
            "recommendations": recommendations,
        }
