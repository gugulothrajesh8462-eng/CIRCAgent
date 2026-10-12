#!/usr/bin/env python3
"""
Circuit Planner Module

Analyzes circuit specifications from natural language prompts and creates
a structured plan for netlist generation including topology selection
and component value planning.
"""

import re
import math
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class CircuitPlan:
    """Represents a structured plan for circuit generation"""
    circuit_type: str
    topology: str
    specifications: Dict[str, Any]
    component_plan: List[Dict[str, Any]] = field(default_factory=list)
    design_notes: str = ""
    potential_issues: List[str] = field(default_factory=list)
    iteration: int = 0
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)
    original_prompt: str = ""  # Preserved verbatim for general/unknown circuit types

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary for serialization"""
        return {
            "circuit_type": self.circuit_type,
            "topology": self.topology,
            "specifications": self.specifications,
            "component_plan": self.component_plan,
            "design_notes": self.design_notes,
            "potential_issues": self.potential_issues,
            "iteration": self.iteration,
        }


class CircuitPlanner:
    """
    Analyzes circuit requirements and generates structured design plans.
    
    The planner performs:
    1. Specification extraction from natural language
    2. Topology selection based on circuit type and specs
    3. Component value planning
    4. Design note generation for the actor
    """

    # Topology recommendations for each circuit type
    TOPOLOGIES = {
        "filter": {
            "low": "RC low-pass filter with optional op-amp buffer for gain",
            "high": "RC high-pass filter with optional op-amp buffer for gain",
            "band": "RC band-pass filter (cascade of high-pass and low-pass) or active filter",
        },
        "amplifier": {
            "inverting": "Inverting op-amp configuration with feedback resistor network",
            "non-inverting": "Non-inverting op-amp configuration with voltage divider feedback",
            "differential": "Differential amplifier with matched resistor pairs",
            "instrumentation": "Three op-amp instrumentation amplifier for high input impedance",
        },
        "oscillator": {
            "Wien bridge": "Wien bridge oscillator with op-amp and RC feedback network",
            "phase shift": "RC phase shift oscillator with three RC stages",
            "RC": "RC relaxation oscillator or phase shift topology",
            "LC": "LC tank circuit with active device for sustained oscillation",
            "Colpitts": "Colpitts oscillator with capacitive voltage divider feedback",
            "Hartley": "Hartley oscillator with tapped inductor feedback",
        },
        "clipper": {
            "positive": "Diode clipper with diode oriented to clip positive peaks",
            "negative": "Diode clipper with diode oriented to clip negative peaks",
            "both": "Dual diode clipper with back-to-back diodes",
        },
        "clamper": {
            "positive": "Positive clamper with diode and capacitor to shift waveform up",
            "negative": "Negative clamper with diode and capacitor to shift waveform down",
        },
        "general": {
            "default": "Use the most appropriate topology for the described circuit",
        },
    }

    # Design guidelines for each circuit type
    DESIGN_GUIDELINES = {
        "filter": [
            "Use fc = 1/(2πRC) for cutoff frequency calculation",
            "Choose standard resistor values (1k-100k range preferred)",
            "For gain, add non-inverting op-amp stage with Av = 1 + Rf/Rg",
            "Ensure op-amp bandwidth exceeds cutoff frequency",
        ],
        "amplifier": [
            "Gain in dB: Av_dB = 20*log10(Vout/Vin)",
            "For inverting: Av = -Rf/Rin",
            "For non-inverting: Av = 1 + Rf/Rg",
            "Include DC supply connections for op-amp",
        ],
        "oscillator": [
            "Frequency determined by RC or LC time constants",
            "Wien bridge: f = 1/(2πRC) for equal R and C in bridge",
            "Include amplitude stabilization (diodes or JFET)",
            "Ensure loop gain >= 1 for sustained oscillation",
        ],
        "clipper": [
            "Clipping level ≈ diode forward voltage (0.7V for Si)",
            "Add DC bias to adjust clipping threshold",
            "Include current limiting resistor",
        ],
        "clamper": [
            "Output DC level shifted by capacitor charging",
            "Time constant RC >> period of input signal",
            "Diode determines clamping direction",
        ],
        "general": [
            "Include all necessary SPICE components with correct syntax",
            "Connect every node — no floating nodes allowed",
            "Use node 0 as the ground reference for all circuits",
            "Include appropriate input voltage/current sources",
            "End the netlist with .end",
            "Use standard value suffixes: k=1e3, m=1e-3, u=1e-6, n=1e-9, p=1e-12",
        ],
    }

    def __init__(self):
        """Initialize the circuit planner"""
        pass

    def analyze(self, prompt: str) -> CircuitPlan:
        """
        Analyze a natural language prompt and create a circuit plan.
        
        Args:
            prompt: Natural language circuit specification
            
        Returns:
            CircuitPlan object with structured design information
        """
        # Extract circuit type
        circuit_type = self._extract_circuit_type(prompt)
        
        # Extract specifications
        specs = self._extract_specifications(prompt, circuit_type)
        
        # Select topology
        topology = self._select_topology(circuit_type, specs)
        
        # Generate component plan
        component_plan = self._plan_components(circuit_type, specs)
        
        # Generate design notes
        design_notes = self._generate_design_notes(circuit_type, specs)
        
        # Identify potential issues
        potential_issues = self._identify_potential_issues(circuit_type, specs)
        
        return CircuitPlan(
            circuit_type=circuit_type,
            topology=topology,
            specifications=specs,
            component_plan=component_plan,
            design_notes=design_notes,
            potential_issues=potential_issues,
            iteration=0,
            original_prompt=prompt,
        )

    def refine(
        self,
        previous_plan: CircuitPlan,
        diagnosis: Dict[str, Any],
        iteration: int
    ) -> CircuitPlan:
        """
        Refine a plan based on feedback from a failed attempt.
        
        Args:
            previous_plan: The plan that led to a failed attempt
            diagnosis: Diagnosis of what went wrong
            iteration: Current iteration number
            
        Returns:
            Refined CircuitPlan
        """
        # Store previous attempt
        previous_plan.previous_attempts.append({
            "iteration": iteration,
            "reward": diagnosis.get("reward", 0),
            "reason": diagnosis.get("reason", "unknown"),
            "error": diagnosis.get("error", ""),
        })
        
        # Update iteration counter
        previous_plan.iteration = iteration
        
        # Add iteration-specific notes
        iteration_notes = self._generate_iteration_notes(diagnosis, previous_plan)
        previous_plan.design_notes += f"\n\nIteration {iteration} notes: {iteration_notes}"
        
        # Update potential issues based on diagnosis
        new_issues = self._diagnose_to_issues(diagnosis)
        previous_plan.potential_issues.extend(new_issues)
        
        # Remove duplicates
        previous_plan.potential_issues = list(set(previous_plan.potential_issues))
        
        return previous_plan

    def _extract_circuit_type(self, prompt: str) -> str:
        """Extract circuit type from prompt"""
        prompt_lower = prompt.lower()
        
        # Check for specific types first (more specific matches)
        if "clamper" in prompt_lower:
            return "clamper"
        elif "clipper" in prompt_lower:
            return "clipper"
        elif "oscillator" in prompt_lower or "osc" in prompt_lower:
            return "oscillator"
        elif "amplifier" in prompt_lower or "amp " in prompt_lower or "amp." in prompt_lower:
            return "amplifier"
        elif "low-pass" in prompt_lower or "lowpass" in prompt_lower:
            return "filter"
        elif "high-pass" in prompt_lower or "highpass" in prompt_lower:
            return "filter"
        elif "band-pass" in prompt_lower or "bandpass" in prompt_lower:
            return "filter"
        
        return "general"  # Unknown circuit type — pass prompt through as-is

    def _extract_specifications(self, prompt: str, circuit_type: str) -> Dict[str, Any]:
        """Extract numerical specifications from prompt"""
        specs = {}
        prompt_lower = prompt.lower()
        
        # Frequency extraction (cutoff, oscillation, etc.)
        freq_match = re.search(r'([\d\.]+)\s*(k|m|g)?hz', prompt_lower)
        if freq_match:
            val = float(freq_match.group(1))
            unit = freq_match.group(2)
            if unit == 'k':
                val *= 1e3
            elif unit == 'm':
                val *= 1e6
            elif unit == 'g':
                val *= 1e9
            
            if circuit_type == "filter":
                specs["cutoff_freq"] = val
            elif circuit_type == "oscillator":
                specs["freq_hz"] = val
            elif circuit_type == "amplifier":
                specs["f_high"] = val
        
        # Gain extraction
        gain_match = re.search(r'([\d\.]+)\s*db', prompt_lower)
        if gain_match:
            specs["gain_db"] = float(gain_match.group(1))
        
        # Also check for "gain of X" pattern
        gain_linear_match = re.search(r'gain (?:of )?(\d+\.?\d*)', prompt_lower)
        if gain_linear_match and "gain_db" not in specs:
            linear_gain = float(gain_linear_match.group(1))
            specs["gain_db"] = 20 * math.log10(linear_gain) if linear_gain > 0 else 0
        
        # Bandwidth extraction
        bw_match = re.search(r'bandwidth (?:of )?([\d\.]+)\s*(k|m|g)?hz', prompt_lower)
        if bw_match:
            val = float(bw_match.group(1))
            unit = bw_match.group(2)
            if unit == 'k':
                val *= 1e3
            elif unit == 'm':
                val *= 1e6
            specs["bandwidth"] = val
        
        # Amplitude extraction (for oscillators)
        amp_match = re.search(r'amplitude (?:of )?(\d+\.?\d*)\s*v', prompt_lower)
        if amp_match:
            specs["v_peak"] = float(amp_match.group(1))
        
        # Check for "V peak" pattern
        vpeak_match = re.search(r'(\d+\.?\d*)\s*v\s*peak', prompt_lower)
        if vpeak_match:
            specs["v_peak"] = float(vpeak_match.group(1))
        
        # Clipping/clamping level
        clip_match = re.search(r'(?:clip|clamp)(?:ing)? (?:level )?(?:of )?(\d+\.?\d*)\s*v', prompt_lower)
        if clip_match:
            specs["clip_voltage"] = float(clip_match.group(1))
            specs["clamp_level"] = float(clip_match.group(1))
        
        # Input signal amplitude (for clippers/clampers)
        vin_match = re.search(r'v_in\s*[=:]?\s*(\d+\.?\d*)\s*v', prompt_lower)
        if vin_match:
            specs["v_in_peak"] = float(vin_match.group(1))
        
        # Direction for clippers/clampers
        if "positive" in prompt_lower:
            specs["direction"] = "positive"
        elif "negative" in prompt_lower:
            specs["direction"] = "negative"
        elif "both" in prompt_lower or "dual" in prompt_lower:
            specs["direction"] = "both"
        
        # Amplifier type
        if "inverting" in prompt_lower:
            specs["amplifier_type"] = "inverting"
        elif "non-inverting" in prompt_lower or "noninverting" in prompt_lower:
            specs["amplifier_type"] = "non-inverting"
        elif "differential" in prompt_lower:
            specs["amplifier_type"] = "differential"
        elif "instrumentation" in prompt_lower:
            specs["amplifier_type"] = "instrumentation"
        
        # Filter type
        if "low-pass" in prompt_lower or "lowpass" in prompt_lower:
            specs["filter_type"] = "low"
        elif "high-pass" in prompt_lower or "highpass" in prompt_lower:
            specs["filter_type"] = "high"
        elif "band-pass" in prompt_lower or "bandpass" in prompt_lower:
            specs["filter_type"] = "band"
        
        # Oscillator type
        for osc_type in ["wien", "phase shift", "rc", "lc", "colpitts", "hartley"]:
            if osc_type in prompt_lower:
                specs["oscillator_type"] = osc_type
                break
        
        # Set defaults
        self._set_defaults(specs, circuit_type)
        
        return specs

    def _set_defaults(self, specs: Dict[str, Any], circuit_type: str) -> None:
        """Set default values for missing specifications"""
        if circuit_type == "filter":
            specs.setdefault("cutoff_freq", 1000)
            specs.setdefault("passband_gain_db", 0)
            specs.setdefault("filter_type", "low")
        elif circuit_type == "amplifier":
            specs.setdefault("gain_db", 20)
            specs.setdefault("amplifier_type", "non-inverting")
        elif circuit_type == "oscillator":
            specs.setdefault("freq_hz", 1000)
            specs.setdefault("v_peak", 5)
            specs.setdefault("oscillator_type", "wien bridge")
        elif circuit_type == "clipper":
            specs.setdefault("clip_voltage", 5)
            specs.setdefault("direction", "positive")
        elif circuit_type == "clamper":
            specs.setdefault("v_in_peak", 5)
            specs.setdefault("clamp_level", 0)
            specs.setdefault("direction", "positive")
        elif circuit_type == "general":
            specs.setdefault("input_freq", 1000)

    def _select_topology(self, circuit_type: str, specs: Dict[str, Any]) -> str:
        """Select appropriate topology based on circuit type and specs"""
        if circuit_type == "filter":
            filter_type = specs.get("filter_type", "low")
            return self.TOPOLOGIES["filter"].get(filter_type, "RC filter")
        elif circuit_type == "amplifier":
            amp_type = specs.get("amplifier_type", "non-inverting")
            return self.TOPOLOGIES["amplifier"].get(amp_type, "Op-amp amplifier")
        elif circuit_type == "oscillator":
            osc_type = specs.get("oscillator_type", "wien bridge")
            return self.TOPOLOGIES["oscillator"].get(osc_type, "RC oscillator")
        elif circuit_type == "clipper":
            direction = specs.get("direction", "positive")
            return self.TOPOLOGIES["clipper"].get(direction, "Diode clipper")
        elif circuit_type == "clamper":
            direction = specs.get("direction", "positive")
            return self.TOPOLOGIES["clamper"].get(direction, "Diode clamper")
        elif circuit_type == "general":
            return self.TOPOLOGIES["general"]["default"]

        return "Standard circuit topology"

    def _plan_components(
        self,
        circuit_type: str,
        specs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate a component value plan"""
        components = []
        
        if circuit_type == "filter":
            # Calculate R and C for filter
            fc = specs.get("cutoff_freq", 1000)
            # Choose C first, calculate R
            # C = 1/(2*pi*fc*R), choose R=10k for audio range
            r_val = 10e3
            c_val = 1 / (2 * 3.14159 * fc * r_val)
            
            components.append({"type": "R", "value": r_val, "role": "filter resistor"})
            components.append({"type": "C", "value": c_val, "role": "filter capacitor"})
            
            if specs.get("gain_db", 0) > 0:
                gain = 10 ** (specs["gain_db"] / 20)
                rf = 10e3 * (gain - 1)
                components.append({"type": "Rf", "value": rf, "role": "feedback resistor"})
                components.append({"type": "Rg", "value": 10e3, "role": "gain setting resistor"})
        
        elif circuit_type == "amplifier":
            gain_db = specs.get("gain_db", 20)
            gain = 10 ** (gain_db / 20)
            
            amp_type = specs.get("amplifier_type", "non-inverting")
            if amp_type == "inverting":
                rin = 10e3
                rf = rin * gain
                components.append({"type": "Rin", "value": rin, "role": "input resistor"})
                components.append({"type": "Rf", "value": rf, "role": "feedback resistor"})
            else:
                rg = 10e3
                rf = rg * (gain - 1)
                components.append({"type": "Rg", "value": rg, "role": "ground resistor"})
                components.append({"type": "Rf", "value": rf, "role": "feedback resistor"})
        
        elif circuit_type == "oscillator":
            freq = specs.get("freq_hz", 1000)
            # Wien bridge: f = 1/(2*pi*R*C)
            c_val = 10e-9  # 10nF
            r_val = 1 / (2 * 3.14159 * freq * c_val)
            
            components.append({"type": "R", "value": r_val, "role": "frequency setting resistor"})
            components.append({"type": "C", "value": c_val, "role": "frequency setting capacitor"})
        
        return components

    def _generate_design_notes(
        self,
        circuit_type: str,
        specs: Dict[str, Any]
    ) -> str:
        """Generate design notes and guidelines for the actor"""
        notes = []
        
        # Add general guidelines for circuit type
        if circuit_type in self.DESIGN_GUIDELINES:
            notes.extend(self.DESIGN_GUIDELINES[circuit_type])
        
        # Add specific notes based on specs
        if specs.get("gain_db", 0) > 40:
            notes.append("High gain detected - consider using multiple stages")
        
        if specs.get("freq_hz", 0) > 100e3 or specs.get("cutoff_freq", 0) > 100e3:
            notes.append("High frequency - ensure op-amp has sufficient GBW")
        
        return "; ".join(notes)

    def _identify_potential_issues(
        self,
        circuit_type: str,
        specs: Dict[str, Any]
    ) -> List[str]:
        """Identify potential design issues to watch for"""
        issues = []
        
        if circuit_type == "filter":
            if specs.get("gain_db", 0) > 0:
                issues.append("Ensure op-amp is properly biased")
            if specs.get("cutoff_freq", 0) > 100e3:
                issues.append("High cutoff frequency may require high-speed op-amp")
        
        elif circuit_type == "oscillator":
            issues.append("Ensure loop gain >= 1 for oscillation startup")
            issues.append("Include amplitude stabilization mechanism")

        elif circuit_type == "amplifier":
            if specs.get("gain_db", 0) > 60:
                issues.append("Very high gain - watch for stability and noise")

        elif circuit_type == "general":
            issues.append("Ensure all nodes are connected to ground reference (node 0)")
            issues.append("Include a complete .control block or the simulator will add one")

        return issues

    def _generate_iteration_notes(
        self,
        diagnosis: Dict[str, Any],
        plan: CircuitPlan
    ) -> str:
        """Generate notes for iteration based on diagnosis"""
        reason = diagnosis.get("reason", "")
        reward = diagnosis.get("reward", 0)
        
        if "no netlist" in reason.lower():
            return "Must include ```spice code block with netlist"
        elif "invalid" in reason.lower() or "compiles" in reason.lower():
            return "Check SPICE syntax - component names, node connections, and .end statement"
        elif "simulation failed" in reason.lower():
            return "Check for missing sources, floating nodes, or unrealistic component values"
        elif "flat output" in reason.lower():
            return "Check biasing, ensure signal path is complete, verify component values"
        
        return f"Previous reward: {reward}. Review and improve the design."

    def _diagnose_to_issues(self, diagnosis: Dict[str, Any]) -> List[str]:
        """Convert diagnosis to actionable issues"""
        issues = []
        reason = diagnosis.get("reason", "").lower()
        
        if "no netlist" in reason:
            issues.append("Must output ```spice code block")
        elif "invalid" in reason or "compile" in reason:
            issues.append("Fix SPICE syntax errors")
        elif "simulation" in reason:
            issues.append("Ensure circuit has proper power and input sources")
        elif "flat" in reason:
            issues.append("Check component values and biasing")
        
        return issues
