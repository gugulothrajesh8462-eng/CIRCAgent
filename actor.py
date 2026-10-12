#!/usr/bin/env python3
"""
Netlist Actor Module

Generates SPICE netlists using LLM calls via OpenRouter API.
The actor takes a circuit plan and produces properly formatted
SPICE netlist code.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from openai import OpenAI

from .planner import CircuitPlan


# Default system prompt for SPICE generation
DEFAULT_SYSTEM_PROMPT = """You are an expert Senior Analog IC Designer and SPICE Netlist Engineer.
Your goal is to generate precise, functional, and minimal SPICE netlists based on user specifications.

## Guidelines:
1. **Output Format**:
   - Provide ONLY the SPICE netlist code inside a Markdown code block: ```spice ... ```.
   - Do NOT include conversational filler, explanations, or tutorials unless explicitly asked.
   - Ensure the netlist is complete and ready for simulation.

2. **Netlist syntax**:
   - Use standard SPICE syntax (ngspice compatible).
   - Component naming conventions:
     - Resistors: R1, R2...
     - Capacitors: C1, C2...
     - Inductors: L1, L2...
     - Voltage Sources: V1, V2... (VCC, VEE for rails)
     - Transistors: Q1 (BJT), M1 (MOSFET)
     - OpAmps: X1 (subcircuit calls or behavioral models)
   - Node naming:
     - Use '0' for ground.
     - Use descriptive node names where useful (e.g., 'no', 'vin', 'vout', 'vcc').

3. **Behavioral Modeling**:
   - If a specific OpAmp model is not provided, use a Voltage Controlled Voltage Source (VCVS or 'E' source) with high gain for ideal OpAmps.
   - Example Ideal OpAmp: `E1 vout 0 non_inv inv 1e6`

4. **Simulation Control**:
   - Do NOT include `.control` ... `.endc` blocks.
   - Do NOT include `.tran`, `.ac`, or `.dc` simulation directives.
   - Focus purely on the circuit topology, component values, and input sources.
   - The simulation commands will be added automatically by the evaluation system.

5. **Expertise**:
   - Ensure component values are realistic (e.g., don't use 1 Farad capacitors for RF circuits).
   - Ensure proper biasing for transistors.
   - Should be robust and functional.

## Example Output:
User: "Design a low-pass RC filter with cutoff 1kHz."
Assistant:
```spice
* Low Pass RC Filter fc=1kHz
R1 vin vout 1k
C1 vout 0 159n
* AC Source for testing
Vin vin 0 AC 1
```
"""

# Enhanced system prompt for iterative refinement
ITERATIVE_SYSTEM_PROMPT = """You are an expert Senior Analog IC Designer and SPICE Netlist Engineer.
Your goal is to generate precise, functional, and minimal SPICE netlists based on user specifications.

You are working in an ITERATIVE REFINEMENT loop where:
1. You generate a netlist
2. It gets simulated
3. You receive feedback on what went wrong
4. You revise and improve the netlist

## Critical Guidelines:
1. **ALWAYS output a complete, valid SPICE netlist** in a ```spice code block
2. **Include ALL necessary components**: resistors, capacitors, sources, op-amps
3. **Use proper SPICE syntax** - component names must start with R, C, L, V, I, E, etc.
4. **Always include an input source** (Vin) unless it's an oscillator
5. **Always end with .end**
6. **Use realistic component values** - resistors in ohms (not 1), capacitors in farads (use u, n, p suffixes)

## Common Mistakes to Avoid:
- Missing input voltage source (Vin)
- Floating nodes (every node must connect to something)
- Missing ground connection (node 0)
- Using .control blocks (these will be added by the simulator)
- Using unrealistic values (1 ohm resistors, 1 farad capacitors)

## When Receiving Feedback:
- Carefully read the error message
- Identify the specific issue (syntax, missing component, simulation failure)
- Fix ONLY that issue while preserving working parts
- Output the COMPLETE revised netlist

Output ONLY the ```spice code block with your netlist."""


class NetlistActor:
    """
    Generates SPICE netlists using LLM API calls.
    
    The actor:
    1. Takes a CircuitPlan from the planner
    2. Constructs appropriate prompts
    3. Calls OpenRouter API
    4. Returns the generated netlist
    """

    def __init__(
        self,
        model: str = "qwen/qwen3-4b:free",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        use_iterative_prompt: bool = True,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the netlist actor.

        Args:
            model: Model identifier (OpenRouter or local)
            api_key: API key (or set OPENROUTER_API_KEY env var)
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
            use_iterative_prompt: Use enhanced prompt for iterative refinement
            base_url: Custom API base URL (for Ollama or other providers)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_iterative_prompt = use_iterative_prompt
        self.base_url = base_url

        # Get API key - try multiple sources
        self.api_key = (
            api_key or
            os.getenv("OPENROUTER_API_KEY") or
            os.getenv("OPENROUTER_KEY") or
            self._load_from_env_file()
        )
        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENROUTER_API_KEY env var "
                "or add to agent/.env file. For Ollama, use 'ollama' as key."
            )

        # Initialize OpenAI client
        # Use custom base URL if provided (for Ollama or other providers)
        if self.base_url:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            # Default to OpenRouter
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

    def _load_from_env_file(self) -> Optional[str]:
        """Load API key from agent/.env file"""
        env_path = Path(__file__).parent / ".env"
        if not env_path.exists():
            return None
        
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENROUTER_KEY') and '=' in line:
                        # Parse: OPENROUTER_KEY = "value"
                        value = line.split('=', 1)[1].strip()
                        # Remove quotes if present
                        value = value.strip('"\'')
                        return value
        except Exception:
            pass
        return None

    def _load_system_prompt(self) -> str:
        """Load system prompt from file or use default"""
        system_prompt_path = "SYSTEMPROMPT.md"
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path, 'r') as f:
                return f.read()
        return DEFAULT_SYSTEM_PROMPT

    def generate(
        self,
        plan: CircuitPlan,
        previous_response: Optional[str] = None,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a SPICE netlist based on a circuit plan.
        
        Args:
            plan: CircuitPlan from the planner
            previous_response: Previous netlist attempt (for refinement)
            feedback: Feedback dictionary with reward, reason, error
            
        Returns:
            Generated netlist string (raw response from LLM)
        """
        # Construct user prompt
        user_prompt = self._construct_user_prompt(plan, previous_response, feedback)
        
        # Select system prompt
        system_prompt = (
            ITERATIVE_SYSTEM_PROMPT if feedback is not None
            else self.system_prompt
        )
        
        # Call LLM API
        response = self._call_llm(system_prompt, user_prompt)
        
        return response

    def _construct_user_prompt(
        self,
        plan: CircuitPlan,
        previous_response: Optional[str],
        feedback: Optional[Dict[str, Any]],
    ) -> str:
        """Construct the user prompt for netlist generation"""
        
        # Base prompt from plan
        base_prompt = self._plan_to_prompt(plan)
        
        # If this is a refinement iteration, add feedback
        if feedback is not None and previous_response is not None:
            refinement_prompt = self._construct_refinement_prompt(
                previous_response, feedback, plan
            )
            return f"{base_prompt}\n\n{refinement_prompt}"
        
        return base_prompt

    def _plan_to_prompt(self, plan: CircuitPlan) -> str:
        """Convert a CircuitPlan to a natural language prompt"""
        specs = plan.specifications
        circuit_type = plan.circuit_type
        
        # Build specification string
        spec_parts = []
        
        if circuit_type == "filter":
            filter_type = specs.get("filter_type", "low")
            spec_parts.append(f"{filter_type}-pass filter")
            if "cutoff_freq" in specs:
                fc = specs["cutoff_freq"]
                if fc >= 1e6:
                    spec_parts.append(f"cutoff {fc/1e6:.2g} MHz")
                elif fc >= 1e3:
                    spec_parts.append(f"cutoff {fc/1e3:.2g} kHz")
                else:
                    spec_parts.append(f"cutoff {fc:.2g} Hz")
            if specs.get("gain_db", 0) > 0:
                spec_parts.append(f"gain {specs['gain_db']} dB")
                
        elif circuit_type == "amplifier":
            amp_type = specs.get("amplifier_type", "non-inverting")
            spec_parts.append(f"{amp_type} op-amp amplifier (use ideal VCVS: E1 vout 0 vplus vminus 1e6)")
            if "gain_db" in specs:
                spec_parts.append(f"gain {specs['gain_db']} dB")
            if "f_high" in specs:
                spec_parts.append(f"bandwidth {specs['f_high']/1e3:.2g} kHz")

        elif circuit_type == "oscillator":
            osc_type = specs.get("oscillator_type", "wien")
            spec_parts.append(f"{osc_type} oscillator (use ideal op-amp VCVS: E1 vout 0 vplus vminus 3, with RC feedback network, NO input source Vin)")
            if "freq_hz" in specs:
                freq = specs["freq_hz"]
                if freq >= 1e6:
                    spec_parts.append(f"frequency {freq/1e6:.2g} MHz")
                elif freq >= 1e3:
                    spec_parts.append(f"frequency {freq/1e3:.2g} kHz")
                else:
                    spec_parts.append(f"frequency {freq:.2g} Hz")
            if "v_peak" in specs:
                spec_parts.append(f"amplitude {specs['v_peak']}V peak")

        elif circuit_type == "clipper":
            direction = specs.get("direction", "positive")
            spec_parts.append(f"{direction} diode clipper (use .model DGENERIC D for the diode model)")
            if "clip_voltage" in specs:
                spec_parts.append(f"clipping at {specs['clip_voltage']}V")

        elif circuit_type == "clamper":
            direction = specs.get("direction", "positive")
            if direction == "negative":
                diode_hint = "D1 vout 0 DGENERIC (anode=vout, cathode=gnd)"
            else:
                diode_hint = "D1 0 vout DGENERIC (anode=gnd, cathode=vout)"
            spec_parts.append(
                f"{direction} diode clamper — use topology: C1 vin vout 10u, "
                f"{diode_hint}, RL vout 0 100k, .model DGENERIC D "
                f"(load resistor MUST connect vout to ground, not to a floating node)"
            )
            if "v_in_peak" in specs:
                spec_parts.append(f"input {specs['v_in_peak']}V peak SIN source")
            if "clamp_level" in specs:
                spec_parts.append(f"clamp to {specs['clamp_level']}V")

        # For unknown/general circuit types, pass the original prompt directly
        # rather than trying to reconstruct it from parsed specs
        if circuit_type == "general":
            original = plan.original_prompt or " ".join(spec_parts) or circuit_type
            return (
                f"{original}\n\n"
                "Output a complete, functional SPICE netlist in a ```spice code block. "
                "Name the primary output node 'vout'. "
                "Connect all nodes to ground (node 0). "
                "End with .end."
            )

        spec_str = " with ".join(spec_parts) if spec_parts else circuit_type

        return f"Design a {spec_str}."

    def _construct_refinement_prompt(
        self,
        previous_response: str,
        feedback: Dict[str, Any],
        plan: CircuitPlan,
    ) -> str:
        """Construct a prompt for iterative refinement"""
        reward = feedback.get("reward", 0)
        reason = feedback.get("reason", "unknown")
        error = feedback.get("error", "")
        
        # Build feedback message
        feedback_msg = f"Your previous attempt received a score of {reward:.2f}/1.0"
        feedback_msg += f"\n\nIssue: {reason}"
        
        if error:
            feedback_msg += f"\nSimulation error: {error}"
        
        # Add specific guidance based on reason
        guidance = self._get_refinement_guidance(reason, plan)
        feedback_msg += f"\n\nPlease fix this issue and provide a corrected netlist."
        feedback_msg += f"\n\nGuidance: {guidance}"
        
        return feedback_msg

    def _get_refinement_guidance(
        self,
        reason: str,
        plan: CircuitPlan,
    ) -> str:
        """Get specific guidance based on failure reason"""
        reason_lower = reason.lower()
        
        if "no netlist" in reason_lower:
            return (
                "You must include your SPICE netlist inside a ```spice code block. "
                "Start with ```spice and end with ```"
            )
        elif "invalid" in reason_lower or "empty" in reason_lower:
            return (
                "Your netlist appears to be empty or contain only comments. "
                "Add actual circuit components (resistors, capacitors, sources, etc.)"
            )
        elif "compile" in reason_lower:
            return (
                "Your netlist has syntax errors. Check: "
                "1) Component names start with R, C, L, V, I, E, etc. "
                "2) All nodes are properly connected "
                "3) The netlist ends with .end"
            )
        elif "simulation failed" in reason_lower or "failed" in reason_lower:
            return (
                "The simulation could not run. Common causes: "
                "1) Missing input voltage source (add Vin) "
                "2) Floating nodes (connect all nodes) "
                "3) Missing ground connection (use node 0)"
            )
        elif "flat" in reason_lower:
            return (
                "The output is flat (no signal). Check: "
                "1) Input source is properly connected "
                "2) Op-amp has power supply connections "
                "3) Component values are appropriate (not too large/small)"
            )
        
        return "Review your circuit design and ensure all components are properly connected."

    def _is_ollama(self) -> bool:
        """Return True if we're talking to a local Ollama instance."""
        return bool(self.base_url and ("11434" in self.base_url or "ollama" in self.base_url.lower()))

    def _call_llm_ollama_native(self, system_prompt: str, user_prompt: str) -> str:
        """Call Ollama via its native /api/chat endpoint with think=false.

        The OpenAI-compat endpoint ignores think=false for some models (e.g. qwen3.5)
        and routes all output to the 'reasoning' field, leaving content empty.
        The native endpoint handles this correctly.
        """
        import re
        import requests

        # Derive native API base from the OpenAI-compat base_url
        # e.g. http://localhost:11434/v1  →  http://localhost:11434
        native_base = self.base_url.rstrip("/")
        if native_base.endswith("/v1"):
            native_base = native_base[:-3]
        url = f"{native_base}/api/chat"

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM API for netlist generation with retry logic"""
        import time
        import re

        max_retries = 5
        base_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                # Use native Ollama API when running locally — avoids think=false
                # being silently ignored by the OpenAI-compat endpoint.
                if self._is_ollama():
                    return self._call_llm_ollama_native(system_prompt, user_prompt)

                # OpenRouter / cloud path
                supports_system = "gemma" not in self.model.lower()
                if supports_system:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                else:
                    messages = [
                        {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"},
                    ]

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = completion.choices[0].message.content or ""
                return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  Rate limited. Retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                print(f"LLM API error: {e}")
                return ""

        return ""

    def extract_netlist(self, response: str) -> str:
        """
        Extract SPICE netlist from LLM response.
        
        Uses the shared utility function for consistency.
        """
        from circuit_utils import extract_netlist
        return extract_netlist(response)
