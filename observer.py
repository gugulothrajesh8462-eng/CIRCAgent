#!/usr/bin/env python3
"""
Simulation Observer Module

Executes SPICE simulations using ngspice and captures results.
The observer runs simulations and returns data for evaluation.
"""

import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from circuit_utils import (
    clean_netlist,
    get_simulation_commands,
    get_input_source,
)


@dataclass
class SimulationResult:
    """Container for simulation results"""
    success: bool
    x_data: Optional[np.ndarray]
    y_data: Optional[np.ndarray]
    error_message: str = ""
    stdout: str = ""
    stderr: str = ""
    output_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "success": self.success,
            "has_data": self.x_data is not None and self.y_data is not None,
            "x_length": len(self.x_data) if self.x_data is not None else 0,
            "y_length": len(self.y_data) if self.y_data is not None else 0,
            "y_range": (
                float(np.max(self.y_data) - np.min(self.y_data))
                if self.y_data is not None else 0
            ),
            "error_message": self.error_message,
        }


class SimulationObserver:
    """
    Executes SPICE simulations and captures results.
    
    The observer:
    1. Takes a netlist string
    2. Prepares it for simulation (clean, add commands)
    3. Runs ngspice
    4. Captures output data and errors
    """

    def __init__(self, ngspice_path: str = "ngspice", timeout_ms: int = 30000):
        """
        Initialize the simulation observer.
        
        Args:
            ngspice_path: Path to ngspice executable
            timeout_ms: Simulation timeout in milliseconds
        """
        self.ngspice_path = ngspice_path
        self.timeout_ms = timeout_ms

    def run(
        self,
        netlist: str,
        circuit_type: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """
        Run a SPICE simulation.
        
        Args:
            netlist: Raw SPICE netlist string
            circuit_type: Type of circuit (filter, amplifier, oscillator, etc.)
            params: Circuit parameters for simulation setup
            
        Returns:
            SimulationResult with success status and data
        """
        params = params or {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cir_file = Path(tmpdir) / "circuit.cir"
            out_file = Path(tmpdir) / "output.dat"
            
            # Prepare the netlist
            full_netlist = self._prepare_netlist(
                netlist, circuit_type, params, str(out_file)
            )
            
            # Write circuit file
            with open(cir_file, 'w') as f:
                f.write(full_netlist)
            
            # Run ngspice
            return self._execute_ngspice(cir_file, out_file)

    def _prepare_netlist(
        self,
        netlist: str,
        circuit_type: str,
        params: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Prepare netlist for simulation"""
        # Clean the netlist
        clean = clean_netlist(netlist)

        # For oscillators, inject .ic on capacitor nodes so VCVS-based circuits self-start.
        # The VCVS (E source) forces vout = gain*(v+ - v-), overriding a vout IC.
        # Setting IC on the capacitor's positive node provides the initial perturbation.
        if 'oscillator' in circuit_type.lower():
            ic_lines = self._oscillator_ic_lines(clean)
            if ic_lines:
                clean = clean + "\n" + ic_lines

        # Get simulation commands
        sim_commands = self._get_commands_with_output(circuit_type, params, output_path)

        # Check if input source is needed
        netlist_lower = clean.lower()
        if 'vin' not in netlist_lower and circuit_type.lower() not in ['oscillator']:
            input_source = get_input_source(circuit_type, params)
            clean = clean + "\n" + input_source

        # Combine
        return clean + "\n" + sim_commands

    def _oscillator_ic_lines(self, netlist: str) -> str:
        """Return .ic directives for capacitor positive nodes to kick-start oscillation."""
        import re
        ic_nodes = set()
        for line in netlist.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('*'):
                continue
            # Match C elements: C<name> <node+> <node-> <value>
            m = re.match(r'^[Cc]\S+\s+(\S+)\s+(\S+)\s+', stripped)
            if m:
                pos_node = m.group(1).lower()
                neg_node = m.group(2).lower()
                # Prefer non-ground nodes; skip '0' and 'gnd'
                if pos_node not in ('0', 'gnd'):
                    ic_nodes.add(pos_node)
                elif neg_node not in ('0', 'gnd'):
                    ic_nodes.add(neg_node)
        return '\n'.join(f'.ic V({n})=0.1' for n in sorted(ic_nodes))

    def _get_commands_with_output(
        self,
        circuit_type: str,
        params: Dict[str, Any],
        output_path: str,
    ) -> str:
        """Get simulation commands with correct output path"""
        params = params.copy()
        output_node = params.get('output_node', 'vout')
        
        ctype = circuit_type.lower()
        if 'filter' in ctype:
            ctype = 'filter'
        elif 'amplifier' in ctype or 'amp' in ctype:
            ctype = 'amplifier'
        elif 'oscillator' in ctype or 'osc' in ctype:
            ctype = 'oscillator'
        elif 'clipper' in ctype:
            ctype = 'clipper'
        elif 'clamper' in ctype:
            ctype = 'clamper'
        
        if ctype in ['filter', 'amplifier']:
            start_freq = params.get('start_freq', 1)
            stop_freq = params.get('stop_freq', 1e6)
            return f"""
.control
ac dec 20 {start_freq} {stop_freq}
wrdata {output_path} vdb({output_node})
quit
.endc
.end
"""
        
        elif ctype == 'oscillator':
            freq = params.get('freq_hz', params.get('frequency', 1000))
            t_stop = max(0.01, 10 / freq) if freq > 0 else 0.01
            t_step = t_stop / 1000
            # uic tells ngspice to use .ic directives injected by _prepare_netlist
            return f"""
.control
tran {t_step:.2e} {t_stop:.2e} uic
wrdata {output_path} v({output_node})
quit
.endc
.end
"""
        
        elif ctype in ['clipper', 'clamper']:
            input_freq = params.get('input_freq', 1000)
            t_stop = max(0.005, 5 / input_freq) if input_freq > 0 else 0.005
            t_step = t_stop / 1000
            return f"""
.control
tran {t_step:.2e} {t_stop:.2e}
wrdata {output_path} v({output_node})
quit
.endc
.end
"""
        
        else:
            return f"""
.control
tran 1u 10m
wrdata {output_path} v({output_node})
quit
.endc
.end
"""

    def _execute_ngspice(
        self,
        cir_file: Path,
        out_file: Path,
    ) -> SimulationResult:
        """Execute ngspice simulation"""
        try:
            result = subprocess.run(
                [self.ngspice_path, '-b', str(cir_file)],
                capture_output=True,
                text=True,
                timeout=self.timeout_ms / 1000,
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Check if output file was created
            if out_file.exists():
                try:
                    data = np.loadtxt(out_file)
                    if data.ndim == 1:
                        # Single column (voltage only)
                        return SimulationResult(
                            success=True,
                            x_data=np.arange(len(data)),
                            y_data=data,
                            stdout=stdout,
                            stderr=stderr,
                            output_file_path=str(out_file),
                        )
                    elif data.ndim == 2 and data.shape[1] >= 2:
                        # Two columns (x, y)
                        return SimulationResult(
                            success=True,
                            x_data=data[:, 0],
                            y_data=data[:, 1],
                            stdout=stdout,
                            stderr=stderr,
                            output_file_path=str(out_file),
                        )
                except Exception as e:
                    return SimulationResult(
                        success=False,
                        x_data=None,
                        y_data=None,
                        error_message=f"Failed to parse output data: {e}",
                        stdout=stdout,
                        stderr=stderr,
                    )
            
            # Output file missing - simulation failed
            # Check for error indicators in stderr
            error_msg = self._extract_error(stderr, stdout)
            
            return SimulationResult(
                success=False,
                x_data=None,
                y_data=None,
                error_message=error_msg or "Output file not created",
                stdout=stdout,
                stderr=stderr,
            )
            
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                x_data=None,
                y_data=None,
                error_message=f"Simulation timeout after {self.timeout_ms}ms",
            )
        except FileNotFoundError:
            return SimulationResult(
                success=False,
                x_data=None,
                y_data=None,
                error_message="ngspice executable not found",
            )
        except Exception as e:
            return SimulationResult(
                success=False,
                x_data=None,
                y_data=None,
                error_message=f"Unexpected error: {e}",
            )

    def _extract_error(self, stderr: str, stdout: str) -> str:
        """Extract meaningful error message from ngspice output"""
        # Combine outputs for searching
        combined = stderr + "\n" + stdout
        combined_lower = combined.lower()
        
        # Look for common error patterns
        error_patterns = [
            ("undefined node", "Undefined node reference"),
            ("singular matrix", "Singular matrix - check for floating nodes"),
            ("time step too small", "Transient analysis failed - time step too small"),
            ("no such file", "File reference error"),
            ("syntax error", "SPICE syntax error"),
            ("unknown", "Unknown element or syntax"),
            ("can't find", "Missing element or model"),
            ("missing", "Missing required element"),
            ("fatal", "Fatal error"),
            ("error", "General error"),
        ]
        
        for pattern, message in error_patterns:
            if pattern in combined_lower:
                return message
        
        # Return first line of stderr if available
        if stderr.strip():
            lines = stderr.strip().split('\n')
            for line in lines:
                if line.strip():
                    return line[:200]  # Truncate long messages
        
        return ""

    def check_syntax(self, netlist: str) -> Tuple[bool, str]:
        """
        Quick syntax check without full simulation.
        
        Returns:
            (is_valid, error_message)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cir_file = Path(tmpdir) / "check.cir"
            
            # Clean and add minimal simulation
            clean = clean_netlist(netlist)
            
            # Add input source if missing
            if 'vin' not in clean.lower():
                clean += "\nVin vin 0 DC 0\n"
            
            # Add .op for quick parsing
            full = clean + "\n.op\n.end\n"
            
            with open(cir_file, 'w') as f:
                f.write(full)
            
            try:
                result = subprocess.run(
                    [self.ngspice_path, '-b', str(cir_file)],
                    capture_output=True,
                    text=True,
                    timeout=10000,
                )
                
                stderr = result.stderr.lower() if result.stderr else ""
                stdout = result.stdout.lower() if result.stdout else ""
                
                # Check for errors
                error_indicators = [
                    'error', 'fatal', 'undefined', 'unknown',
                    'can\'t find', 'no such', 'missing'
                ]
                
                for indicator in error_indicators:
                    if indicator in stderr or indicator in stdout:
                        return False, f"Syntax check failed: {indicator}"
                
                return result.returncode == 0, ""
                
            except Exception as e:
                return False, f"Syntax check error: {e}"
