import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import traci  # type: ignore

# Add the src directory to Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from .traffic import TrafficController

# Add SUMO_HOME to PATH if not already there
if "SUMO_HOME" in os.environ:
    tools: str = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")


class SUMOTrafficSimulation:
    def __init__(
        self,
        sumo_config_file: str,
        traffic_data_file: str | None = None,
        sequence_length: int = 3,
        control_interval: int = 30,
        traffic_model_dir: str | None = None,
    ) -> None:
        """
        Initialize SUMO Traffic Control.

        Args:
            sumo_config_file (str): Path to SUMO config file
            traffic_data_dir (str | None): Directory to save traffic data
            sequence_length (int): Number of states to consider for prediction
            control_interval (int): How often to update traffic light (seconds)
            traffic_model_dir (str | None): Directory containing trained model. If None, manual control is used.
        """
        self.sumo_config_file: str = sumo_config_file
        self.traffic_control_interval: int = control_interval
        self.traffic_light_id = "I4_TL"  # From intersection.net.xml
        self.traffic_data_file: str | None = traffic_data_file
        self.traffic_sequence_length: int = sequence_length
        self.manual: bool = traffic_model_dir is None

        # Initialize traffic controller
        if not self.manual and traffic_model_dir is not None:
            self.traffic_control_model = TrafficController(model_dir=traffic_model_dir)

        # Initialize state history
        self.state_history: List[Dict[str, Any]] = []

        # Map SUMO detector IDs to directions
        self.detector_map: Dict[str, List[str]] = {
            "north": ["E_N"],
            "south": ["E_S"],
            "east": ["E_E"],
            "west": ["E_W"],
        }

        # Map controller states to SUMO phases
        self.phase_map: Dict[str, int] = {
            "north_green": 7,  # North-South green
            "north_yellow": 8,  # North-South yellow
            "east_green": 1,  # East-West green
            "east_yellow": 2,  # East-West yellow
            "south_green": 3,  # South-North green
            "south_yellow": 4,  # South-North yellow
            "west_green": 5,  # West-East green
            "west_yellow": 6,  # West-East yellow
        }

    def _get_sumo_traffic_state(self) -> Dict[str, Any]:
        """
        Get current traffic state from SUMO.

        Returns:
            Dict[str, Any]: Current traffic state
        """
        # Get current simulation time
        sim_time: float = traci.simulation.getTime()  # type: ignore

        # Convert simulation time to formatted timestamp
        timestamp: str = datetime.fromtimestamp(sim_time).strftime("%Y-%m-%dT%H:%M:%S")

        state: Dict[str, Any] = {
            "vehicles": {},
            "emergency": {},
            "light": {},
            "timestamp": timestamp,
        }

        # Get vehicle counts and emergency vehicle presence for each direction
        for direction, detectors in self.detector_map.items():
            # Count vehicles
            vehicle_count = 0
            emergency_present = False

            for detector in detectors:
                vehicles: Any | tuple[Any, ...] = traci.edge.getLastStepVehicleIDs(  # type: ignore
                    detector
                )
                vehicle_count += len(vehicles)  # type: ignore

                # Check for emergency vehicles
                for vehicle in vehicles:  # type: ignore
                    if traci.vehicle.getVehicleClass(vehicle) == "emergency":  # type: ignore
                        emergency_present = True
                        break

            state["vehicles"][direction] = vehicle_count
            state["emergency"][direction] = emergency_present

        # Get traffic light state
        tl_state: Any | tuple[Any, ...] = traci.trafficlight.getRedYellowGreenState(  # type: ignore
            self.traffic_light_id
        )

        # Convert SUMO state to our format
        state["light"] = self._sumo_state_to_trafix_state(tl_state)  # type: ignore

        return state

    def _sumo_state_to_trafix_state(self, sumo_state: str) -> Dict[str, str]:
        """
        Convert SUMO traffic light state to our format.

        Args:
            sumo_state (str): SUMO traffic light state string

        Returns:
            Dict[str, str]: Traffic light state in our format
        """
        # Default all to red
        state: Dict[str, str] = {
            "north": "red",
            "south": "red",
            "east": "red",
            "west": "red",
        }

        # Update based on SUMO state
        # SUMO state is a 12-character string representing all possible turns
        # We simplify to main directions
        if "g" in sumo_state[:3]:  # North
            state["north"] = "green"
        elif "y" in sumo_state[:3]:
            state["north"] = "yellow"

        if "g" in sumo_state[3:6]:  # East
            state["east"] = "green"
        elif "y" in sumo_state[3:6]:
            state["east"] = "yellow"

        if "g" in sumo_state[6:9]:  # South
            state["south"] = "green"
        elif "y" in sumo_state[6:9]:
            state["south"] = "yellow"

        if "g" in sumo_state[9:]:  # West
            state["west"] = "green"
        elif "y" in sumo_state[9:]:
            state["west"] = "yellow"

        return state

    def _trafix_state_to_sumo_phase(self, trafix_state: Dict[str, str]) -> int:
        """
        Convert traffic light state to SUMO phase number.

        Args:
            state (Dict[str, str]): Traffic light state

        Returns:
            int: SUMO phase number
        """
        # Find which direction is green/yellow
        for direction, color in trafix_state.items():
            if color in ["green", "yellow"]:
                return self.phase_map[f"{direction}_{color}"]

        return 0  # Default to all red

    def run(self, steps: int = 3600) -> None:
        """
        Run the traffic simulation.

        Args:
            steps (int): Number of simulation steps to run
        """
        # Start SUMO
        traci.start(["sumo-gui", "-c", self.sumo_config_file])  # type: ignore

        # Date and time
        try:
            step = 0
            next_control: int = self.traffic_control_interval

            while step < steps:
                # Simulate one step
                traci.simulationStep()

                # Get current state
                current_state: Dict[str, Any] = self._get_sumo_traffic_state()

                # Add to history
                self.state_history.append(current_state)

                # Update traffic light if it's time
                if (
                    step >= next_control
                    and len(self.state_history) >= self.traffic_sequence_length
                ):
                    # Get sequence for prediction
                    sequence: List[Dict[str, Any]] = self.state_history[
                        -self.traffic_sequence_length :
                    ]

                    if self.manual:  # Manual control
                        print(json.dumps(sequence, indent=2))

                        # List all available phases with numbered index
                        for idx, (phase_name) in enumerate(self.phase_map.keys()):
                            print(f"{idx}: {phase_name}")

                        # Allow user to select the next phase
                        selected_idx = int(input("Select the next phase index: "))
                        selected_phase_name: str = list(self.phase_map.keys())[
                            selected_idx
                        ]
                        phase = self.phase_map[selected_phase_name]

                        # Clear the print output
                        os.system("cls" if os.name == "nt" else "clear")

                        traci.trafficlight.setPhase(self.traffic_light_id, phase)  # type: ignore

                    else:  # Use model for control
                        # Predict next phase
                        next_state: dict[str, Any] = self.traffic_control_model.predict(
                            sequence
                        )

                        # Get next phase
                        next_phase: int = self._trafix_state_to_sumo_phase(next_state)

                        traci.trafficlight.setPhase(self.traffic_light_id, next_phase)  # type: ignore

                    # Update next control time
                    next_control += self.traffic_control_interval

                # Keep previous traffic light state
                else:
                    # Get previous traffic light state
                    prev_state: Dict[str, Any] = self.state_history[-1]
                    phase: int = self._trafix_state_to_sumo_phase(prev_state["light"])

                    traci.trafficlight.setPhase(self.traffic_light_id, phase)  # type: ignore

                step += 1
        except KeyboardInterrupt:
            traci.close(wait=False)
        except traci.exceptions.FatalTraCIError:  # type: ignore
            pass
        finally:
            if self.traffic_data_file:
                with open(f"{self.traffic_data_file}", "w+") as f:
                    json.dump(self.state_history, f, indent=2)
