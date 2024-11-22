import os
import sys
from typing import Any, Literal

# from .sumo import SUMOTrafficSimulation

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from aitrafix.traffic import TrafficController
from aitrafix.helpers import SUMOTrafficSimulation


def main(
    task: Literal["train", "sim"],
    model_dir: str | None,
    data_dir: str,
    road_dir: str,
    **kwargs: dict[str, Any],
) -> None:
    if task == "train" and model_dir is not None:
        # Initialize controller
        controller = TrafficController()

        # Train model
        controller.train(
            data_dir=f"{data_dir}/traffic-flow.json",
            seq_len=5,
            model_dir=model_dir,
            test_size=0.2,
            save_model=True,
            save_report=True,
        )

    elif task == "sim":
        # Initialize controller
        controller = SUMOTrafficSimulation(
            sumo_config_file=f"{road_dir}/intersection.sumocfg",
            sequence_length=5,
            traffic_model_dir=model_dir,
            traffic_data_file=f"{data_dir}/sumo-traffic-flow-3600.json",
            control_interval=5,
        )

        # Run simulation
        controller.run(steps=3600)  # Run for 1 hour


if __name__ == "__main__":
    # # Train model
    # main(
    #     task="train",
    #     model_dir="../models/traffic",
    #     data_dir="../data/traffic",
    #     road_dir="../data/road",
    # )

    # Run simulation with trained model
    main(
        task="sim",
        model_dir="../models/traffic",
        data_dir="../data/traffic",
        road_dir="../data/road",
    )

    # # Run simulation with manual control
    # main(
    #     task="sim",
    #     model_dir=None,
    #     data_dir="../data/traffic",
    #     road_dir="../data/road",
    # )
