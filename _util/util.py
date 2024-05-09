from pathlib import Path
from typing import Any, Callable

import numpy as np


def user_input_choose_from_list(choices: list[Any], caption: str, list_item_label_selector: Callable[[Any], str] | None = None):
    lines = [
        f'{i} {list_item_label_selector(v) if list_item_label_selector is not None else v}' for i, v in enumerate(choices)]
    print('\n' + '\n'.join(lines))
    index = None
    while index not in np.arange(0, len(choices), 1):
        index = int(input(f"Choose from {caption}: "))
    return choices[index]


def choose_robot_motion():
    repo_root_path = Path(__file__).parents[1]
    robot_motions_path = repo_root_path / "frankaRobot" / "robotMotionPoints"
    robot_motions = [p for _, p in enumerate(
        robot_motions_path.iterdir()) if p.is_file and p.suffix == ".csv"]
    return user_input_choose_from_list(robot_motions, "Robot Motions", lambda m: m.name)
