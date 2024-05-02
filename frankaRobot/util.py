from pathlib import Path


def choose_robot_motion():
    repo_root_path = Path(__file__).parents[1]
    robot_motions_path = repo_root_path / "frankaRobot" / "robotMotionPoints"
    robot_motions = dict([(str(i), p) for i, p in enumerate(robot_motions_path.iterdir())
                          if p.is_file and p.suffix == ".csv"])
    lines = [f'{key} {value.name}' for key, value in robot_motions.items()]
    print("Robot Motions:")
    print('\n'.join(lines) + '\n')
    robot_motion_key = None
    while robot_motion_key not in robot_motions:
        robot_motion_key = input("Choose robot motion (by index): ")
    return robot_motions[robot_motion_key]
