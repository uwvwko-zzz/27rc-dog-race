"""Launch the obstacle course in the MuJoCo viewer."""
from pathlib import Path
import argparse
import mujoco
import mujoco.viewer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preview either ROBOCON 2027 field without a robot policy.')
    parser.add_argument('--map', choices=('obstacle', 'task'), default='obstacle')
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / 'maps' / (
        'task_course_2027.xml' if args.map == 'task' else 'obstacle_course_2027.xml')
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)
