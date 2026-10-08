"""MuJoCo sim-to-sim player for the blind Go2-W ONNX policy.

Terrain maps come from https://github.com/uwvwko-zzz/mujoco-gui.
Unlike ``scripts/play.py``, this script does not create an mjlab environment.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

try:
  import mujoco
  import onnxruntime as ort
  from runtime_control import (
    MapSpec, MotorCommandDelay, RuntimeControl, bundled_map_spec,
    compose_scene, make_runtime_config,
  )
except ImportError as exc:
  raise SystemExit(
    "Install the validation runtime first:\n"
    "python -m pip install 'git+https://github.com/uwvwko-zzz/mujoco-gui.git'"
  ) from exc


ROOT = Path(__file__).resolve().parents[1]
ROBOT_XML = ROOT / "src/assets/robots/unitree_go2w/xmls/go2w.xml"
PHYSICS_DT, POLICY_DT, HISTORY = 0.005, 0.02, 5
LEG_JOINTS = (
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
)
WHEEL_JOINTS = (
  "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
)
WHEEL_GEOMS = tuple(name.replace("_joint", "_collision") for name in WHEEL_JOINTS)
# mjlab writes actions in compiled MuJoCo actuator order, even though the
# observation terms above preserve the explicitly requested SDK joint order.
ACTION_LEG_JOINTS = (
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
ACTION_WHEEL_JOINTS = (
  "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
)
DEFAULT_Q = np.asarray(
  [0.1, 0.9, -1.8, -0.1, 0.9, -1.8,
   0.1, 0.9, -1.8, -0.1, 0.9, -1.8], dtype=np.float64,
)
ACTION_DEFAULT_Q = np.asarray(
  [-0.1, 0.9, -1.8, 0.1, 0.9, -1.8,
   -0.1, 0.9, -1.8, 0.1, 0.9, -1.8], dtype=np.float64,
)
KP = np.asarray([20.0, 20.0, 40.0] * 4)
KD = np.asarray([1.0, 1.0, 2.0] * 4)
EFFORT = np.asarray([23.5, 23.5, 45.0] * 4)
MAPS = (
  "rc26_track", "race_track", "stairs", "cross_stairs", "cross_slope",
  "google_barkour", "gap_jump", "hurdles", "suspended_steps",
  "perlin_rough", "dynamic_obstacles",
)
SPAWNS = {
  name: (0.0, 0.0, 0.4, 0.0) for name in ("flat", *MAPS)
}
SPAWNS["rc26_track"] = (3.7, -9.0, 0.4, math.pi / 2)
SPAWNS["google_barkour"] = (1.0, -1.5, 0.4, 0.0)


def object_id(model, kind, name):
  idx = mujoco.mj_name2id(model, kind, name)
  if idx < 0:
    raise ValueError(f"Model has no {kind.name} named {name!r}")
  return idx


def inverse_rotate(quat, vector):
  w, xyz = float(quat[0]), np.asarray(quat[1:4])
  return (
    vector * (2 * w * w - 1) - 2 * w * np.cross(xyz, vector)
    + 2 * xyz * np.dot(xyz, vector)
  )


class History:
  """Per-term, oldest-to-newest history matching mjlab CircularBuffer."""

  def __init__(self):
    self.values = deque(maxlen=HISTORY)

  def push(self, value):
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if not self.values:
      self.values.extend(value.copy() for _ in range(HISTORY))
    else:
      self.values.append(value.copy())
    return np.concatenate(self.values)

  def clear(self):
    self.values.clear()


class ActorObservation:
  names = (
    "ang", "gravity", "command", "phase", "leg_pos", "leg_vel",
    "wheel_vel", "action",
  )

  def __init__(self):
    self.history = {name: History() for name in self.names}

  def reset(self):
    for item in self.history.values():
      item.clear()

  def build(self, *, ang, quat, command, elapsed, leg_pos, leg_vel, wheel_vel, action):
    moving = np.linalg.norm(command) >= 0.1
    angle = 2 * math.pi * (elapsed % 0.6) / 0.6
    terms = {
      "ang": ang,
      "gravity": inverse_rotate(quat, np.asarray([0.0, 0.0, -1.0])),
      "command": command,
      "phase": (math.sin(angle), math.cos(angle)) if moving else (0.0, 0.0),
      "leg_pos": leg_pos - DEFAULT_Q,
      "leg_vel": leg_vel,
      "wheel_vel": wheel_vel,
      "action": action,
    }
    obs = np.concatenate([self.history[name].push(terms[name]) for name in self.names])
    obs = obs.astype(np.float32, copy=False)
    if obs.shape != (275,) or not np.all(np.isfinite(obs)):
      raise FloatingPointError(f"Invalid actor observation: {obs.shape}")
    return obs[None]


def flat_map(path):
  root = ET.Element("mujoco", {"model": "flat"})
  asset = ET.SubElement(root, "asset")
  ET.SubElement(asset, "texture", {
    "name": "grid", "type": "2d", "builtin": "checker",
    "rgb1": ".2 .27 .34", "rgb2": ".09 .13 .18", "width": "512", "height": "512",
  })
  ET.SubElement(asset, "material", {
    "name": "ground_mat", "texture": "grid", "texrepeat": "8 8", "texuniform": "true",
  })
  world = ET.SubElement(root, "worldbody")
  ET.SubElement(world, "geom", {
    "name": "ground", "type": "plane", "size": "30 30 .1",
    "material": "ground_mat", "friction": ".8 .02 .001", "condim": "3",
  })
  ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
  return MapSpec(path)


def prepare_robot(path):
  """Add the position actuators that mjlab creates programmatically."""
  root = ET.parse(ROBOT_XML).getroot()
  visual = ET.SubElement(root, "visual")
  ET.SubElement(visual, "headlight", {
    "ambient": "0.25 0.25 0.25",
    "diffuse": "0.65 0.65 0.65",
    "specular": "0.25 0.25 0.25",
  })
  worldbody = root.find("worldbody")
  ET.SubElement(worldbody, "light", {
    "name": "validation_key_light", "directional": "true",
    "pos": "0 0 5", "dir": "-0.3 -0.4 -1",
    "ambient": "0.2 0.2 0.2", "diffuse": "0.8 0.8 0.8",
    "specular": "0.2 0.2 0.2", "castshadow": "false",
  })
  ET.SubElement(worldbody, "light", {
    "name": "validation_fill_light", "directional": "true",
    "pos": "0 0 5", "dir": "0.5 0.2 -1",
    "ambient": "0.08 0.08 0.08", "diffuse": "0.35 0.38 0.42",
    "specular": "0.1 0.1 0.1", "castshadow": "false",
  })
  compiler = root.find("compiler")
  compiler.set("meshdir", str((ROBOT_XML.parent / "assets").resolve()))
  option = root.find("option")
  option.attrib.update({
    "timestep": str(PHYSICS_DT), "integrator": "implicitfast",
    "solver": "Newton", "iterations": "10", "ls_iterations": "20",
    "tolerance": "1e-8", "ls_tolerance": "0.01", "ccd_iterations": "100",
    "cone": "pyramidal", "impratio": "1", "gravity": "0 0 -9.81",
  })
  # mjlab applies these articulation overrides after loading the source XML.
  # Reproduce them explicitly in the standalone model.
  for joint in root.findall(".//joint[@name]"):
    name = joint.get("name")
    if name in LEG_JOINTS:
      joint.set("armature", "0.02" if "calf" in name else "0.01")
      joint.set("frictionloss", "0")
    elif name in WHEEL_JOINTS:
      # XmlVelocityActuatorCfg wraps the XML actuator without rewriting the
      # source joint, so retain the wheel's XML armature (0.02).
      joint.set("frictionloss", "0")
  for geom in root.findall(".//geom[@name]"):
    if geom.get("name") in WHEEL_GEOMS:
      geom.set("contype", "1")
      geom.set("conaffinity", "0")
  actuators = root.find("actuator")
  for joint, kp, kd, limit in zip(LEG_JOINTS, KP, KD, EFFORT):
    ET.SubElement(actuators, "position", {
      "name": f"{joint}_motor", "joint": joint,
      "kp": f"{kp:g}", "kv": f"{kd:g}", "inheritrange": "0",
      "ctrllimited": "false", "forcelimited": "true",
      "forcerange": f"{-limit:g} {limit:g}",
    })
  ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_scene(args):
  out = args.output_dir.expanduser().resolve() / ("all_maps" if args.gui else args.map)
  out.mkdir(parents=True, exist_ok=True)
  robot = out / "go2w_native.xml"
  prepare_robot(robot)
  if args.gui:
    specs = {"flat": flat_map(out / "flat.xml")}
    specs.update({
      name: bundled_map_spec(
        name, minimum_box_half_thickness=0.025,
        contact_params={"condim": "3"},
      ) for name in MAPS
    })
    if args.terrain_xml:
      specs["custom"] = MapSpec(args.terrain_xml.expanduser().resolve())
  else:
    if args.terrain_xml:
      name, spec = "custom", MapSpec(args.terrain_xml.expanduser().resolve())
    elif args.map == "flat":
      name, spec = "flat", flat_map(out / "flat.xml")
    else:
      name = args.map
      spec = bundled_map_spec(
        name, minimum_box_half_thickness=0.025,
        contact_params={"condim": "3"},
      )
    specs = {name: spec}
  scene = compose_scene(
    robot, specs, out / "scene.xml", robot_body_name="base_link",
    minimum_box_half_thickness=0.025,
  )
  return scene, specs, "custom" if args.terrain_xml else args.map


@dataclass
class Commands:
  value: np.ndarray
  reset: bool = False
  push: bool = False
  paused: bool = False

  def on_key(self, code):
    try:
      key = chr(code).upper()
    except (ValueError, OverflowError):
      return
    if key == "W":
      self.value[0] = min(1.5, self.value[0] + 0.1)
    elif key == "S":
      self.value[0] = max(-0.8, self.value[0] - 0.1)
    elif key in {"A", "Q"}:
      if key == "A":
        self.value[1] = min(0.8, self.value[1] + 0.1)
      else:
        self.value[2] = min(1.0, self.value[2] + 0.1)
    elif key in {"D", "E"}:
      if key == "D":
        self.value[1] = max(-0.8, self.value[1] - 0.1)
      else:
        self.value[2] = max(-1.0, self.value[2] - 0.1)
    elif key == "X":
      self.value[:] = 0.0
    elif key == "R":
      self.reset = True
    elif key == "P":
      self.push = True
    elif key == " ":
      self.paused = not self.paused
    print(f"[command] vx={self.value[0]:+.2f}, vy={self.value[1]:+.2f}, yaw={self.value[2]:+.2f}")


class Player:
  def __init__(self, model, policy, spawn):
    if (model.nq, model.nv, model.nu) != (23, 22, 16):
      raise ValueError(f"Expected nq/nv/nu=(23,22,16), got {(model.nq, model.nv, model.nu)}")
    self.model, self.data, self.spawn = model, mujoco.MjData(model), spawn
    self.obs, self.last_action, self.steps = ActorObservation(), np.zeros(16, np.float32), 0
    joint_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEG_JOINTS]
    wheel_ids = [object_id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in WHEEL_JOINTS]
    self.leg_qpos = np.asarray([model.jnt_qposadr[i] for i in joint_ids])
    self.leg_dof = np.asarray([model.jnt_dofadr[i] for i in joint_ids])
    self.wheel_qpos = np.asarray([model.jnt_qposadr[i] for i in wheel_ids])
    self.wheel_dof = np.asarray([model.jnt_dofadr[i] for i in wheel_ids])
    self.leg_act = np.asarray([
      object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_motor")
      for n in ACTION_LEG_JOINTS
    ])
    self.wheel_act = np.asarray([
      object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
      for n in ACTION_WHEEL_JOINTS
    ])
    self.motor_delay = MotorCommandDelay(PHYSICS_DT)
    gyro = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    self.gyro_adr = model.sensor_adr[gyro]
    self.base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    policy = policy.resolve()
    if not policy.is_file():
      raise FileNotFoundError(f"ONNX policy not found: {policy}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(
      str(policy), sess_options=session_options, providers=["CPUExecutionProvider"]
    )
    inp, out = self.session.get_inputs()[0], self.session.get_outputs()[0]
    if inp.shape[-1] != 275 or out.shape[-1] != 16:
      raise ValueError(f"Expected ONNX [1,275]->[1,16], got {inp.shape}->{out.shape}")
    self.input_name, self.output_name = inp.name, out.name
    self.reset()

  def reset(self):
    mujoco.mj_resetData(self.model, self.data)
    x, y, z, yaw = self.spawn
    self.data.qpos[:3] = (x, y, z)
    self.data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    self.data.qpos[self.leg_qpos], self.data.qpos[self.wheel_qpos] = DEFAULT_Q, 0.0
    self.data.qvel[:], self.data.ctrl[:] = 0.0, 0.0
    self.last_action[:], self.steps = 0.0, 0
    self.obs.reset()
    self.motor_delay.reset()
    mujoco.mj_forward(self.model, self.data)

  def push(self):
    self.data.xfrc_applied[self.base_id, :3] = (80.0, 0.0, 20.0)

  def step(self, command, runtime=None, state=None):
    obs = self.obs.build(
      ang=self.data.sensordata[self.gyro_adr:self.gyro_adr + 3], quat=self.data.qpos[3:7],
      command=command, elapsed=self.steps * POLICY_DT,
      leg_pos=self.data.qpos[self.leg_qpos], leg_vel=self.data.qvel[self.leg_dof],
      wheel_vel=self.data.qvel[self.wheel_dof], action=self.last_action,
    )
    action = np.asarray(
      self.session.run([self.output_name], {self.input_name: obs})[0], dtype=np.float32,
    ).reshape(-1)
    if action.shape != (16,) or not np.all(np.isfinite(action)):
      raise FloatingPointError("Policy produced an invalid action")
    self.last_action[:] = action
    target = ACTION_DEFAULT_Q + 0.5 * action[:12]
    wheel = np.clip(35 * action[12:], -35, 35)
    if state is not None:
      kp = KP * float(state["kp"]) / 20.0
      kd = KD * float(state["kd"])
      strength = float(state["motor_strength"])
      limit = EFFORT * float(state["torque_limit"]) / 45.0
      self.model.actuator_gainprm[self.leg_act, 0] = kp * strength
      self.model.actuator_biasprm[self.leg_act, 1] = -kp * strength
      self.model.actuator_biasprm[self.leg_act, 2] = -kd * strength
      self.model.actuator_forcerange[self.leg_act, 0] = -limit
      self.model.actuator_forcerange[self.leg_act, 1] = limit
    for _ in range(round(POLICY_DT / PHYSICS_DT)):
      if state is not None:
        delayed = self.motor_delay.apply(
          np.concatenate((target, wheel)), float(state["motor_delay_ms"])
        )
        self.data.ctrl[self.leg_act] = delayed[:12]
        self.data.ctrl[self.wheel_act] = delayed[12:]
      else:
        self.data.ctrl[self.leg_act] = target
        self.data.ctrl[self.wheel_act] = wheel
      if runtime is not None:
        runtime.apply_external_forces(self.model, self.data)
      mujoco.mj_step(self.model, self.data)
    self.data.xfrc_applied[:] = 0.0
    self.steps += 1

  def finite(self):
    return np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))


def default_policy():
  candidates = [
    ROOT / "artifacts/go2w_model_latest/policy.onnx",
    ROOT / "artifacts/go2w_model_7500/policy.onnx",
  ]
  candidates += sorted(
    (ROOT / "logs/rsl_rl/go2w_blind_velocity").glob("*/policy.onnx"),
    key=lambda p: p.stat().st_mtime, reverse=True,
  )
  return next((p for p in candidates if p.is_file()), candidates[0])


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--policy", "--onnx", dest="policy", type=Path,
                      default=default_policy(), help="Path to the ONNX policy")
  parser.add_argument("--map", choices=("flat", *MAPS), default="perlin_rough")
  parser.add_argument("--terrain-xml", type=Path)
  parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/go2w_mujoco_validation")
  parser.add_argument("--vx", type=float, default=0.5)
  parser.add_argument("--vy", type=float, default=0.0)
  parser.add_argument("--yaw", type=float, default=0.0)
  parser.add_argument("--spawn", type=float, nargs=4, metavar=("X", "Y", "Z", "YAW"))
  parser.add_argument("--duration", type=float, default=0.0)
  parser.add_argument("--headless", action="store_true")
  parser.add_argument("--gui", action="store_true", help="Open the browser control panel")
  parser.add_argument("--gui-port", type=int, default=8765)
  parser.add_argument("--gui-fps", type=int, default=120,
                      help="Browser render target FPS (default: 120)")
  parser.add_argument("--no-open-browser", action="store_true")
  return parser.parse_args()


def main():
  args = parse_args()
  if args.gui and args.headless:
    raise SystemExit("--gui and --headless cannot be used together")
  if args.gui_fps <= 0:
    raise SystemExit("--gui-fps must be positive")
  if abs(args.vy) > 1e-9:
    print("[warning] The bundled model_9000 policy was trained with vy=0; lateral control needs a newly trained policy.")
  scene, specs, selected_map = build_scene(args)
  model = mujoco.MjModel.from_xml_path(str(scene))
  spawn = tuple(args.spawn) if args.spawn else SPAWNS.get(selected_map, SPAWNS["flat"])
  player = Player(model, args.policy.expanduser(), spawn)
  commands = Commands(np.asarray([args.vx, args.vy, args.yaw], np.float32))
  print(f"[scene] {scene}\n[policy] {args.policy.resolve()}")
  runtime = None
  if args.gui:
    spawns = {}
    for name in specs:
      x, y, z, yaw = spawn if name == selected_map else SPAWNS.get(name, SPAWNS["flat"])
      spawns[name] = {
        "position": [x, y, z],
        "quaternion": [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
      }
    config = make_runtime_config(
      gui=True, title="Go2-W 盲狗 MuJoCo 验证", maps={name: name for name in specs},
      map_spawns=spawns, kp=20, kd=1, torque_limit=45,
      initial_position=spawns[selected_map]["position"],
      initial_quaternion=spawns[selected_map]["quaternion"],
      command=(max(abs(args.vx), 0.5), max(abs(args.vy), 0.5),
               max(abs(args.yaw), 1.0), 0.4),
      port=args.gui_port, open_browser=not args.no_open_browser,
      render={"width": 960, "height": 720, "fps": args.gui_fps},
    )
    config["runtime_ui"]["default_map"] = selected_map
    config["runtime_parameter_schema"]["commands"] = [
      row for row in config["runtime_parameter_schema"]["commands"]
      if row[0] in {"linear_x", "linear_y", "yaw"}
    ]
    runtime = RuntimeControl(config, map_names=list(specs), base_body_name="base_link")
    print(f"[browser] http://127.0.0.1:{args.gui_port}  W/S 前后 | A/D 横移 | Q/E 转向 | 地图/重置/推力在面板中")
  else:
    print("[keys] W/S vx | A/D vy | Q/E yaw | X stop | R reset | P push | Space pause")
  viewer = None
  if not args.headless and not args.gui:
    from mujoco import viewer as mujoco_viewer

    viewer = mujoco_viewer.launch_passive(
      model, player.data, key_callback=commands.on_key
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = player.base_id
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 135.0, -20.0
  duration = args.duration or (20.0 if args.headless else 0.0)
  start = next_tick = time.perf_counter()
  reported = -1
  while (viewer is None or viewer.is_running()) and (runtime is None or runtime.is_running()):
    elapsed = time.perf_counter() - start
    if duration and elapsed >= duration:
      break
    state = None
    if runtime is not None:
      runtime.update_command()
      state = runtime.runtime_control(model, player.data)
      command = runtime.config["command"]
      commands.value[:] = (command["linear_x"], command["linear_y"], command["yaw"])
      if runtime.consume_reset():
        position = runtime.config["simulation"]["initial_position"]
        quat = runtime.config["simulation"]["initial_quaternion"]
        yaw = 2 * math.atan2(quat[3], quat[0])
        player.spawn = (*position, yaw)
        player.reset()
        runtime.reset_simulation_state()
    if commands.reset:
      player.reset()
      commands.reset = False
    if commands.push:
      player.push()
      commands.push = False
    if not commands.paused:
      player.step(commands.value, runtime=runtime, state=state)
      if not player.finite():
        print("[recovery] Invalid simulation state; resetting")
        player.reset()
    if viewer:
      viewer.sync()
    if int(elapsed) != reported:
      reported = int(elapsed)
      position = player.data.qpos[:3].round(3)
      print(f"[state {elapsed:5.1f}s] pos={position} cmd={commands.value}")
    next_tick += POLICY_DT
    delay = next_tick - time.perf_counter()
    if delay > 0:
      time.sleep(delay)
    elif delay < -POLICY_DT:
      next_tick = time.perf_counter()
  # GLFW teardown in MuJoCo 3.x can segfault on this NVIDIA/X11 desktop after
  # viewer.sync(). Exit the one-shot GUI process before Python destroys it.
  if viewer:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
  if runtime is not None:
    runtime.close()


if __name__ == "__main__":
  main()
