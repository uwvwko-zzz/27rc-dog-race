"""Run the existing Go2-W ONNX deployment on the ROBOCON 2027 course.

Controller, observation history and actuator preparation are reused from the
bundled scripts/go2w_mujoco_validate.py implementation. Model and policy are
included locally; --project can select another compatible deployment tree.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import math
import os
from pathlib import Path
import sys
import threading
import time
import xml.etree.ElementTree as ET

# Choose the headless renderer before the reference deployment imports MuJoCo.
if '--gui' in sys.argv:
    os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT = Path(os.environ.get('GO2W_PROJECT') or ROOT)


class NativeKeyboard:
    """Poll X11 held keys; MuJoCo's key callback does not report key release."""

    def __init__(self):
        from Xlib import X, XK, display, error
        self.display = display.Display()
        self.X, self.error = X, error
        self.pid_atom = self.display.intern_atom('_NET_WM_PID')
        self.codes = {key: self.display.keysym_to_keycode(XK.string_to_keysym(key))
                      for key in 'wsadqe'}

    def pressed(self):
        # Ignore input when the viewer loses focus, including switching apps.
        try:
            window = self.display.get_input_focus().focus
            focused = False
            for _ in range(16):
                if not hasattr(window, 'get_full_property'):
                    break
                prop = window.get_full_property(self.pid_atom, self.X.AnyPropertyType)
                if prop is not None and len(prop.value):
                    focused = int(prop.value[0]) == os.getpid()
                    break
                parent = window.query_tree().parent
                if parent.id == window.id:
                    break
                window = parent
            if not focused:
                return set()
            keymap = self.display.query_keymap()
            return {key for key, code in self.codes.items()
                    if code and keymap[code // 8] & (1 << (code % 8))}
        except self.error.XError:
            return set()

    def close(self):
        self.display.close()


class HoldCommands:
    """Same held-key directions and speed magnitudes as the browser panel."""

    def __init__(self, vx, vy, yaw):
        self.magnitudes = (max(abs(vx), 0.5), max(abs(vy), 0.5), max(abs(yaw), 1.0))
        self.value = np.zeros(3, np.float32)
        self.reset = self.push = self.paused = False
        self.lock = threading.Lock()
        self.blocked = set()
        self.stop_requested = False

    def on_key(self, code):
        with self.lock:
            if code == ord('R'):
                self.reset = True
            elif code == ord('P'):
                self.push = True
            elif code == ord('X'):
                self.stop_requested = True
            elif code == ord(' '):
                self.paused = not self.paused

    def update(self, pressed):
        with self.lock:
            self.blocked.intersection_update(pressed)
            if self.stop_requested:
                self.blocked.update(pressed)
                self.stop_requested = False
            pressed = pressed - self.blocked
            for i, (positive, negative) in enumerate((('w', 's'), ('a', 'd'), ('q', 'e'))):
                self.value[i] = (self.magnitudes[i] if positive in pressed else
                                 -self.magnitudes[i] if negative in pressed else 0.0)


def load_deployment(project):
    source = project / 'scripts/go2w_mujoco_validate.py'
    if not source.is_file():
        raise FileNotFoundError(f'Deployment implementation not found: {source}')
    # The supplied entry point also adds this directory to sys.path.
    sys.path.insert(0, str(project))
    spec = importlib.util.spec_from_file_location('course_go2w_deployment', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # Required by dataclasses on Python 3.8.
    spec.loader.exec_module(module)
    return module


def compose_course(deploy, terrain, output, spawn, gui=False, terrains=None,
                   initial_map='obstacle', shadows=None):
    """Compose one or both courses without changing obstacle dimensions."""
    output.parent.mkdir(parents=True, exist_ok=True)
    robot_path = output.parent / 'go2w_native.xml'
    deploy.prepare_robot(robot_path)
    robot = ET.parse(robot_path).getroot()
    for light in list(robot.findall('./worldbody/light')):
        if light.get('name') in {'validation_key_light', 'validation_fill_light'}:
            robot.find('worldbody').remove(light)
    robot.set('model', 'go2w_robocon_2027')
    robot.insert(0, ET.Comment('Robot/controller: unitree_rl_mjlab. ROBOCON 2027 training maps.'))
    sources = terrains or {initial_map: terrain}
    multiple = len(sources) > 1
    for map_name, path in sources.items():
        course = ET.parse(path).getroot()
        compiler = course.find('compiler')
        if compiler.get('angle', 'degree') == 'degree':
            for node in course.iter():
                for attribute in ('euler', 'axisangle'):
                    if attribute in node.attrib:
                        values = [float(v) for v in node.get(attribute).split()]
                        if attribute == 'euler':
                            values = [math.radians(v) for v in values]
                        else:
                            values[-1] = math.radians(values[-1])
                        node.set(attribute, ' '.join(f'{v:.12g}' for v in values))
        # Namespaced assets/defaults allow both courses to occupy one model.
        renames = {}
        for asset in course.find('asset'):
            if asset.get('file') and not Path(asset.get('file')).is_absolute():
                directory = compiler.get('texturedir' if asset.tag == 'texture' else 'meshdir', '')
                asset.set('file', str((path.parent / directory / asset.get('file')).resolve()))
            if multiple and asset.get('name'):
                name = asset.get('name')
                renames[name] = f'{map_name}_{name}'
                asset.set('name', renames[name])
        for node in course.iter():
            for reference in ('material', 'mesh', 'texture', 'hfield'):
                if node.get(reference) in renames:
                    node.set(reference, renames[node.get(reference)])
        course_default = copy.deepcopy(course.find('default'))
        class_name = f'{map_name}_course' if multiple else 'course'
        course_default.set('class', class_name)
        robot.find('default').append(course_default)
        for geom in course.findall('./worldbody//geom'):
            geom.set('class', class_name)
            if gui:
                geom.set('name', f'map_{map_name}_' + geom.get('name'))
        if multiple:
            for site in course.findall('./worldbody//site'):
                site.set('name', f'{map_name}_' + site.get('name'))
        robot.find('asset').extend(list(course.find('asset')))
        for element in course.find('worldbody'):
            if element.tag in {'light', 'camera'} and robot.find(
                    f'./worldbody/{element.tag}[@name="{element.get("name")}"]') is not None:
                continue
            robot.find('worldbody').append(element)
        if map_name == initial_map:
            for tag in ('visual', 'statistic'):
                for existing in list(robot.findall(tag)):
                    robot.remove(existing)
                node = course.find(tag)
                if node is not None:
                    robot.append(copy.deepcopy(node))
    if multiple:
        # Freeze invisible task boxes while obstacle terrain is selected.
        # Disabling collisions alone would leave these bodies in free fall.
        equality = ET.SubElement(robot, 'equality')
        for body in robot.findall('./worldbody/body'):
            if body.get('name', '').startswith('box_') and body.find('freejoint') is not None:
                ET.SubElement(equality, 'weld', name='inactive_task_' + body.get('name'),
                              body1='world', body2=body.get('name'), solref='0.01 1',
                              active='true' if initial_map != 'task' else 'false')
        visual_map = robot.find('./visual/map')
        if visual_map is None:
            visual_map = ET.SubElement(robot.find('visual'), 'map')
        visual_map.set('shadowclip', '3')
        visual_map.set('shadowscale', '1')
        robot.find('statistic').set('extent', '8')
    base = robot.find('./worldbody/body[@name="base_link"]')
    x, y, z, yaw = spawn
    base.set('pos', f'{x:g} {y:g} {z:g}')
    base.set('quat', f'{math.cos(yaw/2):.12g} 0 0 {math.sin(yaw/2):.12g}')
    if shadows is None:
        shadows = not gui
    if not shadows:
        # The combined field's shadow maps cause speckling on shallow decals
        # and mesh faces. Directional lighting still preserves shape shading.
        for light in robot.findall('./worldbody/light'):
            light.set('castshadow', 'false')
    ET.ElementTree(robot).write(output, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    qpos = model.qpos0.copy()
    ctrl = np.zeros(model.nu)
    for name, angle in zip(deploy.LEG_JOINTS, deploy.DEFAULT_Q):
        jid = deploy.object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = deploy.object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{name}_motor')
        qpos[model.jnt_qposadr[jid]] = angle
        ctrl[aid] = angle
    keyframe = ET.SubElement(robot, 'keyframe')
    ET.SubElement(keyframe, 'key', name='course_home',
                  qpos=' '.join(f'{v:.12g}' for v in qpos),
                  ctrl=' '.join(f'{v:.12g}' for v in ctrl))
    ET.indent(robot, space='  ') if hasattr(ET, 'indent') else None
    ET.ElementTree(robot).write(output, encoding='utf-8', xml_declaration=True)
    return mujoco.MjModel.from_xml_path(str(output))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--policy', '--onnx', type=Path,
                        help='ONNX policy; default: PROJECT/artifacts/go2w_good/policy.onnx')
    parser.add_argument('--map', choices=('obstacle', 'task'), default='obstacle')
    parser.add_argument('--terrain-xml', type=Path, help='Override the selected map XML')
    parser.add_argument('--scene', type=Path, help='Override the combined scene output')
    parser.add_argument('--vx', type=float, default=0.0)
    parser.add_argument('--vy', type=float, default=0.0)
    parser.add_argument('--yaw', type=float, default=0.0, help='Command yaw rate, rad/s')
    parser.add_argument('--start', choices=('1', '2'), default='1')
    parser.add_argument('--spawn', nargs=4, type=float, metavar=('X', 'Y', 'Z', 'YAW'),
                        help='Override initial base position and heading (metres, radians)')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='Simulation seconds; headless default is 10, viewer default is unlimited')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--gui', action='store_true', help='Open the reference browser control panel')
    parser.add_argument('--gui-port', type=int, default=8765)
    parser.add_argument('--gui-fps', type=int, default=60)
    parser.add_argument('--gui-shadows', action='store_true',
                        help='Enable shadow maps in the browser (default: off for a clean image)')
    parser.add_argument('--no-open-browser', action='store_true')
    parser.add_argument('--build-only', action='store_true', help='Write/load combined XML without running policy')
    parser.add_argument('--snapshot', type=Path, help='Save a final simulation image (MUJOCO_GL=egl for headless)')
    args = parser.parse_args()
    if args.policy is None:
        args.policy = args.project.expanduser() / 'artifacts/go2w_good/policy.onnx'
    if args.gui and args.headless:
        parser.error('--gui and --headless cannot be used together')
    if args.gui_fps <= 0 or not 0 <= args.gui_port <= 65535:
        parser.error('--gui-fps must be positive and --gui-port must be in 0..65535')
    if args.duration < 0 or not np.isfinite([args.vx, args.vy, args.yaw, args.duration]).all():
        parser.error('Commands and duration must be finite; duration must be non-negative')
    if args.spawn is not None and not np.isfinite(args.spawn).all():
        parser.error('Spawn values must be finite')
    if args.terrain_xml is None:
        args.terrain_xml = ROOT / 'maps' / ('task_course_2027.xml' if args.map == 'task' else 'obstacle_course_2027.xml')
    if args.scene is None:
        args.scene = ROOT / 'generated' / ('go2w_maps.xml' if args.gui else
                             'go2w_task_course.xml' if args.map == 'task' else 'go2w_course.xml')
    return args


def initial_spawn(args):
    if args.spawn:
        return tuple(args.spawn)
    course = ET.parse(args.terrain_xml).getroot()
    name = f'start_{args.start}'
    site = course.find(f'.//site[@name="{name}"]')
    if site is None:
        raise ValueError(f'Terrain has no {name} site; provide --spawn X Y Z YAW')
    position = np.asarray([float(v) for v in site.get('pos').split()])
    # Both generated start sites are axis-aligned. Add parent body translations.
    parents = {child: parent for parent in course.iter() for child in parent}
    node = parents.get(site)
    while node is not None:
        if node.tag == 'body':
            if any(a in node.attrib for a in ('quat', 'euler', 'axisangle')):
                raise ValueError('Rotated start body: provide explicit --spawn')
            position += np.asarray([float(v) for v in node.get('pos', '0 0 0').split()])
        node = parents.get(node)
    # Base clearance of .4 m matches the reference deployment. Subtract marker offset.
    return (float(position[0]), float(position[1]), float(position[2] - .005 + .4),
            math.pi / 2 if args.map == 'task' else 0.0)


def make_player(deploy, model, policy, spawn, scene):
    if (model.nq, model.nv, model.nu) == (23, 22, 16):
        return deploy.Player(model, policy, spawn)
    # The reference Player's constructor requires a robot-only model. Dynamic
    # task boxes add free joints, but never enter the robot's actor observation.
    robot = mujoco.MjModel.from_xml_path(str(scene.parent / 'go2w_native.xml'))
    if model.nu != robot.nu or model.nq < robot.nq or model.nv < robot.nv:
        raise ValueError('Combined scene has incompatible robot dimensions')
    for field in ('jnt_type', 'jnt_qposadr', 'jnt_dofadr', 'jnt_bodyid'):
        if not np.array_equal(getattr(model, field)[:robot.njnt], getattr(robot, field)):
            raise ValueError(f'Combined scene changed robot {field}')
    if not np.array_equal(model.actuator_trnid, robot.actuator_trnid):
        raise ValueError('Combined scene changed robot actuator order')
    if not np.array_equal(model.sensor_adr[:robot.nsensor], robot.sensor_adr):
        raise ValueError('Combined scene changed robot sensor order')
    player = deploy.Player(robot, policy, spawn)
    player.model, player.data = model, mujoco.MjData(model)
    player.reset()  # mj_resetData restores the boxes as well as the robot.
    return player


def save_snapshot(model, data, output):
    from PIL import Image
    output.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=900, width=1600) as renderer:
        renderer.update_scene(data, camera='overview')
        Image.fromarray(renderer.render()).save(output)
    print(f'[snapshot] {output}')


def browser_maps(args):
    terrains = {'obstacle': ROOT / 'maps/obstacle_course_2027.xml',
                'task': ROOT / 'maps/task_course_2027.xml'}
    terrains[args.map] = args.terrain_xml
    spawns, cameras = {}, {}
    for name, path in terrains.items():
        settings = copy.copy(args)
        settings.map, settings.terrain_xml = name, path
        settings.start = args.start if name == 'obstacle' else '1'
        settings.spawn = args.spawn if name == args.map else None
        spawns[name] = initial_spawn(settings)
        terrain_model = mujoco.MjModel.from_xml_path(str(path))
        cameras[name] = {'statistic': {'extent': float(terrain_model.stat.extent),
                                     'center': terrain_model.stat.center.copy()}}
        for camera in ('overview', 'top'):
            index = mujoco.mj_name2id(terrain_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if index >= 0:
                cameras[name][camera] = (terrain_model.cam_pos[index].copy(), terrain_model.cam_quat[index].copy())
    return terrains, spawns, cameras


def restore_map_state(model, data, runtime):
    """Apply map-specific cameras and freeze only inactive task box joints."""
    active = runtime.maps.active_map
    statistic = runtime.course_cameras[active]['statistic']
    model.stat.extent = statistic['extent']
    model.stat.center[:] = statistic['center']
    if not hasattr(runtime, 'course_geom_groups'):
        runtime.course_geom_groups = model.geom_group.copy()
        runtime.course_site_groups = model.site_group.copy()
    for name, group in runtime.maps.groups.items():
        ids = group['ids']
        # Alpha=0 alone can leave invisible terrain casting shadows. Group 5
        # is excluded by MuJoCo's default viewer/render options entirely.
        model.geom_group[ids] = runtime.course_geom_groups[ids] if name == active else 5
    for index in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index) or ''
        for map_name in runtime.maps.map_names:
            if name.startswith(map_name + '_'):
                model.site_group[index] = runtime.course_site_groups[index] if map_name == active else 5
                break
    for index in range(model.neq):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index) or ''
        if name.startswith('inactive_task_'):
            data.eq_active[index] = active != 'task'
    for name in ('overview', 'top'):
        if name not in runtime.course_cameras[active]:
            continue
        position, quaternion = runtime.course_cameras[active][name]
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if index >= 0:
            model.cam_pos[index], model.cam_quat[index] = position, quaternion
    mujoco.mj_forward(model, data)


def make_browser_runtime(deploy, args, spawn, spawns, cameras):
    x, y, z, yaw = spawn
    quaternion = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    config = deploy.make_runtime_config(
        gui=True, title='2027 ROBOCON 障碍赛 / 任务赛 · Go2-W',
        maps={'obstacle': '2027 障碍赛场地', 'task': '2027 任务赛场地'},
        map_spawns={name: {'position': list(pose[:3]),
                          'quaternion': [math.cos(pose[3]/2), 0, 0, math.sin(pose[3]/2)]}
                    for name, pose in spawns.items()},
        kp=20, kd=1, torque_limit=45,
        initial_position=[x, y, z], initial_quaternion=quaternion,
        command=(max(abs(args.vx), 0.5), max(abs(args.vy), 0.5),
                 max(abs(args.yaw), 1.0), 0.4),
        cameras={'tracking': '机器人跟随', 'overview': '全场斜视', 'top': '全场俯视'},
        tracking_camera={'camera_distance': 3.0, 'camera_azimuth': 135.0,
                         'camera_elevation': -25.0},
        port=args.gui_port, open_browser=not args.no_open_browser,
        render={'width': 1280, 'height': 960, 'fps': args.gui_fps, 'jpeg_quality': 92},
    )
    # Match the supplied Go2-W GUI: this policy has no commanded height input.
    config['runtime_parameter_schema']['commands'] = [
        row for row in config['runtime_parameter_schema']['commands']
        if row[0] in {'linear_x', 'linear_y', 'yaw'}
    ]
    config['runtime_ui']['default_map'] = args.map
    runtime = deploy.RuntimeControl(config, map_names=list(spawns), base_body_name='base_link')
    runtime.course_cameras = cameras
    print(f'[browser] {runtime.panel.url}  W/S 前后 | A/D 横移 | Q/E 转向 | 面板重置/推力/调参')
    return runtime


def main():
    args = parse_args()
    args.project = args.project.expanduser().resolve()
    args.terrain_xml = args.terrain_xml.expanduser().resolve()
    args.scene = args.scene.expanduser().resolve()
    if args.scene == args.terrain_xml or args.scene == args.project / 'src/assets/robots/unitree_go2w/xmls/go2w.xml':
        raise ValueError('Scene output must differ from source XML files')
    deploy = load_deployment(args.project)
    spawn = initial_spawn(args)
    terrains, spawns, cameras = browser_maps(args) if args.gui else (None, None, None)
    model = compose_course(deploy, args.terrain_xml, args.scene, spawn, gui=args.gui,
                           terrains=terrains, initial_map=args.map,
                           shadows=not args.gui or args.gui_shadows)
    print(f'[scene] {args.scene}\n[model] nq={model.nq} nv={model.nv} nu={model.nu}, physics_dt={model.opt.timestep}')
    if args.build_only:
        return
    player = make_player(deploy, model, args.policy.expanduser(), spawn, args.scene)
    commands = (deploy.Commands(np.asarray([args.vx, args.vy, args.yaw], np.float32))
                if args.headless or args.gui else HoldCommands(args.vx, args.vy, args.yaw))
    print(f'[policy] {args.policy.resolve()}\n[spawn] {spawn}')
    if not args.gui:
        print('[keys] 按住 W/S 前后 | A/D 横移 | Q/E 转向，松开停止 | X 停止 | R 重置 | P 推力 | Space 暂停')
    runtime = make_browser_runtime(deploy, args, spawn, spawns, cameras) if args.gui else None
    viewer = None
    keyboard = None
    if not args.headless and not args.gui:
        from mujoco import viewer as mj_viewer
        keyboard = NativeKeyboard()
        viewer = mj_viewer.launch_passive(model, player.data, key_callback=commands.on_key)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = player.base_id
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 135.0, -25.0
    duration = args.duration or (10.0 if args.headless else 0.0)
    ticks = 0
    next_tick = time.perf_counter()
    try:
        while (viewer is None or viewer.is_running()) and (runtime is None or runtime.is_running()):
            if duration and ticks * deploy.POLICY_DT >= duration - 1e-9:
                break
            if keyboard is not None:
                commands.update(keyboard.pressed())
            if commands.reset:
                player.reset()
                commands.reset = False
            if commands.push:
                player.push()
                commands.push = False
            state = None
            if runtime is not None:
                runtime.update_command()
                with runtime.model_lock:
                    state = runtime.runtime_control(model, player.data)
                    command = runtime.config['command']
                    commands.value[:] = (command['linear_x'], command['linear_y'], command['yaw'])
                    if runtime.consume_reset():
                        position = runtime.config['simulation']['initial_position']
                        quat = runtime.config['simulation']['initial_quaternion']
                        player.spawn = (*position, 2 * math.atan2(quat[3], quat[0]))
                        player.reset()
                        runtime.reset_simulation_state()
                        restore_map_state(model, player.data, runtime)
            if not commands.paused:
                if runtime is None:
                    player.step(commands.value)
                else:
                    with runtime.model_lock:
                        player.step(commands.value, runtime=runtime, state=state)
                ticks += 1
                if not player.finite():
                    raise FloatingPointError('Non-finite simulation state')
                if ticks % round(1 / deploy.POLICY_DT) == 0:
                    print(f'[state {ticks*deploy.POLICY_DT:5.1f}s] pos={player.data.qpos[:3].round(3)} cmd={commands.value}')
            if viewer is not None:
                viewer.sync()
            if viewer is not None or runtime is not None:
                next_tick += deploy.POLICY_DT
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -deploy.POLICY_DT:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        if keyboard is not None:
            keyboard.close()
        if runtime is not None:
            runtime.close()
        if args.snapshot:
            save_snapshot(model, player.data, args.snapshot.expanduser().resolve())
        print(f'[done] simulated={ticks*deploy.POLICY_DT:.2f}s pos={player.data.qpos[:3].round(3)} finite={player.finite()}')
        if viewer is not None:
            # Match supplied deployment's workaround for GLFW teardown on this desktop.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == '__main__':
    main()
