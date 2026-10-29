"""Generate a self-contained MJCF training course from ROBOCON 2027 V2.0 rules."""
from pathlib import Path
import math
import random
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'maps/obstacle_course_2027.xml'
GREEN = '0.35 0.62 0.20 1'
YELLOW = '0.95 0.85 0.08 1'
RED = '0.95 0.08 0.05 1'

def nums(values):
    return ' '.join(f'{x:.8g}' for x in values)

def build():
    root = ET.Element('mujoco', model='robocon_2027_obstacle_course')
    root.append(ET.Comment('Units: metres, kilograms, seconds. Layout is a training interpretation of Fig. 1-2, not the unpublished competition layout.'))
    ET.SubElement(root, 'compiler', angle='degree', autolimits='true')
    ET.SubElement(root, 'option', timestep='0.002', gravity='0 0 -9.81', integrator='implicitfast')
    ET.SubElement(root, 'statistic', center='0 0 0', extent='8')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', azimuth='90', elevation='-65', offwidth='1600', offheight='900')
    ET.SubElement(visual, 'headlight', ambient='0.12 0.12 0.12', diffuse='0.15 0.15 0.15', specular='0 0 0')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction='0.9 0.005 0.0001', condim='3', solref='0.01 1', solimp='0.9 0.95 0.001')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'material', name='course_matte', rgba='1 1 1 1',
                  specular='0.05', shininess='0.1', reflectance='0')
    default.find('geom').set('material', 'course_matte')
    # Each tire uses separate convex annular sectors: a single concave mesh would fill the hole.
    count = 48
    vertices = [(r*math.cos(a), r*math.sin(a), z) for z in (-0.0825, 0.0825)
                for r in (0.165, 0.275) for a in (-math.pi/count, math.pi/count)]
    ET.SubElement(asset, 'mesh', name='tire_sector', vertex=nums(v for p in vertices for v in p))
    ramp_length = 0.36 / math.tan(math.radians(20))
    ramp_vertices = [(-.4,0,0),(.4,0,0),(-.4,ramp_length,0),(.4,ramp_length,0),(-.4,0,.36),(.4,0,.36)]
    ET.SubElement(asset, 'mesh', name='ramp_wedge', vertex=nums(v for p in ramp_vertices for v in p))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', name='course_key_light', pos='0 0 8', dir='-0.3 -0.4 -1',
                  directional='true', castshadow='true', ambient='0.04 0.04 0.04',
                  diffuse='0.55 0.55 0.55', specular='0.08 0.08 0.08')
    ET.SubElement(world, 'light', name='course_fill_light', pos='0 0 8', dir='0.5 0.2 -1',
                  directional='true', castshadow='false', ambient='0 0 0',
                  diffuse='0.12 0.12 0.12', specular='0 0 0')
    ET.SubElement(world, 'camera', name='overview', pos='0 -10 12', xyaxes='1 0 0 0 0.768221 0.640184')
    ET.SubElement(world, 'camera', name='top', pos='0 0 15', xyaxes='1 0 0 0 1 0')
    def body(name, pos):
        return ET.SubElement(world, 'body', name=name, pos=nums(pos))
    def geom(parent, name, pos, size=None, kind='box', color=GREEN, **kw):
        attrs = dict(name=name, type=kind, pos=nums(pos), rgba=color)
        if size is not None: attrs['size'] = nums(size)
        attrs.update({k: str(v) for k,v in kw.items()})
        return ET.SubElement(parent, 'geom', **attrs)
    def line(parent, name, p, q, color=RED, width=.004):
        return ET.SubElement(parent, 'geom', name=name, type='capsule', fromto=nums((*p,*q)), size=str(width), rgba=color, contype='0', conaffinity='0', group='2')
    def square(parent, name, center, sx, sy, z):
        x,y=center
        for side,(p,q) in enumerate([((x-sx/2,y-sy/2,z),(x+sx/2,y-sy/2,z)),((x+sx/2,y-sy/2,z),(x+sx/2,y+sy/2,z)),((x+sx/2,y+sy/2,z),(x-sx/2,y+sy/2,z)),((x-sx/2,y+sy/2,z),(x-sx/2,y-sy/2,z))]):
            for i in range(8):
                a=[p[j]+(q[j]-p[j])*i/8 for j in range(3)]
                b=[p[j]+(q[j]-p[j])*(i+.55)/8 for j in range(3)]
                line(parent,f'{name}_{side}_{i}',a,b)
    # A finite box floor; boundary markings have no collision and are not walls.
    geom(world, 'arena_floor', (0,0,-.05), (7,3,.05), color=YELLOW)
    for i,(p,q) in enumerate([((-7,-3,.004),(7,-3,.004)),((7,-3,.004),(7,3,.004)),((7,3,.004),(-7,3,.004)),((-7,3,.004),(-7,-3,.004))]):
        line(world,f'boundary_{i}',p,q,'0.05 0.15 0.95 1',.008)
    square(world,'start_1_mark',(0,0),.8,.8,.006)
    ET.SubElement(world,'site',name='start_1',pos='0 0 0.005',size='.4 .4 .002',type='box',rgba='1 0 0 0',group='3')
    # Bridge and wall.
    b=body('balance_beam',(-2,1.55,0))
    geom(b,'balance_beam_surface',(0,0,.05),(1.5,.2,.05))
    b=body('high_wall',(.95,1.55,0))
    geom(b,'wall_main',(0,0,.1475),(.04,.4,.1475))
    geom(b,'wall_steel_cap',(0,0,.2975),(.04,.4,.0025),color='0.55 0.58 0.60 1')
    # V2.0: 0.1 m-high platforms and blocks; total length 3.5 m, two staggered rows.
    b=body('stepping_stones',(2.29,1.55,0))
    for label,x in [('entry',.4),('exit',3.1)]:
        geom(b,f'stones_{label}_platform',(x,0,.05),(.4,.4,.05))
    for row,(x0,y) in enumerate([(1.,.2),(1.3,-.2)]):
        for i in range(3): geom(b,f'stone_{row}_{i}',(x0+.6*i,y,.05),(.1,.1,.05))
    # Clearance frame: actual inner width .8 and beam UNDERSIDE z=.29.
    b=body('clearance_frame',(1.37,-1.88,0))
    for i,x in enumerate([-.4,0,.4]):
        for j,y in enumerate([-.41,.41]): geom(b,f'frame_leg_{i}_{j}',(x,y,.145),(.01,.01,.145))
        geom(b,f'frame_crossbar_{i}',(x,0,.30),(.01,.42,.01))
    for j,y in enumerate([-.41,.41]): geom(b,f'frame_longbar_{j}',(0,y,.30),(.41,.01,.01))
    # L-shaped gap assembly; gap is empty space over the arena floor.
    b=body('gap_assembly',(-5.95,-1.9,0))
    geom(b,'ramp_platform',(0,0,.18),(.4,.4,.18),color=YELLOW)
    geom(b,'ramp',(0,.4,0),kind='mesh',mesh='ramp_wedge')
    geom(b,'stairs_platform',(1.16,0,.18),(.4,.4,.18))
    geom(b,'stair_middle',(1.72,0,.12),(.16,.4,.12))
    geom(b,'stair_low',(2.04,0,.06),(.16,.4,.06))
    square(b,'start_2_mark',(0,0),.8,.8,.366)
    ET.SubElement(b,'site',name='start_2',pos='0 0 .365',size='.4 .4 .002',type='box',rgba='1 0 0 0',group='3')
    # Slalom: six poles, four in x and three in y with the corner shared.
    b=body('slalom',(3.2,-1.9,0))
    for i,(x,y) in enumerate([(0,0),(.9,0),(1.8,0),(2.7,0),(2.7,.9),(2.7,1.8)]):
        geom(b,f'slalom_pole_{i}',(x,y,.05),(.05,.05),kind='cylinder')
    for name,(x,y) in [('entry',(-.4,0)),('corner',(3.,-.3)),('exit',(2.7,2.2))]:
        ET.SubElement(b,'site',name=f'slalom_{name}',type='cylinder',pos=nums((x,y,.006)),size='.15 .002',rgba='1 0 0 0',group='3')
        for i in range(12):
            a=2*math.pi*i/12; c=a+2*math.pi*.55/12
            line(b,f'slalom_{name}_mark_{i}',(x+.15*math.cos(a),y+.15*math.sin(a),.006),(x+.15*math.cos(c),y+.15*math.sin(c),.006))
    b=body('tire_array',(-2.45,-2.3,0))
    for row in range(2):
        for col in range(4):
            t=ET.SubElement(b,'body',name=f'tire_{row}_{col}',pos=nums((.275+.55*col,.275+.55*row,.0825)))
            for k in range(count):
                geom(t,f'tire_{row}_{col}_sector_{k}',(0,0,0),kind='mesh',mesh='tire_sector',euler=nums((0,0,360*k/count)),color='0.045 0.045 0.05 1',friction='1.1 0.005 0.0001')
    # L pit footprint: [0,2]x[0,2] excluding bottom-right quadrant.
    b=body('brick_bark_pit',(-6.35,.35,0))
    b.append(ET.Comment('Fixed roughness surrogate; fill depth, fence thickness and material properties are not specified by the rules.'))
    geom(b,'pit_base_vertical',(.5,1,.015),(.5,1,.015),color='0.36 0.24 0.12 1')
    geom(b,'pit_base_horizontal',(1.5,1.5,.015),(.5,.5,.015),color='0.55 0.28 0.20 1')
    edges=[((0,0),(1,0)),((1,0),(1,1)),((1,1),(2,1)),((2,1),(2,2)),((2,2),(0,2)),((0,2),(0,0)),((0,2),(1,1))]
    for i,(p,q) in enumerate(edges):
        length=math.dist(p,q)
        geom(b,f'pit_rail_{i}',((p[0]+q[0])/2,(p[1]+q[1])/2,.075),(length/2,.01,.075),euler=nums((0,0,math.degrees(math.atan2(q[1]-p[1],q[0]-p[0])))))
    rng=random.Random(2027)
    idx={'brick':0,'bark':0}
    for ix in range(40):
        for iy in range(40):
            x=(ix+.5)*.05; y=(iy+.5)*.05
            if x>1 and y<1: continue
            if min(x,y,2-y)<.045 or (x>1 and min(y-1,2-x)<.045) or (y<1 and 1-x<.045) or abs(x+y-2)<.06: continue
            # Diagonal divider from (0,2) to (1,1).
            material='brick' if x+y>2 else 'bark'
            if material=='brick' and (ix%2 or iy%2): continue
            if material=='bark' and rng.random()>.65: continue
            if material=='brick':
                size=(rng.uniform(.022,.038),rng.uniform(.016,.024),rng.uniform(.012,.035)); color='0.60 0.30 0.20 1'
            else:
                size=(rng.uniform(.005,.01),rng.uniform(.005,.01),rng.uniform(.004,.009)); color='0.33 0.19 0.09 1'
            i=idx[material]; idx[material]+=1
            geom(b,f'pit_{material}_{i}',(x,y,.03+size[2]),size,color=color,euler=nums((0,0,rng.uniform(0,180))))
    if hasattr(ET, 'indent'):
        ET.indent(root, space='  ')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(OUTPUT,encoding='utf-8',xml_declaration=True)
    print(f'Generated {OUTPUT} ({len(root.findall(".//geom"))} geoms)')

if __name__=='__main__': build()
