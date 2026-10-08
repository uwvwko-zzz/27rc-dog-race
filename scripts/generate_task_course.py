"""Generate the ROBOCON 2027 task field (rules section 3.2, figures 2–6)."""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('food', 'tool', 'instrument', 'medicine')
COLORS = ('0.35 0.62 0.20 1', '0.55 0.55 0.55 1', '0.12 0.32 0.70 1', '0.78 0.03 0.04 1')
RED = '0.95 0.08 0.05 1'

def nums(v):
    return ' '.join(f'{x:.10g}' for x in v)

def build():
    root = ET.Element('mujoco', model='robocon_2027_task_course')
    root.append(ET.Comment('SI units. Fig.2 start area is outside the south edge: a 1m opening is a simulation interpretation. Box row permutations are training examples.'))
    ET.SubElement(root, 'compiler', angle='degree', autolimits='true')
    ET.SubElement(root, 'option', timestep='0.002', gravity='0 0 -9.81', integrator='implicitfast')
    ET.SubElement(root, 'statistic', center='0 -.3 .1', extent='4.5')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1600', offheight='900', azimuth='90', elevation='-65')
    ET.SubElement(visual, 'headlight', ambient='0.12 0.12 0.12', diffuse='0.15 0.15 0.15', specular='0 0 0')
    # Wider shadow depth range avoids wall/floor shadow acne in the enclosed field.
    ET.SubElement(visual, 'map', shadowclip='3', shadowscale='1')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction='0.9 0.005 0.0001', condim='3', solref='0.01 1',
                  solimp='0.9 0.95 0.001', material='course_matte')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'material', name='course_matte', rgba='1 1 1 1', specular='0.05', shininess='0.1', reflectance='0')
    for category in CATEGORIES:
        ET.SubElement(asset, 'texture', name=f'{category}_label_texture', type='2d',
                      file=f'../assets/task_labels/{category}.png')
        ET.SubElement(asset, 'material', name=f'{category}_label_material',
                      texture=f'{category}_label_texture', texuniform='false',
                      rgba='1 1 1 1', specular='0', shininess='0')
    # One textured visual cube avoids overlapping label plates and their shadows.
    # Face right/up vectors keep all four side labels upright and outward-facing.
    face_frames = [((0,0,.1),(1,0,0),(0,1,0)),
                   ((0,0,-.1),(1,0,0),(0,-1,0)),
                   ((.1,0,0),(0,1,0),(0,0,1)),
                   ((-.1,0,0),(0,-1,0),(0,0,1)),
                   ((0,.1,0),(-1,0,0),(0,0,1)),
                   ((0,-.1,0),(1,0,0),(0,0,1))]
    vertices, faces, texcoords = [], [], []
    for i,(center,right,up) in enumerate(face_frames):
        for x,y in [(-.1,-.1),(.1,-.1),(-.1,.1),(.1,.1)]:
            vertices.extend(center[j]+x*right[j]+y*up[j] for j in range(3))
        faces.extend((4*i,4*i+1,4*i+2,4*i+1,4*i+3,4*i+2))
        texcoords.extend((0,1,1,1,0,0,1,0))
    ET.SubElement(asset, 'mesh', name='label_cube', vertex=nums(vertices),
                  face=' '.join(str(i) for i in faces), texcoord=nums(texcoords),
                  smoothnormal='false')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', name='course_key_light', pos='0 0 8', dir='-0.3 -0.4 -1',
                  directional='true', castshadow='true', ambient='0.04 0.04 0.04',
                  diffuse='0.55 0.55 0.55', specular='0.08 0.08 0.08')
    ET.SubElement(world, 'light', name='course_fill_light', pos='0 0 8', dir='0.5 0.2 -1',
                  directional='true', castshadow='false', ambient='0 0 0',
                  diffuse='0.12 0.12 0.12', specular='0 0 0')
    ET.SubElement(world, 'camera', name='top', pos='0 -.4 10', xyaxes='1 0 0 0 1 0')
    ET.SubElement(world, 'camera', name='overview', pos='6 -9 8', xyaxes='0.83205 0.55470 0 -0.32444 0.48666 0.81111')
    def geom(parent, name, pos, size=None, color='1 1 1 1', **kwargs):
        attrs=dict(name=name, type='box', pos=nums(pos), rgba=color)
        if size is not None: attrs['size']=nums(size)
        attrs.update(kwargs)
        return ET.SubElement(parent, 'geom', **attrs)
    def line(name,p,q,color=RED,width=.002):
        return ET.SubElement(world,'geom',name=name,type='capsule',fromto=nums((*p,*q)),size=str(width),
                             rgba=color,contype='0',conaffinity='0',group='2')
    def outline(name,x,y,width,depth,dashed=False):
        corners=[(x-width/2,y-depth/2,.003),(x+width/2,y-depth/2,.003),
                 (x+width/2,y+depth/2,.003),(x-width/2,y+depth/2,.003)]
        for side in range(4):
            p,q=corners[side],corners[(side+1)%4]
            for i in range(8 if dashed else 1):
                n=8 if dashed else 1
                a=[p[j]+(q[j]-p[j])*i/n for j in range(3)]
                b=[p[j]+(q[j]-p[j])*(i+(.55 if dashed else 1))/n for j in range(3)]
                line(f'{name}_{side}_{i}',a,b)
    yellow='0.95 0.85 0.08 1'; orange='0.95 0.43 0.08 1'
    geom(world,'task_floor',(0,0,-.05),(2,3,.05),yellow)
    geom(world,'task_start_floor',(0,-3.4,-.05),(.5,.4,.05),yellow)
    # Walls sit outside the 6x4 inner playing surface. Minimum official height .6m.
    for name,pos,size in [('west',(-2.025,0,.3),(.025,3.05,.3)),
                          ('east',(2.025,0,.3),(.025,3.05,.3)),
                          ('north',(0,3.025,.3),(2,.025,.3)),
                          ('south_left',(-1.25,-3.025,.3),(.75,.025,.3)),
                          ('south_right',(1.25,-3.025,.3),(.75,.025,.3))]:
        geom(world,'task_wall_'+name,pos,size,orange)
    outline('task_start_mark',0,-3.4,1,.8,True)
    ET.SubElement(world,'site',name='start_1',type='box',pos='0 -3.4 .005',size='.5 .4 .002',rgba='1 0 0 0',group='3')
    # Storage markings: .2m squares, .8m clear gaps, overall 3.2x1.2m.
    rows=(-1.7,-.7); columns=(-1.5,-.5,.5,1.5)
    # Near row in category order, far row a reproducible sample permutation.
    permutations=((0,1,2,3),(2,0,3,1))
    for row,y in enumerate(rows):
        for col,x in enumerate(columns):
            outline(f'storage_{row}_{col}_mark',x,y,.2,.2)
            ET.SubElement(world,'site',name=f'storage_{row}_{col}',type='box',pos=nums((x,y,.001)),
                          size='.1 .1 .001',rgba='1 0 0 0',group='3')
            category=CATEGORIES[permutations[row][col]]
            box=ET.SubElement(world,'body',name=f'box_{category}_{row}',pos=nums((x,y,.101)))
            ET.SubElement(box,'freejoint',name=f'box_{category}_{row}_joint')
            inertia=.25*.2**2/6
            ET.SubElement(box,'inertial',pos='0 0 0',mass='.25',diaginertia=nums((inertia,)*3))
            geom(box,f'box_{category}_{row}_collision',(0,0,0),(.1,.1,.1),
                 friction='0.6 0.005 0.0001',group='3')
            geom(box,f'box_{category}_{row}_visual',(0,0,0),type='mesh',mesh='label_cube',
                 material=f'{category}_label_material',contype='0',conaffinity='0',mass='0',group='2')
    # Return zones: .3m squares with .6m clear gaps, left edge .5m from wall.
    for i,(category,color) in enumerate(zip(CATEGORIES,COLORS)):
        x=-1.35+.9*i
        # Painted regions are decorations, so they should not cast self-shadows.
        ET.SubElement(world,'site',name=f'return_{category}_mark',type='box',
                      pos=nums((x,1.85,.001)),size='.15 .15 .001',rgba=color,
                      material='course_matte',group='2')
        ET.SubElement(world,'site',name=f'return_{category}',type='box',pos=nums((x,1.85,.002)),
                      size='.15 .15 .001',rgba='1 1 1 0',group='3')
    # Fixed U-shaped .8m wide, .2m high barriers; .075m tube diameter.
    radius=.0375
    for side,center in [('left',-.9),('right',.9)]:
        barrier=ET.SubElement(world,'body',name='parking_bar_'+side,pos=nums((center,.6,0)))
        half=.4-radius; z=.2-radius
        for leg,x in enumerate((-half,half)):
            ET.SubElement(barrier,'geom',name=f'bar_{side}_leg_{leg}',type='cylinder',
                          fromto=nums((x,0,0,x,0,z)),size=str(radius),rgba=yellow)
            ET.SubElement(barrier,'geom',name=f'bar_{side}_corner_{leg}',type='sphere',
                          pos=nums((x,0,z)),size=str(radius),rgba=yellow)
        for segment in range(5):
            a=-half+2*half*segment/5; b=-half+2*half*(segment+1)/5
            ET.SubElement(barrier,'geom',name=f'bar_{side}_stripe_{segment}',type='cylinder',
                          fromto=nums((a,0,z,b,0,z)),size=str(radius),
                          rgba=yellow if segment%2==0 else '0.025 0.025 0.025 1')
    if hasattr(ET,'indent'): ET.indent(root,space='  ')
    path=ROOT/'maps/task_course_2027.xml'
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)
    print(f'Generated {path}; 8 dynamic boxes, 8 storage zones, 4 return zones, 2 parking bars')

if __name__=='__main__': build()
