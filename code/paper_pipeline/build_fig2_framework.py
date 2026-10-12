#!/usr/bin/env python3
"""Figure 2 (DeepST-INLA framework schematic), drawn as a 180 mm wide SVG.

Writes paper/figures/Fig2_DeepST-INLA_framework.svg and, when Playwright/Chromium is
available, the vector PDF (text kept as embedded TrueType) and a 600 dpi PNG preview.
Palette is shared with build_natcom_figures.py (validated for colour-vision deficiency).
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
import math, random
random.seed(3)
W,H=1020,490
ITS=' font-style="italic"'
def DS(d): return ' stroke-dasharray="%s"'%d if d else ''
F='Helvetica, Arial, sans-serif'
BL,BLt='#3F74AD','#E4EDF7'; GR,GRt='#009E73','#E0F4EC'; OR,ORt='#D55E00','#FBEADD'; PU,PUt='#7B2D8E','#F3E8F6'
GY='#6B7280'; DK='#1F2430'; LG='#C9CDD3'; SL='#56606E'
o=[]
def a(s): o.append(s)
def t(x,y,s,size=13,w='normal',fill=DK,anchor='start',it=False):
    a(f'<text x="{x}" y="{y}" font-family="{F}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}"{ITS if it else ""}>{s}</text>')
# sub/superscripts are never set below 5 pt (10 units) so that all text stays within 5-7 pt
def sub(s,size=10): return f'<tspan baseline-shift="sub" font-size="{max(size,10)}">{s}</tspan>'
def sup(s,size=10): return f'<tspan baseline-shift="super" font-size="{max(size,10)}">{s}</tspan>'
LOSS='<tspan font-style="italic">L</tspan>'
def rect(x,y,w,h,fill='none',stroke='none',sw=1,rx=0,dash=None,op=1):
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{DS(dash)} opacity="{op}"/>')
def circ(x,y,r,fill='none',stroke='none',sw=1,op=1):
    a(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"/>')
def line(x1,y1,x2,y2,stroke=GY,sw=1,dash=None,op=1):
    a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{DS(dash)} opacity="{op}" stroke-linecap="round"/>')
def path(d,fill='none',stroke='none',sw=1,op=1,dash=None):
    a(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}" stroke-linejoin="round" stroke-linecap="round"{DS(dash)}/>')
mk={BL:'b',GR:'g',OR:'o',PU:'p',GY:'k',SL:'s'}
def arrow(pts,col=GY,sw=2,dash=None):
    d='M '+' L '.join(f'{x} {y}' for x,y in pts)
    a(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round" marker-end="url(#ah-{mk[col]})"{DS(dash)}/>')
def sigma(x,y,r=17,col=PU):
    circ(x,y,r,'#FFFFFF',col,2); t(x,y+6,'Σ',17,'bold',col,'middle')
def lock(x,y,col=PU):
    path(f'M {x-5} {y} L {x-5} {y-5} A 5 5 0 0 1 {x+5} {y-5} L {x+5} {y}',stroke=col,sw=2)
    rect(x-8,y,16,12,col,rx=2); circ(x,y+6,1.8,'#FFFFFF')
def density(x0,y0,w,h,col,tint,mu=0.35,s=0.45,rug=True,thr=None):
    pts=[];N=80
    xs=[i/N for i in range(N+1)]
    def f(u):
        u=max(u,1e-3); return math.exp(-(math.log(u/mu))**2/(2*s*s))/u
    mx=max(f(u) for u in xs)
    for u in xs: pts.append((x0+u*w, y0+h-f(u)/mx*(h-8)))
    d=f'M {x0} {y0+h} '+' '.join(f'L {px:.1f} {py:.1f}' for px,py in pts)+f' L {x0+w} {y0+h} Z'
    path(d,fill=tint,stroke='none')
    path('M '+' L '.join(f'{px:.1f} {py:.1f}' for px,py in pts),stroke=col,sw=2)
    if thr is not None:
        tx=x0+thr*w
        tp=[(px,py) for px,py in pts if px>=tx]
        d=f'M {tx} {y0+h} '+' '.join(f'L {px:.1f} {py:.1f}' for px,py in tp)+f' L {x0+w} {y0+h} Z'
        path(d,fill=col,op=0.55)
    line(x0,y0+h,x0+w,y0+h,DK,1.2)
    if rug:
        for i in range(34):
            u=math.exp(random.gauss(math.log(mu),s));
            if u<1: line(x0+u*w,y0+h+3,x0+u*w,y0+h+9,col,1,op=0.7)
IC='#7F8792'
def sigma(x,y,r=12,col=PU):
    circ(x,y,r,'#FFFFFF',col,1.6); t(x,y+r*0.36,'Σ',r*1.0,'bold',col,'middle')
def lock(x,y,s=0.7,col=PU):
    path(f'M {x-5*s} {y} L {x-5*s} {y-5*s} A {5*s} {5*s} 0 0 1 {x+5*s} {y-5*s} L {x+5*s} {y}',stroke=col,sw=1.6)
    rect(x-8*s,y,16*s,12*s,col,rx=2*s); circ(x,y+6*s,1.6*s,'#FFFFFF')
def density(x0,y0,w,h,col,tint,mu=0.35,s=0.45,rug=True,thr=None):
    N=70; xs=[i/N for i in range(N+1)]
    f=lambda u: math.exp(-(math.log(max(u,1e-3)/mu))**2/(2*s*s))/max(u,1e-3)
    mx=max(f(u) for u in xs); pts=[(x0+u*w,y0+h-f(u)/mx*(h-4)) for u in xs]
    path(f'M {x0} {y0+h} '+' '.join(f'L {px:.1f} {py:.1f}' for px,py in pts)+f' L {x0+w} {y0+h} Z',fill=tint)
    path('M '+' L '.join(f'{px:.1f} {py:.1f}' for px,py in pts),stroke=col,sw=1.6)
    if thr is not None:
        tx=x0+thr*w; tp=[(px,py) for px,py in pts if px>=tx]
        path(f'M {tx} {y0+h} '+' '.join(f'L {px:.1f} {py:.1f}' for px,py in tp)+f' L {x0+w} {y0+h} Z',fill=col,op=0.55)
    line(x0,y0+h,x0+w,y0+h,DK,1.1)
    if rug:
        for i in range(26):
            u=math.exp(random.gauss(math.log(mu),s))
            if u<1: line(x0+u*w,y0+h+2.5,x0+u*w,y0+h+7,col,1,op=0.75)
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{H*180/W:.2f}mm" viewBox="0 0 {W} {H}">')
a('<defs>'+''.join(f'<marker id="ah-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.2" markerHeight="4.2" orient="auto-start-reverse"><path d="M0,0.5 L10,5 L0,9.5 z" fill="{c}"/></marker>' for c,k in mk.items())+'</defs>')
rect(0,0,W,H,'#FFFFFF')
# frames + panel labels (8 pt = 16 U)
rect(0,24,150,428,'#FAFAFB',LG,1.1,8); t(2,17,'a',16,'bold'); t(14,17,'Inputs',13,'bold')
rect(162,24,858,428,'#FAFAFB',LG,1.1,8); t(164,17,'b',16,'bold'); t(176,17,'DeepST–INLA',13,'bold')
for x,s in ((172,'TEMPORAL–GRAPH EXPERTS'),(402,'PREDICTIVE PATHS (H1–H3)'),(650,'FUSION &amp; CALIBRATION'),(860,'OUTPUTS')):
    t(x,44,s,10,'bold','#8A919C')
# ===== a
t(10,50,'Dengue cases',12,'bold',SL)
for i,v in enumerate([10,14,12,20,35,60,85,70,40,22,15,12]):
    bh=v/85*32; rect(12+i*10.4,92-bh,7,bh,'#A3AAB4',rx=0.8)
line(10,93,138,93,SL,1); t(12,104,'t−11',10,fill=IC); t(138,104,'t',10,fill=IC,anchor='end')
t(10,128,'Environment',12,'bold',SL)
def i_temp(x,y): rect(x-2,y-6,4,8,'none',IC,1.1,2); circ(x,y+4,2.8,IC)
def i_hum(x,y): path(f'M {x} {y-6} C {x+5} {y} {x+5} {y+5} {x} {y+5} C {x-5} {y+5} {x-5} {y} {x} {y-6} Z',stroke=IC,sw=1.1)
def i_rain(x,y):
    path(f'M {x-5} {y+1} C {x-8} {y+1} {x-8} {y-4} {x-4} {y-4} C {x-3} {y-8} {x+4} {y-8} {x+4} {y-4} C {x+8} {y-4} {x+8} {y+1} {x+4} {y+1} Z',stroke=IC,sw=1.1)
    for dx in (-3,0,3): line(x+dx,y+3,x+dx-1.5,y+6,IC,1.1)
def i_soil(x,y):
    for k in range(3): line(x-6,y+1+k*2.5,x+6,y+1+k*2.5,IC,1.1)
    line(x,y,x,y-5,IC,1.1); path(f'M {x} {y-3} C {x-5} {y-4} {x-5} {y-7} {x-1} {y-6}',stroke=IC,sw=1)
def i_sun(x,y):
    circ(x,y,2.8,'none',IC,1.1)
    for k in range(8):
        an=k*math.pi/4; line(x+4.6*math.cos(an),y+4.6*math.sin(an),x+6.6*math.cos(an),y+6.6*math.sin(an),IC,1.1)
def i_dew(x,y): path(f'M {x-6} {y+4} C {x-3} {y-5} {x+5} {y-6} {x+6} {y-5} C {x+5} {y+2} {x} {y+6} {x-6} {y+4} Z',stroke=IC,sw=1.1); circ(x+1,y,1.4,IC)
def i_et(x,y):
    line(x-6,y+5,x+6,y+5,IC,1.1)
    for dx in (-3,3): path(f'M {x+dx} {y+3} C {x+dx-1.5} {y+1} {x+dx+1.5} {y-1} {x+dx} {y-4}',stroke=IC,sw=1.1); path(f'M {x+dx-1.6} {y-2.5} L {x+dx} {y-5.5} L {x+dx+1.6} {y-2.5}',stroke=IC,sw=1)
for k,(fn,lab) in enumerate([(i_temp,'Temperature'),(i_hum,'Humidity'),(i_rain,'Rainfall'),(i_soil,'Soil wetness'),(i_sun,'Solar radiation'),(i_dew,'Dew point'),(i_et,'ET0')]):
    y=144+k*15; fn(18,y-3.5); t(30,y,lab,11,fill=SL)
t(10,264,'Commune graph',12,'bold',SL)
nodes=[(18,288),(44,276),(66,292),(94,280),(124,290),(30,312),(56,320),(86,308),(114,318),(16,340),(46,348),(76,342),(104,350),(132,334)]
edges=set()
for i,(x,y) in enumerate(nodes):
    ds=sorted(((x-u)**2+(y-v)**2,j) for j,(u,v) in enumerate(nodes) if j!=i)
    for _,j in ds[:3]: edges.add(tuple(sorted((i,j))))
for i,j in edges:
    hl=7 in (i,j); line(*nodes[i],*nodes[j],SL if hl else '#C3C8CF',1.4 if hl else 1)
for i,(x,y) in enumerate(nodes): circ(x,y,4.5 if i==7 else 3.4,SL if i==7 else '#FFFFFF',SL if i==7 else IC,1)
t(10,372,'554 communes; adjacency',10.5,fill=IC); t(10,384,'+ 8 nearest neighbours',10.5,fill=IC)
t(10,418,'12 channels × 12 months',11,'bold',SL); t(10,431,'per commune, issue month',10.5,fill=IC)
arrow([(150,230),(170,230)],GY,1.6)
# ===== b1 experts
rect(180,68,210,350,ORt,OR,1.3,8)
rect(172,60,210,350,'#FFFFFF',BL,1.5,8)
rect(296,52,80,16,BL,rx=8); t(336,63.5,'Base expert',10.5,'bold','#FFFFFF','middle')
rect(296,410,88,16,OR,rx=8); t(340,421.5,'Warning expert',10.5,'bold','#FFFFFF','middle')
t(184,82,'Causal TCN',12,'bold'); t(184,95,'dilation 1, 2, 4',10.5,fill=GY)
xs=[186+i*13 for i in range(12)]; ys=[170,148,126,104]
hl={3:{11}}; dil=[1,2,4]
for L in (3,2,1):
    d=dil[L-1]; nx=set()
    for n in hl[L]:
        for k in (0,1,2):
            m=n-k*d
            if m>=0: nx.add(m); line(xs[n],ys[L],xs[m],ys[L-1],BL,1.1,op=0.85)
    hl[L-1]=nx
for L in range(4):
    for i,x in enumerate(xs):
        on=i in hl.get(L,set()); circ(x,ys[L],3.2,BL if on else '#FFFFFF',BL if on else '#B9BFC7',1)
t(xs[0],184,'t−11',10,fill=GY,anchor='middle'); t(xs[-1],184,'t',10,fill=GY,anchor='middle')
arrow([(257,188),(257,200)],BL,1.6)
cx,cy=257,240
nb=[(cx+34*math.cos(math.radians(v)),cy+32*math.sin(math.radians(v))) for v in (-90,-30,30,90,150,210)]
for (x,y),wv in zip(nb,[3.2,1.1,2.2,1,1.6,4]): line(cx,cy,x,y,BL,wv,op=0.75)
for x,y in nb: circ(x,y,5,'#FFFFFF',SL,1.1)
circ(cx,cy,8.5,BL); t(cx,cy+3.6,'i',10,'bold','#FFFFFF','middle'); t(cx+9,cy-19,'α'+sub('ij',8),10.5,fill=BL)
t(cx,290,'Graph attention',12,'bold',anchor='middle')
def chip(x,y,w,label,icon):
    rect(x,y,w,22,'#F5F6F7',LG,1,11); icon(x+13,y+11); t(x+25,y+15,label,11)
def cal(x,y):
    rect(x-6,y-5,12,10.5,'#FFFFFF',SL,1,1.5); rect(x-6,y-5,12,3,SL)
    for i in range(3):
        for j in range(2): rect(x-4+i*3.4,y-0.5+j*3,2,1.6,SL)
def hor(x,y):
    for i,hh in enumerate((4,7,10)): rect(x-6+i*4.6,y+5-hh,3,hh,BL)
chip(184,302,138,'Seasonal memory',cal); chip(184,330,138,'Horizon embedding',hor)
arrow([(322,313),(350,313)],SL,1.3); arrow([(322,341),(350,341)],SL,1.3)
arrow([(293,240),(350,240)],BL,1.6)
random.seed(5)
for i in range(32):
    v=random.random(); c=int(235-v*150); rect(352,100+i*8,14,8,f'rgb({c-40},{c-10},{min(255,c+25)})','#FFFFFF',0.6)
rect(352,100,14,256,'none',BL,1.2); t(359,92,'h (32-D)',11,'bold',BL,'middle')
t(184,378,LOSS+' = '+LOSS+sub('count')+' + λ '+LOSS+sub('focal'),12,'bold')
rect(184,388,8,8,BL,rx=1.5); t(196,395.5,'base λ = 0',10.5)
rect(260,388,8,8,OR,rx=1.5); t(272,395.5,'warning λ = 0.25',10.5)
t(172,444,'Same architecture; two networks trained independently',10,fill=GY,it=True)
# ===== b2 lanes
for y0,h,tint in ((60,92,'#F2F6FB'),(160,92,'#EFF7F2'),(260,148,'#FCF3EE')):
    rect(400,y0,240,h,tint,rx=7)
def headnode(x,y,col,labs):
    for k,l in enumerate(labs):
        yy=y+k*20-(len(labs)-1)*10; circ(x,yy,8,'#FFFFFF',col,1.3); t(x,yy+3.6,l,10.5,'bold',col,'middle')
t(408,76,'Neural count draws',12,'bold',BL)
arrow([(368,112),(430,112)],BL,1.6); headnode(440,112,BL,['μ','σ']); arrow([(450,112),(528,112)],BL,1.6)
t(489,106,'log(1+y)',10,fill=GY,anchor='middle')
density(532,86,102,50,BL,BLt)
t(408,176,'Structured count draws',12,'bold',GR)
arrow([(368,212),(404,212)],GR,1.6)
for i in range(32): rect(406,192+i*1.25,7,1.25,'#9FD0B9' if i%2 else '#7FBEA0')
rect(406,192,7,40,'none',GR,0.9)
path('M 413 192 L 440 204 L 440 220 L 413 232 Z',fill='#E2F2EA',stroke='#9CCDB5',sw=0.9)
for i in range(4): rect(440,204+i*4,6,4,GR,'#FFFFFF',0.5)
t(426,246,'32 → 4',10,fill=GY,anchor='middle')
arrow([(449,212),(462,212)],GR,1.4)
def hexa(x,y,r): return 'M '+' L '.join(f'{x+r*math.cos(math.radians(60*k+30)):.1f} {y+r*math.sin(math.radians(60*k+30)):.1f}' for k in range(6))+' Z'
for rr in range(4):
    for cc in range(5):
        x=470+cc*10.4+(5.2 if rr%2 else 0); y=197+rr*9
        v=0.5+0.5*math.sin(cc*0.8+0.4)*math.cos(rr*0.9-0.5); g=int(245-v*150)
        path(hexa(x,y,5.8),fill=f'rgb({g-60},{min(255,g+5)},{g-25})',stroke='#FFFFFF',sw=0.7)
t(494,246,'NB–INLA, BYM2',10,fill=GY,anchor='middle')
arrow([(524,212),(530,212)],GR,1.4)
density(532,186,102,50,GR,GRt,mu=0.33,s=0.38)
t(408,276,'Warning count draws',12,'bold',OR)
arrow([(390,312),(430,312)],OR,1.6); headnode(440,312,OR,['μ','σ']); arrow([(450,312),(528,312)],OR,1.6)
t(489,306,'log(1+y)',10,fill=GY,anchor='middle')
density(532,286,102,50,OR,ORt,mu=0.45,s=0.55)
t(408,360,'Direct outbreak signal',12,'bold',OR)
arrow([(390,382),(430,382)],OR,1.6); headnode(440,382,OR,['z']); arrow([(450,382),(528,382)],OR,1.6)
t(489,376,'sigmoid',10,fill=GY,anchor='middle')
rect(532,375,102,14,'#FFFFFF',OR,1.1,7); rect(532,375,64,14,OR,rx=7,op=0.85)
t(538,385,'p'+sub('aux',8),10,'bold','#FFFFFF'); t(532,401,'0',10,fill=GY); t(634,401,'1',10,fill=GY,anchor='end')
# ===== b3 fusion
arrow([(636,112),(670,112),(670,198)],BL,1.6)
arrow([(636,212),(656,212)],GR,1.6)
arrow([(636,312),(670,312),(670,226)],OR,1.6)
sigma(670,212); lock(690,190)
for x0,w,c in ((684,8,BL),(692,14,GR),(706,6,OR)): rect(x0,240,w,4,c)
t(684,256,'draw mixture',10,fill=GY)
arrow([(682,212),(694,212)],PU,1.6)
t(696,150,'DeepST–INLA',12.5,'bold',PU); t(696,163,'predictive distribution',10.5,fill=GY)
density(696,172,124,58,PU,PUt,mu=0.36,s=0.5,rug=False,thr=0.62)
tx=696+0.62*124; line(tx,168,tx,232,DK,1.1,'3 2.5'); t(tx+3,178,'T'+sub('c,m',8),10.5,'bold')
arrow([(822,206),(838,206),(838,112),(858,112)],PU,1.6)
# output 1
t(860,66,'Incidence forecast',12.5,'bold',PU)
rect(860,72,152,90,'#FFFFFF',PU,1.3,7)
hist=[30,34,31,45,60,82,96,88,70,55]
X=lambda i:868+i*9.6; Y=lambda v:148-v*0.68
path('M '+' L '.join(f'{X(i):.1f} {Y(v):.1f}' for i,v in enumerate(hist)),stroke=DK,sw=1.3)
med=[48,44,42]; lo95=[28,20,15]; hi95=[74,86,98]; lo50=[40,34,30]; hi50=[58,58,58]
fx=[X(9)]+[X(9)+15*(k+1) for k in range(3)]
for lo,hi,op in ((lo95,hi95,0.18),(lo50,hi50,0.32)):
    up=[Y(55)]+[Y(v) for v in hi]; dn=[Y(55)]+[Y(v) for v in lo]
    path('M '+' L '.join(f'{fx[k]:.1f} {up[k]:.1f}' for k in range(4))+' L '+' L '.join(f'{fx[k]:.1f} {dn[k]:.1f}' for k in (3,2,1,0))+' Z',fill=PU,op=op)
path(f'M {fx[0]} {Y(55)} '+' '.join(f'L {fx[k+1]:.1f} {Y(med[k]):.1f}' for k in range(3)),stroke=PU,sw=1.4)
for k in range(3): circ(fx[k+1],Y(med[k]),2.2,PU); t(fx[k+1],158,f'H{k+1}',10,'bold',PU,'middle')
line(X(9)+5,78,X(9)+5,150,LG,1,'2.5 2')
t(860,176,'Scored by CRPS',10.5,fill=GY)
line(936,188,936,282,'#B3B8C0',1.1,'4 3'); rect(884,222,104,26,'#FAFAFB')
t(936,233,'One model,',10,fill=GY,anchor='middle',it=True); t(936,245,'two evaluation targets',10,fill=GY,anchor='middle',it=True)
# alert chain
arrow([(780,232),(780,342)],PU,1.6); t(774,292,'p'+sub('count',8)+' = P(y &gt; T)',10.5,fill=PU,anchor='end')
arrow([(636,382),(780,382),(780,370)],OR,1.6); t(708,376,'p'+sub('aux',8),11,'bold',OR,'middle')
sigma(780,356); lock(762,333)
t(708,402,'w'+sub('c',8)+'p'+sub('count',8)+' + w'+sub('a',8)+'p'+sub('aux',8),10,fill=GY,anchor='middle')
arrow([(792,356),(800,356)],PU,1.4)
rect(801,322,46,66,'#FFFFFF',LG,1,5)
line(807,380,843,380,DK,0.9); line(807,380,807,328,DK,0.9)
pts=[(807+u*36,380-46/(1+math.exp(-(u-0.5)*10))) for u in [i/30 for i in range(31)]]
path('M '+' L '.join(f'{x:.1f} {y:.1f}' for x,y in pts),stroke=PU,sw=1.4)
line(807,356,843,356,OR,1.1,'3 2'); t(809,352,'τ'+sub('h',8),10,'bold',OR)
t(824,400,'Platt + τ'+sub('h',8),10,fill=GY,anchor='middle'); lock(841,314)
arrow([(848,356),(858,356)],PU,1.4)
# output 2
t(860,298,'Outbreak alert',12.5,'bold',PU)
rect(860,304,152,106,'#FFFFFF',PU,1.3,7)
alerts={3,8,13,19,24}; k=0
for rr in range(4):
    for cc in range(7):
        x=872+cc*12.5+(6.2 if rr%2 else 0); y=318+rr*12.5
        on=k in alerts; circ(x,y,4,OR if on else '#E3E5E8',OR if on else '#C3C7CD',0.9); k+=1
bx,by=988,340; s=0.75
path(f'M {bx-13*s} {by+9*s} C {bx-11*s} {by+4*s} {bx-11*s} {by-4*s} {bx-10*s} {by-7*s} C {bx-8*s} {by-16*s} {bx+8*s} {by-16*s} {bx+10*s} {by-7*s} C {bx+11*s} {by-4*s} {bx+11*s} {by+4*s} {bx+13*s} {by+9*s} Z',fill=OR)
circ(bx,by+13*s,3*s,OR)
for sg in (1,-1): path(f'M {bx+sg*16*s} {by-13*s} C {bx+sg*20*s} {by-8*s} {bx+sg*20*s} {by-2*s} {bx+sg*17*s} {by+2*s}',stroke=OR,sw=1.2)
t(870,400,'alert if p ≥ τ'+sub('h',8),11)
t(860,424,'Scored by Brier, sensitivity,',10.5,fill=GY); t(860,437,'specificity and F1',10.5,fill=GY)
# legend
ly=476; x=4
sigma(x+10,ly-4,10); t(x+25,ly,'weighted combination',11); x+=146
lock(x+6,ly-8); t(x+18,ly,'selected on 2019–2020 validation, frozen for 2021–2025 test',11); x+=340
for c,lab in ((BL,'base expert'),(GR,'structured INLA'),(OR,'warning expert'),(PU,'DeepST–INLA hybrid')):
    rect(x,ly-8.5,9,9,c,rx=2); t(x+14,ly,lab,11); x+=len(lab)*5.6+34
a('</svg>')
svg = '\n'.join(x for x in o if x)
(OUT / 'Fig2_DeepST-INLA_framework.svg').write_text(svg)
print('wrote', OUT / 'Fig2_DeepST-INLA_framework.svg')
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        try:
            br = pw.chromium.launch()
        except Exception:
            br = pw.chromium.launch(executablePath='/opt/pw-browsers/chromium')
        pg = br.new_page(viewport={'width': 680, 'height': 327}, device_scale_factor=6.25)
        pg.goto((OUT / 'Fig2_DeepST-INLA_framework.svg').as_uri())
        pg.screenshot(path=str(OUT / 'Fig2_DeepST-INLA_framework_preview.png'))
        q = br.new_page()
        q.set_content('<html><body style="margin:0">' + svg.replace('width="180mm"', 'width="180mm" style="display:block"') + '</body></html>')
        q.pdf(path=str(OUT / 'Fig2_DeepST-INLA_framework.pdf'), width='180mm', height=f'{H * 180 / W:.2f}mm',
              print_background=True, page_ranges='1')
        br.close()
    print('wrote PDF + PNG preview')
except Exception as e:  # PDF/PNG export is optional
    print('SVG written; export PDF with Inkscape/Chromium instead:', e)
