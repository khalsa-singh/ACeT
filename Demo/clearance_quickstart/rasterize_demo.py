"""Pillow fallback renderer for the simple SVG primitives emitted by run_demo.py."""
from __future__ import annotations
import os,re,xml.etree.ElementTree as ET
from pathlib import Path

def render(svg_path: Path, scale: int = 3):
    from PIL import Image, ImageDraw, ImageFont
    root=ET.parse(svg_path).getroot();width=int(root.get('width'));height=int(root.get('height'))
    image=Image.new('RGBA',(width*scale,height*scale),'white');draw=ImageDraw.Draw(image)
    def font(size,bold=False):
        candidates=[('Arial Bold.ttf' if bold else 'Arial.ttf'),('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')]
        win=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
        candidates += [str(win/('arialbd.ttf' if bold else 'arial.ttf'))]
        for candidate in candidates:
            try:return ImageFont.truetype(candidate,round(size*scale))
            except OSError:pass
        return ImageFont.load_default(size=round(size*scale))
    def num(el,k,d=0):return float(el.get(k,str(d)))*scale
    for el in root:
        tag=el.tag.rsplit('}',1)[-1]
        fill=el.get('fill');stroke=el.get('stroke');linew=max(1,round(float(el.get('stroke-width','1'))*scale))
        if tag=='rect':
            x=num(el,'x');y=num(el,'y');w=width*scale if el.get('width')=='100%' else num(el,'width');h=height*scale if el.get('height')=='100%' else num(el,'height')
            draw.rectangle((x,y,x+w,y+h),fill=None if fill in ('none',None) else fill,outline=stroke,width=linew)
        elif tag=='circle':
            x=num(el,'cx');y=num(el,'cy');r=num(el,'r');draw.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=stroke,width=linew)
        elif tag=='line':draw.line((num(el,'x1'),num(el,'y1'),num(el,'x2'),num(el,'y2')),fill=stroke or '#222',width=linew)
        elif tag=='path':
            numbers=list(map(float,re.findall(r'-?\d+(?:\.\d+)?',el.get('d',''))));pts=[(numbers[i]*scale,numbers[i+1]*scale) for i in range(0,len(numbers),2)]
            draw.line(pts,fill=stroke or '#222',width=linew)
        elif tag=='text':
            cls=el.get('class','axis');size={'title':20,'axis':13,'small':12,'metric':14}.get(cls,13)
            f=font(size,cls=='title');anchor={'middle':'ms','end':'rs'}.get(el.get('text-anchor'),'ls')
            x=num(el,'x');y=num(el,'y');text=''.join(el.itertext());transform=el.get('transform','')
            if transform.startswith('rotate('):
                a,cx,cy=list(map(float,re.findall(r'-?\d+(?:\.\d+)?',transform)))
                # Pad before rotating: text centered on a left-edge baseline would
                # otherwise be clipped before its vertical orientation is applied.
                pad=round(draw.textlength(text,font=f))+round(size*scale*2)
                layer=Image.new('RGBA',(image.width+2*pad,image.height+2*pad))
                ImageDraw.Draw(layer).text((x+pad,y+pad),text,font=f,fill=el.get('fill','#111'),anchor=anchor)
                layer=layer.rotate(-a,center=(cx*scale+pad,cy*scale+pad),resample=Image.Resampling.BICUBIC)
                image.alpha_composite(layer.crop((pad,pad,pad+image.width,pad+image.height)));draw=ImageDraw.Draw(image)
            else:draw.text((x,y),text,font=f,fill=el.get('fill','#111'),anchor=anchor)
    return image.convert('RGB')
