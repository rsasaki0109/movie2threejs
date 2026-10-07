"""Encode a complete, verified browser capture as a small looping room preview."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from encode_hero import encode_with_gifski


def encode(frames: Path, out: Path, name: str, title: str, gifski: Path | None):
    capture=json.loads((frames/'capture.json').read_text())
    if capture['captured_frames']!=capture['frames']:
        raise ValueError('Gallery preview requires a complete capture')
    if not re.fullmatch(r'[a-z0-9-]+',name) or not re.fullmatch(r'[A-Za-z0-9 ]+',title):
        raise ValueError('Use a simple demo ID and title')
    out.mkdir(parents=True,exist_ok=True)
    duration=capture['frames']/capture['fps']
    mp4,gif=out/f'{name}.mp4',out/f'{name}.gif'
    # Dissolve back to the original room at the loop boundary, including physics.
    filters=(f'fps={capture["fps"]},settb=AVTB,split[body][first];'
        f'[first]trim=end_frame=1,loop=loop=-1:size=1:start=0,setpts=N/({capture["fps"]}*TB),'
        f'trim=duration={duration},format=rgba,fade=t=in:st={duration-.4:.6f}:d=0.33:alpha=1[still];'
        '[body][still]overlay=shortest=1:format=rgb,format=yuv420p[out]')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate',str(capture['fps']),
        '-i',str(frames/'frame_%05d.png'),'-filter_complex',filters,'-map','[out]',
        '-an','-c:v','libx264','-crf','17','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',str(mp4)],check=True)
    executable=gifski or shutil.which('gifski')
    if not executable: raise RuntimeError('Install gifski or pass --gifski')
    for fps,quality in [(15,80),(15,65),(15,50),(10,50),(10,35)]:
        fps=min(fps,capture['fps'])
        encode_with_gifski(executable,mp4,gif,fps,quality,quality,quality,width=480)
        if gif.stat().st_size<=2_000_000:break
    if gif.stat().st_size>2_000_000:raise ValueError('Preview exceeds 2 MB')
    properties=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=width,height:format=duration','-of','json',str(gif)],text=True))
    report={'id':name,'title':title,'capture_world_sha256':capture['world_sha256'],
        'shot_sha256':capture['shot_sha256'],'captured_frames':capture['frames'],
        'source':'Actual fixed-step browser viewer output; no generated imagery',
        'gif_duration_seconds':float(properties['format']['duration']),'gif_fps':fps,'quality':quality,
        'gif_width':properties['streams'][0]['width'],'gif_height':properties['streams'][0]['height'],
        'gif_bytes':gif.stat().st_size,'mp4_bytes':mp4.stat().st_size,
        'gif_sha256':hashlib.sha256(gif.read_bytes()).hexdigest(),'mp4_sha256':hashlib.sha256(mp4.read_bytes()).hexdigest()}
    (out/f'{name}-provenance.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('docs/demos'))
    p.add_argument('--name',required=True);p.add_argument('--title',required=True);p.add_argument('--gifski',type=Path)
    a=p.parse_args();encode(a.frames,a.out,a.name,a.title,a.gifski)
