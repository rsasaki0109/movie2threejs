"""Make small README interaction previews from the verified real-room hero MP4."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from encode_hero import encode_with_gifski

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [('explore', 2.0, 3.8), ('push', 5.5, 2.8), ('throw', 8.3, 2.7)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gifski',type=Path)
    args=parser.parse_args()
    executable=args.gifski or shutil.which('gifski')
    source = ROOT/'docs/hero.mp4'
    out = ROOT/'docs/demos'
    out.mkdir(exist_ok=True)
    receipts=[]
    for name,start,duration in CLIPS:
        gif=out/f'{name}.gif'
        if executable:
            cache=ROOT/'.cache/interactions'; cache.mkdir(parents=True,exist_ok=True)
            clip=cache/f'{name}.mp4'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(source),
                            '-t',str(duration),'-an','-c:v','libx264','-crf','15','-pix_fmt','yuv420p',str(clip)],check=True)
            fps,colors=15,None
            for quality in [80,65,50]:
                encode_with_gifski(executable,clip,gif,fps,quality,quality,quality,width=480)
                if gif.stat().st_size<=2_000_000: break
        else:
            quality=None
            for fps, colors in [(12,128),(10,96),(10,64),(8,64)]:
                filters=f'fps={fps},scale=480:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle'
                subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',str(start),'-i',str(source),
                                '-t',str(duration),'-filter_complex',filters,'-loop','0',str(gif)],check=True)
                if gif.stat().st_size<=2_000_000: break
        if gif.stat().st_size>2_000_000:
            raise ValueError(f'{name} preview exceeds 2 MB')
        measured=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=width,height:format=duration','-of','json',str(gif)],text=True))
        stream=measured['streams'][0]
        receipts.append({'name':name,'source_start_seconds':start,'requested_duration_seconds':duration,
                         'measured_duration_seconds':float(measured['format']['duration']),
                         'width':stream['width'],'height':stream['height'],
                         'fps':fps,'colors':colors,'quality':quality,'encoder':'gifski' if executable else 'ffmpeg',
                         'gif_bytes':gif.stat().st_size,'gif_sha256':hashlib.sha256(gif.read_bytes()).hexdigest()})
    provenance={'source':'docs/hero.mp4','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'scope':'Three interaction excerpts from the same measured apartment workbench, not three different reconstructions',
                'width':480,'clips':receipts}
    (out/'interactions-provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()
