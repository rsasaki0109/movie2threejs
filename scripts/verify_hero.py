"""Check exported hero dimensions, size, provenance and recorded physical actions."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from PIL import Image


def distance(a, b, axes='xyz'):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in axes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, default=Path('recordings/hero/capture.json'))
    parser.add_argument('--assets', type=Path, default=Path('docs'))
    parser.add_argument('--repeat-a', type=Path, required=True)
    parser.add_argument('--repeat-b', type=Path, required=True)
    args = parser.parse_args()
    capture = json.loads(args.capture.read_text(encoding='utf-8'))
    snapshots = capture['snapshots']
    assert capture['captured_frames'] == capture['frames'] == len(snapshots)
    assert (capture['width'], capture['height']) == (1920, 1080)
    assert all(s['tick'] == i * (60 // capture['fps']) for i, s in enumerate(snapshots))
    pushes = [a for a in snapshots[-1]['actions'] if a['type'] == 'push']
    assert pushes and pushes[0]['object'] is not None, 'push ray missed'
    push = pushes[0]
    before = max((s for s in snapshots if s['time'] < push['time']), key=lambda s: s['time'])
    get_object = lambda s: next(o for o in s['objects'] if o['id'] == push['object'])
    a, b = get_object(before), get_object(snapshots[-1])
    displacement = distance(a['position'], b['position'])
    drop = a['position']['y'] - b['position']['y']
    q = b['rotation']
    up_dot = 1 - 2 * (q['x'] ** 2 + q['z'] ** 2)
    assert displacement > .2 and up_dot < .5, 'pushed object did not visibly tip'
    walked = distance(snapshots[0]['player'], snapshots[-1]['player'], 'xz')
    assert walked > .1 and len(snapshots[-1]['balls']) == 1
    repeats = [json.loads((p/'snapshots.json').read_text(encoding='utf-8')) for p in [args.repeat_a,args.repeat_b]]
    assert repeats[0] == repeats[1]
    assert all(s == snapshots[s['frame']] for s in repeats[0]), 'capture and sampled replay differ'
    for frame in [repeats[0][0]['frame'],repeats[0][-1]['frame']]:
        name = f'frame_{frame:05d}.png'
        assert (args.repeat_a/name).read_bytes() == (args.repeat_b/name).read_bytes()
    gif, mp4 = args.assets/'hero.gif', args.assets/'hero.mp4'
    with Image.open(gif) as im:
        assert im.width == 960 and im.info.get('loop') == 0
        duration = 0
        for frame in range(im.n_frames):
            im.seek(frame); duration += im.info.get('duration',0)/1000
    assert 10 <= duration <= 15 and gif.stat().st_size <= 8_000_000
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height:format=duration','-of','json',str(mp4)]))
    assert (probe['streams'][0]['width'],probe['streams'][0]['height']) == (1920,1080)
    provenance = json.loads((args.assets/'hero-provenance.json').read_text(encoding='utf-8'))
    source = json.loads((args.assets/'hero-source.json').read_text(encoding='utf-8'))
    assert provenance['source_video_sha256'] == source['sha256']
    assert provenance['capture_world_sha256'] == capture['world_sha256']
    assert provenance['shot_sha256'] == capture['shot_sha256']
    report = {'scope':'real public Eyeful Tower workbench; cinematic camera plus Rapier actions',
        'capture_renderer':capture['renderer'],'capture_seconds':capture['seconds'],
        'capture_frames':len(snapshots),'capture_resolution':[1920,1080],
        'fps':capture['fps'],'physics_hz':60,'push_object':push['object'],
        'pushed_object_displacement_m':displacement,'pushed_object_drop_m':drop,
        'pushed_object_final_up_dot':up_dot,'player_displacement_m':walked,'thrown_balls':1,
        'equal_physics_samples':len(repeats[0]),'repeat_first_and_last_pixels_equal':True,
        'gif_bytes':gif.stat().st_size,'gif_duration_seconds':round(duration,3),
        'mp4_bytes':mp4.stat().st_size,'mp4_duration_seconds':float(probe['format']['duration']),
        'gif_sha256':hashlib.sha256(gif.read_bytes()).hexdigest(),
        'mp4_sha256':hashlib.sha256(mp4.read_bytes()).hexdigest()}
    (args.assets/'hero-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
