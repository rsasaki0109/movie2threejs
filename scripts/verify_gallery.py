"""Verify each packaged room, deterministic physics, walking and featured controls."""
import argparse
import functools
import hashlib
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]


def verify(site: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if not self.path.startswith('/movie2threejs/'):
                self.send_error(404);return
            self.path=self.path.removeprefix('/movie2threejs')
            super().do_GET()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(site.resolve())))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=f'http://127.0.0.1:{server.server_port}/movie2threejs/'
    demos=json.loads((site/'demos.json').read_text())
    reports=[];errors=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(args=['--enable-gpu','--use-gl=angle','--use-angle=d3d11'])
            gallery=browser.new_page(viewport={'width':1280,'height':900})
            gallery.on('pageerror',lambda e:errors.append(str(e)))
            gallery.goto(url);gallery.wait_for_function(f"document.querySelectorAll('.card').length === {len(demos)}")
            gallery.wait_for_function('Array.from(document.images).every(i=>i.complete && i.naturalWidth>0)')
            gallery.screenshot(path=str(out/'gallery-desktop.png'),full_page=True)
            gallery.set_viewport_size({'width':390,'height':844})
            assert not gallery.evaluate('document.documentElement.scrollWidth > innerWidth')
            gallery.screenshot(path=str(out/'gallery-mobile.png'),full_page=True);gallery.close()
            for demo in demos:
                assets=site/'demos'/demo['id'];world=json.loads((assets/'world.json').read_text())
                manifest=json.loads((assets/'manifest.json').read_text())
                assert len(world['objects'])==demo['objects']==manifest['objects']
                for file in manifest['files']:
                    data=(assets/file['path']).read_bytes()
                    assert len(data)==file['bytes'] and hashlib.sha256(data).hexdigest()==file['sha256']
                shot=json.loads((assets/'hero-shot.json').read_text())
                event=next(e for e in shot['events'] if e['type']=='push')
                subject=world['demo'].get('subject_id',6)
                frames=[0,int(event['time']*shot['fps'])-1,round(shot['duration']*shot['fps'])-1]
                runs=[]
                for repeat in range(2):
                    page=browser.new_page(viewport={'width':960,'height':540})
                    page.on('pageerror',lambda e:errors.append(str(e)))
                    page.goto(url+f"demos/{demo['id']}/?record=hero-shot.json")
                    page.wait_for_function('window.playworld?.ready || window.playworldError',timeout=180000)
                    assert page.evaluate('window.playworldError') is None
                    states=[page.evaluate('n=>playworld.record.nextFrame(n)',f) for f in frames]
                    if repeat==0:page.locator('canvas').screenshot(path=str(out/(demo['id']+'-physics.png')))
                    runs.append(states);page.close()
                assert runs[0]==runs[1],demo['id']+' replay differs'
                initial,before,final=runs[0]
                assert any(a.get('object')==subject for a in final['actions']),demo['id']+' push missed'
                position=lambda state:next(o['position'] for o in state['objects'] if o['id']==subject)
                distance=lambda a,b:sum((a[k]-b[k])**2 for k in 'xyz')**.5
                moved=distance(position(before),position(final))
                assert moved>.2,demo['id']+' object did not move visibly'
                walk=distance(initial['player'],final['player'])
                assert walk>.1,demo['id']+' character did not walk'
                assert len(final['balls'])==1
                # Use real pointer-lock activation and the public featured action.
                page=browser.new_page(viewport={'width':960,'height':540})
                page.on('pageerror',lambda e:errors.append(str(e)))
                page.goto(url+f"demos/{demo['id']}/")
                page.wait_for_function('window.playworld?.ready || window.playworldError',timeout=180000)
                assert page.evaluate('window.playworldError') is None
                page.locator('#featured-push').click()
                assert page.locator('#featured-push').inner_text()=='Pushed!'
                page.locator('#reset-world').click()
                page.locator('#play').click();page.wait_for_function('playworld.controls.isLocked')
                start=page.evaluate('playworld.camera.position.toArray()')
                walk_key=next(e['keys'][0] for e in shot['events'] if e['type']=='walk' and e['keys'])
                page.keyboard.down(walk_key);page.wait_for_timeout(500);page.keyboard.up(walk_key)
                end=page.evaluate('playworld.camera.position.toArray()')
                interactive_walk=sum((a-b)**2 for a,b in zip(start,end))**.5
                assert interactive_walk>.1,demo['id']+' interactive walking failed'
                page.keyboard.press('Escape');page.close()
                reports.append({'id':demo['id'],'objects':len(world['objects']),'colliders':len(world['colliders']),
                    'deterministic_replay':True,'subject_id':subject,'movement_after_push_m':moved,
                    'initial_settling_m':distance(position(initial),position(before)),
                    'scripted_character_movement_m':walk,'interactive_character_movement_m':interactive_walk,
                    'balls':1,'featured_button':True,'world_sha256':hashlib.sha256((assets/'world.json').read_bytes()).hexdigest(),
                    'shot_sha256':hashlib.sha256((assets/'hero-shot.json').read_bytes()).hexdigest()})
                print('Verified:',demo['id'],f'{moved:.3f} m physical movement',flush=True)
            browser.close()
        assert not errors,errors
        result={'scope':'Local Chromium under a GitHub Pages-style path; no public deployment',
                'demos':reports,'gallery_mobile_overflow':False,'browser_errors':errors}
        (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:server.shutdown();server.server_close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',type=Path,default=ROOT/'_site');p.add_argument('--out',type=Path,default=ROOT/'.cache/gallery-verification')
    a=p.parse_args();verify(a.site,a.out)
