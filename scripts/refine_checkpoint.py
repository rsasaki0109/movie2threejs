"""Warm-start a saved gsplat checkpoint with fixed cameras and Gaussian count.

This creates fresh Adam optimizers: it is NOT an exact resume of the original
trainer. Uses the same filename-sorted holdout split as the pinned trainer.
Random full-resolution crops reduce memory without changing camera calibration.
No paid runtime is allocated by this script.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from playworld.colmap_io import read_model


def load_view(scene, image, camera, max_size):
    if camera.model != 'PINHOLE':
        raise ValueError('refinement requires already undistorted PINHOLE images')
    path = (Path(scene)/'images'/image.name).resolve()
    if not path.is_relative_to((Path(scene)/'images').resolve()):
        raise ValueError('image path escaped scene')
    with Image.open(path) as source:
        source = source.convert('RGB')
        w, h = source.size
        ratio = min(1., max_size/max(w, h))
        width, height = max(1, round(w*ratio)), max(1, round(h*ratio))
        if min(width, height) < 11:
            raise ValueError('resized photograph is smaller than the SSIM window')
        pixels = np.asarray(source.resize((width, height), Image.Resampling.LANCZOS)).copy()
    K = camera.K.copy()
    K[0] *= width/camera.width
    K[1] *= height/camera.height
    viewmat = np.eye(4)
    viewmat[:3, :3], viewmat[:3, 3] = image.R, image.tvec
    return pixels, K, viewmat


def crop_view(pixels, K, size, rng):
    height, width = pixels.shape[:2]
    cw, ch = min(size, width), min(size, height)
    x, y = int(rng.integers(width-cw+1)), int(rng.integers(height-ch+1))
    cropped = K.copy()
    cropped[0, 2] -= x
    cropped[1, 2] -= y
    return pixels[y:y+ch, x:x+cw].copy(), cropped


def depth_samples(xyz, rgb, pixels, K, viewmat, limit=4096, seed=0):
    """Conservative SfM depth anchors: frontmost points with matching photo color.

    Visibility is only a sparse point z-buffer, not a ground-truth depth map.
    Color gating rejects some occluded tracks but cannot detect identical-color
    occluders. This prior is experimental and evaluated on heldout photographs.
    """
    from playworld.objects import View, visible, project
    view=View(K,viewmat[:3,:3],viewmat[:3,3],pixels.shape[1],pixels.shape[0])
    vis,u,v=visible(xyz,view)
    _,_,z=project(xyz,view)
    ids=np.flatnonzero(vis)
    agreement=np.mean(np.abs(rgb[ids].astype(float)-pixels[v[ids],u[ids]]),axis=1)/255
    ids=ids[agreement < .12]
    if len(ids)>limit:
        ids=np.sort(np.random.default_rng(seed).choice(ids,limit,replace=False))
    return np.column_stack([u[ids],v[ids]]),z[ids]


def ssim(x, y):
    """Mean SSIM with separable 11-pixel Gaussian, sigma 1.5, valid boundary."""
    import torch
    import torch.nn.functional as F
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-axis**2/(2*1.5**2))
    kernel /= kernel.sum()
    moments = torch.cat([x, y, x*x, y*y, x*y], dim=1)
    channels = moments.shape[1]
    moments = F.conv2d(moments, kernel.view(1,1,1,11).expand(channels,1,1,11), groups=channels)
    moments = F.conv2d(moments, kernel.view(1,1,11,1).expand(channels,1,11,1), groups=channels)
    mx, my, ex, ey, exy = moments.chunk(5, dim=1)
    vx, vy, cov = (ex-mx*mx).clamp_min(0), (ey-my*my).clamp_min(0), exy-mx*my
    return (((2*mx*my+.01**2)*(2*cov+.03**2)) /
            ((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()


def refine(args):
    import torch
    import torch.nn.functional as F
    import gsplat
    from gsplat import export_splats, rasterization
    if not torch.cuda.is_available():
        raise RuntimeError('a local CUDA GPU is required')
    if args.steps < 1 or args.eval_every < 1 or args.patch_size < 11 or args.max_size < 11 or args.test_every < 2:
        raise ValueError('invalid step, crop, image-size or holdout settings')
    if any(not math.isfinite(v) or v < 0 for v in [args.scale_reg,args.opacity_reg,args.depth_weight]) or not math.isfinite(args.means_lr) or args.means_lr <= 0:
        raise ValueError('regularization must be finite and nonnegative')
    if args.out.exists():
        raise FileExistsError(args.out)
    rec = read_model(args.scene/'sparse')
    images = rec.sorted_images()
    train_ids = [i for i in range(len(images)) if i % args.test_every]
    val_ids = [i for i in range(len(images)) if not i % args.test_every]
    if not train_ids or not val_ids:
        raise ValueError('both training and heldout photographs are required')
    data = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    tensors = data['splats']
    n = len(tensors['means'])
    shapes = {'means':(n,3), 'quats':(n,4), 'scales':(n,3), 'opacities':(n,),
              'sh0':(n,1,3), 'shN':(n,15,3)}
    for key, shape in shapes.items():
        if tuple(tensors[key].shape) != shape or not torch.isfinite(tensors[key]).all():
            raise ValueError(f'invalid SH3 checkpoint tensor: {key}')
    args.out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    params = torch.nn.ParameterDict({key:torch.nn.Parameter(tensors[key].float().to('cuda')) for key in shapes})
    del tensors, data
    rates = {'means':args.means_lr, 'scales':1e-3, 'quats':5e-4,
             'opacities':1e-2, 'sh0':1e-3, 'shN':1e-4}
    optimizers = {key:torch.optim.Adam([param], lr=rates[key], eps=1e-15, foreach=False)
                  for key,param in params.items()}
    views = [load_view(args.scene, im, rec.cameras[im.camera_id], args.max_size) for im in images]
    priors = {i:depth_samples(rec.xyz,rec.rgb,*views[i],seed=args.seed+i)
              for i in train_ids} if args.depth_weight else {}
    report = {'format':'playworld-checkpoint-refinement/1','status':'running',
              'source_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'source_step':int(torch.load(args.checkpoint,map_location='cpu',weights_only=True)['step']),
              'warm_start_not_exact_resume':True,'optimizer_state':'fresh Adam','gaussians':n,
              'camera_optimization':False,'densification':False,'seed':args.seed,
              'train_images':[images[i].name for i in train_ids],
              'validation_images':[images[i].name for i in val_ids],
              'settings':{key:str(v) if isinstance(v,Path) else v for key,v in vars(args).items()},
              'learning_rates':rates,'gpu':torch.cuda.get_device_name(0),
              'torch':torch.__version__,'gsplat':gsplat.__version__,'evaluations':[]}
    report['depth_prior_training_samples']={images[i].name:len(values[0]) for i,values in priors.items()}
    report['source_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save_report():
        (args.out/'refinement.json').write_text(json.dumps(report,indent=2)+'\n')
    def render(pixels, K, viewmat):
        target = torch.from_numpy(pixels).to('cuda').float().div(255).unsqueeze(0)
        colors, _, _ = rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),
            opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],dim=1),
            viewmats=torch.tensor(viewmat,device='cuda',dtype=torch.float32).unsqueeze(0),
            Ks=torch.tensor(K,device='cuda',dtype=torch.float32).unsqueeze(0),
            width=pixels.shape[1],height=pixels.shape[0],sh_degree=3,packed=True,
            render_mode='RGB+ED' if args.depth_weight else 'RGB')
        return colors[...,:3], target, colors[...,3] if args.depth_weight else None
    def evaluate(step):
        metrics = []
        with torch.no_grad():
            for i in val_ids:
                colors, target, _ = render(*views[i])
                colors = colors.clamp(0,1)
                mse = float(F.mse_loss(colors,target))
                score = float(ssim(colors.permute(0,3,1,2),target.permute(0,3,1,2)))
                metrics.append({'image':images[i].name,'psnr':-10*math.log10(max(mse,1e-12)),'ssim':score})
                if step == 0 or i in val_ids[::3]:
                    pair = torch.cat([target,colors],dim=2)[0].cpu().numpy()
                    Image.fromarray((pair*255).astype(np.uint8)).save(args.out/f'val-{step:05d}-{i:03d}.jpg',quality=95)
        result = {'added_steps':step, 'psnr':float(np.mean([m['psnr'] for m in metrics])),
                  'ssim':float(np.mean([m['ssim'] for m in metrics])),'per_image':metrics}
        report['evaluations'].append(result)
        save_report()
        print('eval', json.dumps({k:v for k,v in result.items() if k!='per_image'}),flush=True)
        return result
    def save_model(step, name):
        checkpoint = {'step':report['source_step']+step,'added_steps':step,'warm_start':True,
                      'splats':{key:value.detach().cpu() for key,value in params.items()}}
        torch.save(checkpoint,args.out/f'{name}.pt')
        export_splats(**{key:value.detach() for key,value in params.items()},format='ply',save_to=str(args.out/f'{name}.ply'))
    started = time.perf_counter()
    try:
        baseline = evaluate(0)
        if args.evaluate_only:
            report.update(status='evaluated',completed_added_steps=0,best_added_steps=0,
                          best_psnr=baseline['psnr'],seconds=time.perf_counter()-started,
                          peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
            return report
        best = baseline['psnr']
        best_step = 0
        training_started = time.perf_counter()
        for step in range(1,args.steps+1):
            i = int(rng.choice(train_ids))
            pixels,original_K,V = views[i]
            pixels,K = crop_view(pixels,original_K,args.patch_size,rng)
            for optimizer in optimizers.values():optimizer.zero_grad(set_to_none=True)
            colors,target,depth = render(pixels,K,V)
            l1 = F.l1_loss(colors,target)
            similarity = ssim(colors.permute(0,3,1,2),target.permute(0,3,1,2))
            loss = .8*l1+.2*(1-similarity)
            depth_loss = None
            if args.depth_weight:
                uv,z = priors[i]
                uv=uv-np.rint([original_K[0,2]-K[0,2],original_K[1,2]-K[1,2]]).astype(int)
                inside=(uv[:,0]>=0)&(uv[:,0]<pixels.shape[1])&(uv[:,1]>=0)&(uv[:,1]<pixels.shape[0])
                if inside.any():
                    coords=torch.tensor(uv[inside],device='cuda',dtype=torch.long)
                    expected=torch.tensor(z[inside],device='cuda',dtype=torch.float32)
                    predicted=depth[0,coords[:,1],coords[:,0]]
                    # Relative depth penalizes a near-camera floating surface
                    # without allowing far outdoor tracks to dominate the loss.
                    depth_loss=F.smooth_l1_loss(predicted/expected,torch.ones_like(expected),beta=.1)
                    loss += args.depth_weight*depth_loss
            if args.scale_reg:loss += args.scale_reg*params['scales'].exp().mean()
            if args.opacity_reg:loss += args.opacity_reg*params['opacities'].sigmoid().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite refinement loss')
            loss.backward()
            for optimizer in optimizers.values():optimizer.step()
            if step == 1 or step % 50 == 0:
                print(json.dumps({'step':step,'loss':float(loss.detach()),'elapsed_seconds':time.perf_counter()-training_started,
                                  'depth_loss':float(depth_loss.detach()) if depth_loss is not None else None,
                                  'peak_memory_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                result = evaluate(step)
                if result['psnr'] > best:
                    best,best_step = result['psnr'],step
                    save_model(step,'best')
                save_model(step,'latest')
        torch.cuda.synchronize()
        report.update(status='complete',completed_added_steps=args.steps,best_added_steps=best_step,
                      best_psnr=best,seconds=time.perf_counter()-started,
                      peak_memory_mib=torch.cuda.max_memory_allocated()/2**20)
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',seconds=time.perf_counter()-started)
        raise
    finally:
        save_report()
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--eval-every',type=int,default=250)
    p.add_argument('--patch-size',type=int,default=512)
    p.add_argument('--max-size',type=int,default=1118)
    p.add_argument('--test-every',type=int,default=8)
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--scale-reg',type=float,default=0)
    p.add_argument('--opacity-reg',type=float,default=0)
    p.add_argument('--means-lr',type=float,default=1.6e-5,help='warm-start position learning rate in capture coordinates')
    p.add_argument('--depth-weight',type=float,default=0,
                   help='experimental color-gated sparse-depth prior; zero disables it')
    p.add_argument('--evaluate-only',action='store_true',help='render heldout views without optimizer steps')
    refine(p.parse_args())
