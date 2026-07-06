"""
Gradient alignment between the main loss and the auxiliary-exit losses.

Motivation: "does the auxiliary loss work with or against the main loss?"
The scalar losses cannot answer this; the standard formalization (gradient
similarity in multi-task learning, cf. Du et al. 2018; Yu et al. 2020 PCGrad)
is the cosine between grad(L_main) and grad(L_aux) over shared parameters:

    cos > 0  -> the two losses pull the weights in the same direction (cooperate)
    cos < 0  -> they pull in opposite directions (conflict)

Both exits (s16 and s32) are measured in the SAME forward pass, so the
s16-vs-s32 comparison is exactly paired: same checkpoint, same image, same
dropout state, same g_main. The exits are parameter-free (they reuse the
decoder score maps), so every parameter touched by an aux loss is shared.

What is reported per checkpoint (CSV row):
  - cos16/cos32 (mean, std, sem), frac_conflict (share of batches with cos<0)
  - paired_diff = mean_i(cos16_i - cos32_i) with its sem  <- headline statistic
  - proj16/proj32 = dot(g_aux, g_main)/||g_main||: first-order effect of a unit
    aux step on L_main (multiply by the run's lambda for the training-time pull;
    the cosine itself is invariant to lambda > 0)
  - ratio16/ratio32 = ||g_aux|| / ||g_main|| (unweighted; multiply by lambda)
  - cos_act16/cos_act32: cosine in ACTIVATION space at the tap tensor
    (dL_aux/d score4 vs dL_main/d score4, and likewise at score5) -- the most
    direct test of whether the main loss wants the score map to move where the
    aux loss pushes it
  - cos_agg16/cos_agg32: cosine of batch-AGGREGATED gradients (systematic,
    low-noise version of the mean per-batch cosine)
  - cos_s16_s32: how similarly the two aux losses pull the shared parameters
  - null_cos: cosine between g_main of consecutive (different) images -- the
    natural "cooperation ceiling" against which cos16/cos32 should be read
    (raw cosines in ~1.3e8 dimensions concentrate near 0; interpret sign and
    relative ordering, never raw magnitude)
  - per-block cosines (features_123/4/5, classifier, score_feat4, upscore_5).
    Note: the classifier holds ~89% of the aux-reachable parameters, so the
    overall cosine is dominated by it; compare per-block values only within
    the same block. s32 reaches only the encoder blocks + classifier.
  - mean L_main / L_aux16 / L_aux32 over the sampled batches

Protocol notes:
  - Dropout is DISABLED by default (deterministic, reproducible gradients);
    pass --dropout to measure with training-time dropout active (masks are
    seeded and shared by both losses within a batch, so the per-batch
    comparison stays fair either way).
  - The same --seed fixes the sampled images for every checkpoint and run.
  - This measures the gradient decomposition of the TRAINING loss construction
    evaluated at saved checkpoint states; it mirrors main.py's default aux
    variant (label downsampled, unweighted CE, ignore_index=-1). Checkpoints
    trained with --aux_upsample are REJECTED (their aux loss went through
    learned deconvolutions that this script does not replicate); runs trained
    with --aux_interp or --class_weights must be matched with the
    corresponding flags here, or the measured gradients are not the trained ones.
  - Measuring along a model's own trajectory shows the conflict as it happened
    during that run; for a controlled s16-vs-s32 comparison, ALSO run this on
    the BASELINE (no-aux) checkpoints -- same weights for both exits, so any
    difference is attributable purely to the tap location.
  - Checkpoints are saved both periodically and on mIoU improvement; use
    --periodic_only so different runs are sampled on the identical epoch grid.

Usage (from src/, same conventions as main.py):
  python grad_alignment.py --data ./train --param "out/FCN-epoch-*.pth"
  python grad_alignment.py --data ./train --param out/ --periodic_only --batches 200
  python grad_alignment.py --selftest        # synthetic smoke test, no dataset needed
"""

import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from models.FCN_8 import FCN8s
from utils.criterion import CrossEntropyLoss2d

# blocks that can receive gradient from the aux losses (s32 reaches only the
# first four; score_feat3/upscore_4/upscore never receive aux gradient)
BLOCKS = ["features_123", "features_4", "features_5", "classifier",
          "score_feat4", "upscore_5"]
EXITS = ["s16", "s32"]
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./train', help='path to SBD root (same as main.py --data)')
    p.add_argument('--param', type=str, default=None, help='checkpoint file, glob or directory')
    p.add_argument('--batches', type=int, default=200, help='images sampled per checkpoint')
    p.add_argument('--no_skip', action='store_true', default=False)
    p.add_argument('--class_weights', action='store_true', default=False,
                   help='use the same cached class weights as a --class_weights training run')
    p.add_argument('--aux_interp', action='store_true', default=False,
                   help='mirror an --aux_interp training run (bilinear-upsampled aux logits, full-res target)')
    p.add_argument('--dropout', action='store_true', default=False,
                   help='measure with training-time dropout active (default: dropout off, deterministic)')
    p.add_argument('--periodic_only', action='store_true', default=False,
                   help='keep only epoch %% 25 == 0 checkpoints (same grid across runs)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', type=str, default='grad_alignment.csv')
    p.add_argument('--selftest', action='store_true', default=False)
    return p.parse_args()


def find_checkpoints(param, periodic_only):
    if os.path.isdir(param):
        files = glob.glob(os.path.join(param, '*.pth'))
    else:
        files = glob.glob(param)
    if not files:
        raise FileNotFoundError('no checkpoint matches %s' % param)

    def epoch_of(path):
        m = re.search(r'epoch-(\d+)\.pth$', os.path.basename(path))
        return int(m.group(1)) if m else -1
    files = sorted(files, key=epoch_of)
    if periodic_only:
        files = [f for f in files if epoch_of(f) % 25 == 0]
    return files


def load_model(path, n_class, no_skip, device):
    try:
        ckpt = torch.load(path, map_location='cpu')
    except Exception:
        # older-torch pickles may need weights_only=False on torch >= 2.6
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    if any(k.startswith('aux_upscore') for k in state):
        raise RuntimeError(
            '%s was trained with --aux_upsample; its aux loss went through learned '
            'deconvolutions that this script does not replicate. Refusing to produce '
            'unfaithful measurements.' % path)
    model = FCN8s(n_class, aux=True, no_skip=no_skip)
    model.load_state_dict(state)
    epoch = ckpt.get('epoch', None) if isinstance(ckpt, dict) else None
    if epoch is None:
        m = re.search(r'epoch-(\d+)\.pth$', os.path.basename(path))
        epoch = int(m.group(1)) if m else -1
    return model.to(device), epoch


def set_dropout(model, active):
    """Call AFTER model.train(): train() re-enables every submodule."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train(active)


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm() + EPS))


def aux_loss(criterion, aux_map, targets, aux_interp):
    if aux_interp:                       # mirrors main.py --aux_interp branch
        up = F.interpolate(aux_map, size=targets.shape[-2:],
                           mode='bilinear', align_corners=False)
        return criterion(up, targets)
    gt_low = F.interpolate(targets.unsqueeze(1).float(), size=aux_map.shape[-2:],
                           mode='nearest').squeeze(1).long()   # mirrors main.py default
    return criterion(aux_map, gt_low)


def batch_metrics(model, criterion, inputs, targets, params, names, aux_interp):
    """One forward, both exits; returns a dict of per-batch metrics or None."""
    out = model(inputs)
    assert isinstance(out, tuple), 'model must be in train mode with aux=True'
    h, aux = out
    taps = {'s16': aux['s16'], 's32': aux['s32']}

    loss_main = criterion(h, targets)
    loss_aux = {e: aux_loss(criterion, taps[e], targets, aux_interp) for e in EXITS}
    if not (torch.isfinite(loss_main) and all(torch.isfinite(l) for l in loss_aux.values())):
        return None

    # parameter-space gradients (shared graph -> identical dropout mask for all)
    g_main = torch.autograd.grad(loss_main, params, retain_graph=True, allow_unused=True)
    g_aux = {e: torch.autograd.grad(loss_aux[e], params, retain_graph=True, allow_unused=True)
             for e in EXITS}
    # activation-space gradients at the tap tensors
    gm_act = torch.autograd.grad(loss_main, [taps['s16'], taps['s32']],
                                 retain_graph=True, allow_unused=True)
    ga_act16 = torch.autograd.grad(loss_aux['s16'], taps['s16'], retain_graph=True)[0]
    ga_act32 = torch.autograd.grad(loss_aux['s32'], taps['s32'])[0]

    m = {'loss_main': loss_main.item(), 'loss_aux16': loss_aux['s16'].item(),
         'loss_aux32': loss_aux['s32'].item()}

    flats = {}
    for e in EXITS:
        pairs = [(gm, ga, n) for gm, ga, n in zip(g_main, g_aux[e], names)
                 if gm is not None and ga is not None]
        if not pairs:
            return None
        vm = torch.cat([gm.reshape(-1) for gm, _, _ in pairs])
        va = torch.cat([ga.reshape(-1) for _, ga, _ in pairs])
        s = '16' if e == 's16' else '32'
        m['cos' + s] = cos(vm, va)
        m['proj' + s] = float(torch.dot(vm, va) / (vm.norm() + EPS))
        r = float(va.norm() / (vm.norm() + EPS))
        m['ratio' + s] = r
        # gradient magnitude similarity (PCGrad, Def. 2); lambda-dependent:
        # for the training-time value use r' = lambda * ratio in 2r'/(1+r'^2)
        m['phi' + s] = 2 * r / (1 + r * r)
        for b in BLOCKS:
            bm = [gm.reshape(-1) for gm, _, n in pairs if n.startswith(b + '.')]
            ba = [ga.reshape(-1) for _, ga, n in pairs if n.startswith(b + '.')]
            if bm:
                fm, fa = torch.cat(bm), torch.cat(ba)
                if fm.norm() > EPS and fa.norm() > EPS:
                    m['cos%s_%s' % (s, b)] = cos(fm, fa)
        flats[e] = va
    if not np.isfinite(m['cos16']) or not np.isfinite(m['cos32']):
        return None

    # s16-vs-s32 pull similarity on the parameters both exits reach
    both = [(a16, a32) for a16, a32, gm in zip(g_aux['s16'], g_aux['s32'], g_main)
            if a16 is not None and a32 is not None and gm is not None]
    v16 = torch.cat([a.reshape(-1) for a, _ in both])
    v32 = torch.cat([a.reshape(-1) for _, a in both])
    m['cos_s16_s32'] = cos(v16, v32)

    # activation-space cosines at the taps
    m['cos_act16'] = cos(gm_act[0].reshape(-1), ga_act16.reshape(-1))
    m['cos_act32'] = cos(gm_act[1].reshape(-1), ga_act32.reshape(-1))

    # full flattened gradients for aggregation / null (moved to CPU by caller)
    m['_g_main'] = torch.cat([g.reshape(-1) for g in g_main if g is not None])
    m['_g_a16'] = flats['s16']
    m['_g_a32'] = flats['s32']
    return m


def run_checkpoint(model, loader, criterion, n_batches, device, dropout, aux_interp, seed):
    model.train()                     # forward only returns the aux dict in train mode
    set_dropout(model, dropout)       # MUST come after train(); train() resets submodules
    if dropout:
        assert all(mm.training for mm in model.modules() if isinstance(mm, torch.nn.Dropout2d))
    else:
        assert not any(mm.training for mm in model.modules() if isinstance(mm, torch.nn.Dropout2d))
    torch.manual_seed(seed)           # reproducible dropout masks (when active)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    scalars, skipped = [], 0
    agg = {}          # CPU accumulators of summed gradients
    prev_g_main = None
    null_list = []

    for ib, data in enumerate(loader):
        if ib >= n_batches:
            break
        inputs = data[0].to(device)
        targets = data[1].to(device).long()
        m = batch_metrics(model, criterion, inputs, targets, params, names, aux_interp)
        if m is None:
            skipped += 1
            continue
        g_main = m.pop('_g_main').detach().to('cpu')
        g_a16 = m.pop('_g_a16').detach().to('cpu')
        g_a32 = m.pop('_g_a32').detach().to('cpu')
        for k, v in (('g_main', g_main), ('g_a16', g_a16), ('g_a32', g_a32)):
            agg[k] = v if k not in agg else agg[k] + v
        if prev_g_main is not None and prev_g_main.numel() == g_main.numel():
            null_list.append(cos(prev_g_main, g_main))
        prev_g_main = g_main
        scalars.append(m)

    def col(key):
        vals = [s[key] for s in scalars if key in s and np.isfinite(s[key])]
        return np.array(vals) if vals else np.array([np.nan])

    c16, c32 = col('cos16'), col('cos32')
    paired = c16 - c32 if len(c16) == len(c32) else np.array([np.nan])
    n = max(len(c16), 1)
    row = {
        'batches': len(scalars), 'skipped': skipped,
        'cos16_mean': c16.mean(), 'cos16_std': c16.std(), 'cos16_sem': c16.std() / np.sqrt(n),
        'cos32_mean': c32.mean(), 'cos32_std': c32.std(), 'cos32_sem': c32.std() / np.sqrt(n),
        'frac_conflict16': float((c16 < 0).mean()), 'frac_conflict32': float((c32 < 0).mean()),
        'paired_diff_mean': paired.mean(), 'paired_diff_sem': paired.std() / np.sqrt(n),
        'proj16_mean': col('proj16').mean(), 'proj32_mean': col('proj32').mean(),
        'ratio16_mean': col('ratio16').mean(), 'ratio32_mean': col('ratio32').mean(),
        'phi16_mean': col('phi16').mean(), 'phi32_mean': col('phi32').mean(),
        'cos_act16_mean': col('cos_act16').mean(), 'cos_act32_mean': col('cos_act32').mean(),
        'cos_s16_s32_mean': col('cos_s16_s32').mean(),
        'cos_agg16': cos(agg['g_main'], agg['g_a16']) if agg else float('nan'),
        'cos_agg32': cos(agg['g_main'], agg['g_a32']) if agg else float('nan'),
        'null_cos_mean': float(np.mean(null_list)) if null_list else float('nan'),
        'loss_main_mean': col('loss_main').mean(),
        'loss_aux16_mean': col('loss_aux16').mean(), 'loss_aux32_mean': col('loss_aux32').mean(),
    }
    for s in ('16', '32'):
        for b in BLOCKS:
            row['cos%s_%s' % (s, b)] = col('cos%s_%s' % (s, b)).mean()
    return row


CSV_COLS = (['epoch', 'batches', 'skipped',
             'cos16_mean', 'cos16_std', 'cos16_sem', 'frac_conflict16',
             'cos32_mean', 'cos32_std', 'cos32_sem', 'frac_conflict32',
             'paired_diff_mean', 'paired_diff_sem',
             'proj16_mean', 'proj32_mean', 'ratio16_mean', 'ratio32_mean',
             'phi16_mean', 'phi32_mean',
             'cos_act16_mean', 'cos_act32_mean', 'cos_s16_s32_mean',
             'cos_agg16', 'cos_agg32', 'null_cos_mean',
             'loss_main_mean', 'loss_aux16_mean', 'loss_aux32_mean']
            + ['cos16_' + b for b in BLOCKS] + ['cos32_' + b for b in BLOCKS])


def selftest(device):
    """Synthetic smoke test: random weights + random data, both exits."""
    torch.manual_seed(0)
    n_class = 21
    model = FCN8s(n_class, aux=True).to(device)
    criterion = CrossEntropyLoss2d()
    model.train()
    set_dropout(model, False)
    assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.Dropout2d)), \
        'dropout override must survive model.train()'
    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    for i in range(2):
        x = torch.randn(1, 3, 256, 320, device=device)
        y = torch.randint(-1, n_class, (1, 256, 320), device=device)
        m = batch_metrics(model, criterion, x, y, params, names, aux_interp=False)
        assert m is not None
        for k in ('cos16', 'cos32', 'cos_act16', 'cos_act32', 'cos_s16_s32',
                  'proj16', 'proj32', 'ratio16', 'ratio32'):
            assert np.isfinite(m[k]), k
        assert 'cos32_score_feat4' not in m and 'cos32_upscore_5' not in m, \
            's32 must not receive gradient through decoder-only blocks'
        assert 'cos16_score_feat4' in m and 'cos16_upscore_5' in m
        print('[selftest] batch=%d cos16=%+.4f cos32=%+.4f act16=%+.4f act32=%+.4f s16xs32=%+.4f'
              % (i, m['cos16'], m['cos32'], m['cos_act16'], m['cos_act32'], m['cos_s16_s32']))
    # determinism check with dropout off
    x = torch.randn(1, 3, 256, 320, device=device)
    y = torch.randint(-1, n_class, (1, 256, 320), device=device)
    m1 = batch_metrics(model, criterion, x, y, params, names, aux_interp=False)
    m2 = batch_metrics(model, criterion, x, y, params, names, aux_interp=False)
    assert abs(m1['cos16'] - m2['cos16']) < 1e-6, 'gradients must be deterministic with dropout off'
    # aux_interp variant must also run
    m3 = batch_metrics(model, criterion, x, y, params, names, aux_interp=True)
    assert m3 is not None and np.isfinite(m3['cos16'])
    print('[selftest] OK')


def main():
    opt = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    if opt.selftest:
        selftest(device)
        return

    from data.data import SBDClassSeg  # imported here so --selftest needs no dataset
    dataset = SBDClassSeg(opt.data, split='train', transform=True)
    gen = torch.Generator().manual_seed(opt.seed)  # same images for every checkpoint
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True,
                                         num_workers=2, pin_memory=True, generator=gen)
    n_class = len(SBDClassSeg.class_names)

    class_weights = None
    if opt.class_weights:
        from utils.class_weights import compute_class_weights
        cache_path = os.path.join(opt.data, 'class_weights.npy')
        class_weights = compute_class_weights(dataset, n_class, cache_path).to(device)
        print('class weights:', np.round(class_weights.cpu().numpy(), 3))
    criterion = CrossEntropyLoss2d(weight=class_weights)

    ckpts = find_checkpoints(opt.param, opt.periodic_only)
    print('%d checkpoint(s); %d batches each; dropout=%s; aux_interp=%s'
          % (len(ckpts), opt.batches, opt.dropout, opt.aux_interp))

    rows = []
    for path in ckpts:
        model, epoch = load_model(path, n_class, opt.no_skip, device)
        gen.manual_seed(opt.seed)  # identical image order for every checkpoint
        row = run_checkpoint(model, loader, criterion, opt.batches, device,
                             opt.dropout, opt.aux_interp, opt.seed)
        row['epoch'] = epoch
        rows.append(row)
        print('epoch %3d | cos16 %+.4f | cos32 %+.4f | diff %+.4f (sem %.4f) | '
              'act16 %+.4f act32 %+.4f | null %+.4f'
              % (epoch, row['cos16_mean'], row['cos32_mean'], row['paired_diff_mean'],
                 row['paired_diff_sem'], row['cos_act16_mean'], row['cos_act32_mean'],
                 row['null_cos_mean']))
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    with open(opt.out, 'w') as f:
        f.write(','.join(CSV_COLS) + '\n')
        for r in sorted(rows, key=lambda r: r['epoch']):
            f.write(','.join('%.6g' % r[c] if isinstance(r[c], float) else str(r[c])
                             for c in CSV_COLS) + '\n')
    print('wrote', opt.out)


if __name__ == '__main__':
    main()
