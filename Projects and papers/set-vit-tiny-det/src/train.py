import argparse, yaml, torch, torch.nn as nn
from pathlib import Path
from src.datasets.coco import build_coco_loaders
from src.models.deformable_detr_backbone import build_model   # <-- your baseline
from src.models.set_modules.hbs import HBS, odd_kernel_from_stride
from src.models.set_modules.api import APIAuxPath
from src.utils.masks import boxes_to_mask, downscale_boxes

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    return ap.parse_args()

def main():
    args = parse()
    cfg = yaml.safe_load(open(args.config))

    device = cfg['device']
    train_loader, val_loader = build_coco_loaders(cfg)

    model, head, criterion, strides = build_model(cfg)  # returns strides per scale (e.g., [8,16,32])
    model.to(device); head.to(device)

    # HBS modules per scale
    if cfg['model']['hbs']['enabled']:
        hbs_blocks = nn.ModuleList()
        for i, s in enumerate(strides):
            K = odd_kernel_from_stride(s)
            hbs_blocks.append(HBS(head.feat_channels[i], reduction=cfg['model']['hbs']['reduction'], kernel_size=K))
        hbs_blocks.to(device)
    else:
        hbs_blocks = None

    # API auxiliary path
    api_aux = None
    if cfg['model']['api']['enabled']:
        api_aux = APIAuxPath(head=head,
                             rho=cfg['model']['api']['rho'],
                             lambda_cls=cfg['model']['api']['lambda_cls'],
                             lambda_reg=cfg['model']['api']['lambda_reg']).to(device)

    opt = torch.optim.AdamW([p for p in list(model.parameters()) + list(head.parameters())
                            + (list(hbs_blocks.parameters()) if hbs_blocks else [])], 
                            lr=cfg['optim']['lr'], weight_decay=cfg['optim']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.get('precision','amp')=='amp'))

    for epoch in range(cfg['optim']['epochs']):
        model.train(); head.train()
        for it, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            targets = [{k:v.to(device) if torch.is_tensor(v) else v for k,v in t.items()} for t in batch['targets']]

            with torch.cuda.amp.autocast(enabled=(cfg['precision']=='amp')):
                feats = model(imgs)                      # list of multi-scale features
                # Build binary masks per scale
                Ms = []
                for f, s in zip(feats, strides):
                    B, C, H, W = f.shape
                    # scale boxes to (H,W)
                    boxes_batch = []
                    for t in targets:
                        boxes = t['boxes'].clone()  # xyxy in image coords
                        scale_x = W / imgs.shape[-1]
                        scale_y = H / imgs.shape[-2]
                        boxes = downscale_boxes(boxes, scale_y, scale_x)
                        boxes_batch.append(boxes)
                    # merge to per-image mask list
                    M_list = []
                    for b in range(B):
                        M_list.append(boxes_to_mask(boxes_batch[b], H, W, imgs.device))
                    Ms.append(torch.stack(M_list, dim=0))  # (B,1,H,W)

                # HBS (training only)
                if hbs_blocks:
                    feats_hbs = [hbs(feat, M) for hbs, feat, M in zip(hbs_blocks, feats, Ms)]
                else:
                    feats_hbs = feats

                # Main path loss
                outputs = head(feats_hbs)
                loss_dict = criterion(outputs, targets)
                loss_main = sum(loss_dict.values())

                # Auxiliary API path
                loss_aux = 0.0
                if api_aux:
                    aux_loss_dict = api_aux(feats_hbs, criterion, targets)
                    loss_aux = cfg['loss']['aux_weight'] * sum(aux_loss_dict.values())

                loss = loss_main + loss_aux

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), cfg['optim']['grad_clip'])
            scaler.step(opt); scaler.update()

            if (it+1) % cfg['log']['interval'] == 0:
                print(f"ep{epoch} it{it+1} loss={loss.item():.3f} main={loss_main.item():.3f} aux={float(loss_aux):.3f}")

        # TODO: save checkpoint; run validation; log AP/AP_S

if __name__ == "__main__":
    main()
