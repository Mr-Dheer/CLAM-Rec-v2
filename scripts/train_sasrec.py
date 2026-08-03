"""
Train a SASRec checkpoint on OUR preprocessed interactions, in the exact
[kwargs, state_dict] format that clam_rec.model.sasrec.RecSys loads. Training on
our own interactions guarantees the item ids match our itemmap (so CLIP rows align).

Standard A-LLMRec SASRec training: BCE on pos/neg dot-product logits at non-pad
positions, Adam(lr=1e-3, betas=(0.9,0.98)). Reuses our SASRec model + SeqDataset.

Usage:
  python scripts/train_sasrec.py --interactions assets/Prime_Pantry.txt \
    --out assets/sasrec_Prime_Pantry.pth --device cuda:0 --epochs 200
"""
import argparse, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clam_rec.model.sasrec import SASRec
from clam_rec.data.partition import data_partition
from clam_rec.data.seq_dataset import SeqDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--maxlen", type=int, default=50)
    ap.add_argument("--hidden_units", type=int, default=50)
    ap.add_argument("--num_blocks", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=1)
    ap.add_argument("--dropout_rate", type=float, default=0.5)
    ap.add_argument("--l2_emb", type=float, default=0.0)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    user_train, _, _, usernum, itemnum = data_partition(args.interactions)
    print(f"users={usernum} items={itemnum}")

    sargs = SimpleNamespace(device=dev, maxlen=args.maxlen, hidden_units=args.hidden_units,
                            num_blocks=args.num_blocks, num_heads=args.num_heads,
                            dropout_rate=args.dropout_rate)
    model = SASRec(usernum, itemnum, sargs).to(dev)
    for _, p in model.named_parameters():
        try: torch.nn.init.xavier_normal_(p.data)
        except ValueError: pass

    ds = SeqDataset(user_train, usernum, itemnum, args.maxlen)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    model.train(); t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tot = 0.0; nb = 0
        for u, seq, pos, neg in loader:
            u, seq, pos, neg = (x.numpy() for x in (u, seq, pos, neg))
            pos_logits, neg_logits = model(u, seq, pos, neg)   # (B, maxlen)
            pos_lab = torch.ones(pos_logits.shape, device=dev)
            neg_lab = torch.zeros(neg_logits.shape, device=dev)
            idx = np.where(pos != 0)
            opt.zero_grad()
            loss = bce(pos_logits[idx], pos_lab[idx]) + bce(neg_logits[idx], neg_lab[idx])
            for p in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(p)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 20 == 0 or ep == 1:
            print(f"[sasrec] ep {ep}/{args.epochs} loss={tot/nb:.4f} ({time.time()-t0:.0f}s)")

    model.eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save([model.kwargs, model.state_dict()], args.out)
    print(f"saved SASRec -> {args.out}")


if __name__ == "__main__":
    main()
