"""
Deep Mutual Learning — BI-RADS 7-class classification
2 models: Qwen2.5-14B | Qwen3-14B
Hardware: single GPU (H100 80GB) — both models on cuda:0, run sequentially
Paper: https://arxiv.org/pdf/1706.00384

DML loss:
    L_a = CE(logits_a, y) + λ · KL( p_b ‖ p_a )
    L_b = CE(logits_b, y) + λ · KL( p_a ‖ p_b )
"""

import os
import json
import glob
import shutil
import tempfile
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler

from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import get_peft_model, LoraConfig, TaskType
from safetensors.torch import load_file as st_load_file, save_file as st_save_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ── GPU ───────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda:0')

# ── Model paths ───────────────────────────────────────────────────────────────
PATH_A = '/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b-instruct/1'
PATH_B = '/kaggle/input/models/qwen-lm/qwen-3/transformers/14b/1'

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NUM_EPOCHS   = 3
BATCH_SIZE   = 2
GRAD_ACCUM   = 4        # effective batch = 8
MAX_LEN      = 896
NUM_CLASSES  = 7
PATIENCE     = 3
LR           = 2e-4
WEIGHT_DECAY = 0.01
SEED         = 252
LAMBDA_DML   = 1.0
OVERSAMPLE   = {5: 2, 6: 3}

LORA_R      = 8
LORA_ALPHA  = 16
LORA_DROP   = 0.0
LORA_TARGET = 'all-linear'

# ── Seed ──────────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Tokenizers ────────────────────────────────────────────────────────────────
tokenizer_a = AutoTokenizer.from_pretrained(PATH_A, trust_remote_code=True)
tokenizer_b = AutoTokenizer.from_pretrained(PATH_B, trust_remote_code=True)

for tok in (tokenizer_a, tokenizer_b):
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

# ── Prompt ────────────────────────────────────────────────────────────────────
BIRADS_CATEGORIES = {
    0: 'Incompleto — necessita de imagens adicionais',
    1: 'Negativo — sem achados',
    2: 'Achado benigno',
    3: 'Provavelmente benigno — seguimento em curto intervalo',
    4: 'Suspeito — biópsia deve ser considerada',
    5: 'Altamente suspeito de malignidade',
    6: 'Malignidade comprovada por biópsia',
}
CAT_STR = '\n'.join(f'    {k}: {v}' for k, v in BIRADS_CATEGORIES.items())
CLINICAL_NOTES = (
    'Regras: 0=imagem incompleta, 1=negativo, 2=benigno, '
    '3=provavelmente benigno(seguimento 6m), '
    '4=suspeito(biópsia considerar), 5=alta suspeita(biópsia obrigatória), '
    '6=malignidade confirmada por biópsia prévia.\n'
    'ATENÇÃO: 3≠2(sem seguimento), 3≠0(sem conclusão), 5≠4(alta vs suspeita).\n'
)

def create_prompt(report_text: str) -> str:
    return (
        '<|im_start|>user\n'
        'Você é um radiologista especialista em mamografia.\n\n'
        'Tarefa: Classifique o laudo mamográfico abaixo em uma das categorias BI-RADS (0–6).\n'
        'Instrução: Responda APENAS com o número da categoria. NÃO inclua explicação.\n\n'
        f'Categorias BI-RADS:\n{CAT_STR}\n\n'
        f'{CLINICAL_NOTES}\n'
        f'Laudo:\n{report_text.strip()}\n\n'
        'Categoria BI-RADS (número apenas): '
    )

# ── Dataset ───────────────────────────────────────────────────────────────────
class BiRadsDataset(Dataset):
    def __init__(self, prompts: list, targets: list):
        self.prompts = prompts
        self.targets = targets

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)

# ── Key-remap helper ──────────────────────────────────────────────────────────
def _prepare_load_path(
    model_path: str,
    ckpt_prefix: str = 'model.',
) -> tuple[str, bool]:
    """
    Returns (load_path, created_tempdir).
    Neither Qwen2.5-14B nor Qwen3-14B needs remapping in practice, but the
    helper is kept for safety in case a checkpoint variant ships with the
    'model.' prefix wrapping bare architecture keys.
    """
    import safetensors

    ckpt_files = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
    if not ckpt_files:
        print('  [remap] No safetensors found — skipping.')
        return model_path, False

    with safetensors.safe_open(ckpt_files[0], framework='pt', device='cpu') as fh:
        first_ckpt_keys = list(fh.keys())

    prefixed_keys = [k for k in first_ckpt_keys if k.startswith(ckpt_prefix)]
    if not prefixed_keys:
        print(f'  [remap] No {ckpt_prefix!r} prefix in checkpoint — skipping.')
        return model_path, False

    # Both Qwen2.5 and Qwen3 are pure LLMs: AutoModel exposes bare keys
    # (layers.X, not model.layers.X), so ckpt 'model.X' → model 'X'.
    # No VLM wrapping needed — just strip the prefix.
    sample_ckpt = prefixed_keys[0]
    sample_bare = sample_ckpt[len(ckpt_prefix):]

    # Quick check: if checkpoint already has bare keys alongside model.* keys,
    # or if no model.* keys exist at all, nothing to do.
    if not prefixed_keys:
        print('  [remap] Checkpoint already matches model keys — skipping.')
        return model_path, False

    print(f'  [remap] Mismatch detected.')
    print(f'          checkpoint key : {sample_ckpt}')
    print(f'          model key      : {sample_bare}')
    print(f'          rule           : strip {ckpt_prefix!r}')
    print(f'  [remap] Writing {len(ckpt_files)} remapped shard(s) to temp dir …')

    tmp = tempfile.mkdtemp(prefix='ckpt_remap_')

    def _remap_key(k: str) -> str:
        return k[len(ckpt_prefix):] if k.startswith(ckpt_prefix) else k

    for item in os.listdir(model_path):
        if item.endswith('.safetensors'):
            continue
        src = os.path.join(model_path, item)
        dst = os.path.join(tmp, item)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)

    for shard_file in ckpt_files:
        tensors  = st_load_file(shard_file, device='cpu')
        remapped = {_remap_key(k): v for k, v in tensors.items()}
        st_save_file(remapped, os.path.join(tmp, os.path.basename(shard_file)))
        del tensors, remapped

    index_src = os.path.join(model_path, 'model.safetensors.index.json')
    if os.path.exists(index_src):
        with open(index_src) as fh:
            idx = json.load(fh)
        idx['weight_map'] = {_remap_key(k): v for k, v in idx['weight_map'].items()}
        with open(os.path.join(tmp, 'model.safetensors.index.json'), 'w') as fh:
            json.dump(idx, fh, indent=2)

    print(f'  [remap] Ready → {tmp}')
    return tmp, True

# ── Model ─────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.config.use_cache = False
        for attr in ('text_config', 'language_model_config'):
            sub = getattr(self.config, attr, None)
            if sub is not None:
                sub.use_cache = False

        load_path, is_tmp = _prepare_load_path(model_path)
        try:
            backbone = AutoModel.from_pretrained(
                load_path,
                config=self.config,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
                device_map=0,
                trust_remote_code=True,
            )
        finally:
            if is_tmp:
                shutil.rmtree(load_path, ignore_errors=True)

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=LORA_TARGET,
            bias='none',
            inference_mode=False,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROP,
        )
        self.backbone = get_peft_model(backbone, peft_config)
        self.backbone.print_trainable_parameters()

        hidden_size = self._get_hidden_size()
        self.head = nn.Linear(hidden_size, NUM_CLASSES, bias=False).to(DEVICE)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def _get_hidden_size(self) -> int:
        cfg = self.config
        if hasattr(cfg, 'hidden_size'):
            return cfg.hidden_size
        for attr in ('text_config', 'language_model_config'):
            sub = getattr(cfg, attr, None)
            if sub and hasattr(sub, 'hidden_size'):
                return sub.hidden_size
        raise ValueError(f'Cannot find hidden_size in {type(cfg).__name__}')

    def forward(self, encodings: dict) -> torch.Tensor:
        out    = self.backbone(**encodings).last_hidden_state
        hidden = out[:, -1, :].to(torch.float32)
        return self.head(hidden)

# ── DML KL loss (2-model: one peer each) ─────────────────────────────────────
def kl_div(logits_src: torch.Tensor, logits_peer: torch.Tensor) -> torch.Tensor:
    """KL( p_peer ‖ p_src )"""
    return F.kl_div(
        F.log_softmax(logits_src, dim=-1),
        F.softmax(logits_peer.detach(), dim=-1),
        reduction='batchmean',
    )

# ── Optimizer ─────────────────────────────────────────────────────────────────
def build_optimizer(model: nn.Module) -> torch.optim.AdamW:
    no_decay = {'bias', 'LayerNorm.weight', 'layer_norm.weight'}
    decay, no_d = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_d if any(nd in name for nd in no_decay) else decay).append(p)
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': WEIGHT_DECAY},
         {'params': no_d,  'weight_decay': 0.0}],
        lr=LR, betas=(0.9, 0.999), eps=1e-8,
    )

# ── Oversampling ──────────────────────────────────────────────────────────────
def oversample_rare(
    prompts: list,
    targets: list,
    oversample_map: dict,
    seed: int = SEED,
) -> tuple[list, list]:
    extra_p, extra_t = [], []
    for cls, times in oversample_map.items():
        idxs = [i for i, t in enumerate(targets) if t == cls]
        for _ in range(times - 1):
            extra_p.extend(prompts[i] for i in idxs)
            extra_t.extend(targets[i] for i in idxs)
        print(f'  Oversample BI-RADS {cls}: {len(idxs)} → {len(idxs) * times}')
    combined_p = prompts + extra_p
    combined_t = targets + extra_t
    order = np.random.default_rng(seed).permutation(len(combined_p))
    return [combined_p[i] for i in order], [combined_t[i] for i in order]

# ── Tokenize helper ───────────────────────────────────────────────────────────
def tokenize_batch(tok, batch_prompts: list) -> dict:
    enc = tok(
        list(batch_prompts),
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=MAX_LEN,
    )
    return {k: v.to(DEVICE, non_blocking=True) for k, v in enc.items()}

# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    epoch: int,
    name: str,
) -> float:
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    for batch_prompts, batch_targets in tqdm(
        loader, desc=f'[{name}] Val E{epoch + 1}', total=len(loader)
    ):
        enc    = tokenize_batch(tokenizer, batch_prompts)
        labels = torch.tensor(batch_targets, dtype=torch.long).to(DEVICE)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(enc)

        probs        = torch.softmax(logits.float(), dim=1)
        confs, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    preds_np  = np.array(all_preds)
    labels_np = np.array(all_labels)
    confs_np  = np.array(all_confs)
    f1        = f1_score(labels_np, preds_np, average='macro')

    print(f'\n  [{name}] F1-macro: {f1:.4f}')
    cm = confusion_matrix(labels_np, preds_np, labels=list(range(NUM_CLASSES)))
    print('  CM  ' + '   '.join(f'P{c}' for c in range(NUM_CLASSES)))
    for i, row in enumerate(cm):
        print(f'  T{i}  [ {" ".join(f"{v:4d}" for v in row)} ]')
    wrong = preds_np != labels_np
    print(f'  Wrong: {wrong.sum()} / {len(labels_np)}')
    for c in range(NUM_CLASSES):
        cw  = wrong & (labels_np == c)
        tag = (f'{cw.sum():3d} wrong | avg conf {confs_np[cw].mean():.3f}'
               if cw.sum() else '  0 wrong ✓')
        print(f'  BI-RADS {c}: {tag}')

    return f1

# ── Training ──────────────────────────────────────────────────────────────────
def train(
    train_prompts: list,
    train_targets: list,
    val_prompts: list,
    val_targets: list,
) -> None:
    print('\n' + '=' * 60)
    print('  TRAINING  (single GPU, sequential forward/backward)')
    print('  Models: Qwen2.5-14B (A)  |  Qwen3-14B (B)')
    print('=' * 60)

    # ── oversampling & class weights ──────────────────────────────────────
    train_prompts, train_targets = oversample_rare(
        train_prompts, train_targets, OVERSAMPLE
    )
    print(f'\n  Train samples after oversampling : {len(train_targets)}')
    print(f'  Val samples                      : {len(val_targets)}')

    counts  = np.bincount(train_targets, minlength=NUM_CLASSES).astype(float)
    weights = len(train_targets) / (counts + 1e-6)
    print('\n  Class weights:')
    for c, (cnt, w) in enumerate(zip(counts, weights)):
        print(f'    BI-RADS {c}: n={int(cnt):4d}  w={w:.2f}')
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    # ── datasets & loaders ────────────────────────────────────────────────
    train_ds = BiRadsDataset(train_prompts, train_targets)
    val_ds   = BiRadsDataset(val_prompts,   val_targets)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, pin_memory=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=0,
    )

    # ── models ────────────────────────────────────────────────────────────
    print('\nLoading model A  (Qwen2.5-14B)...')
    model_a = Net(PATH_A)
    print('Loading model B  (Qwen3-14B)...')
    model_b = Net(PATH_B)

    KEYS   = ('a', 'b')
    models = {'a': model_a, 'b': model_b}
    toks   = {'a': tokenizer_a, 'b': tokenizer_b}
    peers  = {'a': 'b', 'b': 'a'}   # each model's DML peer

    opts = {k: build_optimizer(models[k]) for k in KEYS}

    total_steps  = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = max(1, total_steps // 10)
    scheds = {
        k: get_cosine_schedule_with_warmup(opts[k], warmup_steps, total_steps)
        for k in KEYS
    }

    scaler = GradScaler(device='cuda')

    best_f1 = {k: 0.0   for k in KEYS}
    no_imp  = {k: 0     for k in KEYS}
    stopped = {k: False for k in KEYS}

    # ── epoch loop ────────────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        print(f'\n{"─" * 60}')
        print(f'  Epoch {epoch + 1}/{NUM_EPOCHS}')
        print(f'{"─" * 60}')

        for k in KEYS:
            if not stopped[k]:
                models[k].train()
                opts[k].zero_grad(set_to_none=True)

        loss_sum = {k: 0.0 for k in KEYS}

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'E{epoch + 1} Train',
        )

        for step, (batch_prompts, batch_targets) in pbar:

            labels = torch.tensor(batch_targets, dtype=torch.long).to(DEVICE)

            # ── tokenize for active models ─────────────────────────────────
            encs = {
                k: tokenize_batch(toks[k], batch_prompts)
                for k in KEYS if not stopped[k]
            }

            # ── sequential forward passes ──────────────────────────────────
            logits: dict[str, torch.Tensor | None] = {k: None for k in KEYS}
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for k in KEYS:
                    if not stopped[k]:
                        logits[k] = models[k](encs[k]).float()

            # ── DML losses ────────────────────────────────────────────────
            # L_k = CE(logits_k, y) + λ · KL( p_peer ‖ p_k )
            active_losses: dict[str, torch.Tensor] = {}
            for k in KEYS:
                lg = logits[k]
                if lg is None:
                    continue
                ce = F.cross_entropy(lg, labels, weight=class_weights)
                peer_lg = logits[peers[k]]
                if peer_lg is not None:
                    kl = kl_div(lg, peer_lg)
                else:
                    kl = lg.new_zeros(1).squeeze()
                active_losses[k] = (ce + LAMBDA_DML * kl) / GRAD_ACCUM
                loss_sum[k] += active_losses[k].item() * GRAD_ACCUM

            # ── sequential backward passes ─────────────────────────────────
            for loss in active_losses.values():
                scaler.scale(loss).backward()

            # ── optimizer step ─────────────────────────────────────────────
            if (step + 1) % GRAD_ACCUM == 0:
                for k in active_losses:
                    scaler.unscale_(opts[k])
                    torch.nn.utils.clip_grad_norm_(
                        models[k].parameters(), max_norm=10.0
                    )
                    scaler.step(opts[k])
                    scheds[k].step()
                    opts[k].zero_grad(set_to_none=True)
                scaler.update()

            n_steps = step + 1
            pbar.set_postfix({
                f'loss_{k}': f'{loss_sum[k] / n_steps:.3f}' for k in KEYS
            })

        n_steps = len(train_loader)
        parts   = '  '.join(
            f'{k.upper()}:{loss_sum[k] / n_steps:.4f}' for k in KEYS
        )
        print(f'\n  Avg loss — {parts}  ({(time.time() - t0) / 60:.1f} min)')

        # ── validation ────────────────────────────────────────────────────
        print('\n  -- Validation --')
        for k in KEYS:
            if stopped[k]:
                print(f'  [{k.upper()}] already stopped — skipping')
                continue

            f1 = validate(models[k], val_loader, toks[k], epoch, k.upper())

            if f1 > best_f1[k]:
                best_f1[k] = f1
                no_imp[k]  = 0
                models[k].backbone.save_pretrained(f'backbone_{k}_best')
                torch.save(models[k].head.state_dict(), f'head_{k}_best.pt')
                print(f'  [{k.upper()}] ✓ New best F1: {best_f1[k]:.4f}')
            else:
                no_imp[k] += 1
                print(f'  [{k.upper()}] No improvement ({no_imp[k]}/{PATIENCE})')
                if no_imp[k] >= PATIENCE:
                    print(f'  [{k.upper()}] Early stopping.')
                    stopped[k] = True

        if all(stopped.values()):
            print('\n  All models early-stopped.')
            break

    # ── summary ───────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  Training complete.')
    f1_parts = '  '.join(f'{k.upper()}:{best_f1[k]:.4f}' for k in KEYS)
    print(f'  Best F1 — {f1_parts}')
    print('=' * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(SEED)

    print('PyTorch:', torch.__version__)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  cuda:{i}  {p.name}  {p.total_memory / 1e9:.0f} GB')

    train_df = pd.read_csv('/kaggle/input/datasets/rayenv2/train-clean/df_clean.csv')
    ext_df   = pd.read_csv('/kaggle/input/datasets/rayenv2/train-clean/train_external.csv')

    ext_rare = ext_df[ext_df['target'].isin([4, 5, 6])].copy()
    aug_df   = (pd.concat([train_df, ext_rare], ignore_index=True)
                  .sample(frac=1, random_state=SEED)
                  .reset_index(drop=True))

    print(f'\nTotal samples: {len(aug_df)}')
    print(aug_df['target'].value_counts().sort_index())

    all_prompts = [create_prompt(r) for r in aug_df['report']]
    all_targets = aug_df['target'].tolist()

    lengths = np.array([
        len(tokenizer_a(p, add_special_tokens=False).input_ids)
        for p in all_prompts
    ])
    print(
        f'\nPrompt lengths — mean={lengths.mean():.0f}  '
        f'p99={np.percentile(lengths, 99):.0f}  max={lengths.max()}'
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(all_prompts)),
        test_size=0.15,
        stratify=all_targets,
        random_state=SEED,
    )

    train_prompts = [all_prompts[i] for i in train_idx]
    val_prompts   = [all_prompts[i] for i in val_idx]
    train_targets = [all_targets[i] for i in train_idx]
    val_targets   = [all_targets[i] for i in val_idx]

    train(train_prompts, train_targets, val_prompts, val_targets)

    print('\nDone.')