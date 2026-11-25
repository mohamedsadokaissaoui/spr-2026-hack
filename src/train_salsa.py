import os
import gc
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    get_cosine_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ALLOC_CONF']     = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')

# ── Constants ─────────────────────────────────────────────────────────────────
model_path  = '/kaggle/input/models/qwen-lm/qwen-3/transformers/8b/1'
DEVICE      = 'cuda:0'
num_folds   = 3
num_epochs  = 6
batch_size  = 4
grad_accum  = 4
MAX_LEN     = 1024
NUM_CLASSES = 7
PATIENCE    = 2
OVERSAMPLE  = {5: 4, 6: 2}
START_FOLD  = 1
SEED        = 252

# ── SALSA: answer tokens ──────────────────────────────────────────────────────
ANSWER_TOKENS = ['0', '1', '2', '3', '4', '5', '6']
Q = 0.955

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

ANSWER_TOKEN_IDS = []
for t in ANSWER_TOKENS:
    ids = tokenizer.encode(t, add_special_tokens=False)
    assert len(ids) == 1, f"Token '{t}' is not a single token: {ids}"
    ANSWER_TOKEN_IDS.append(ids[0])
ANSWER_TOKEN_IDS = torch.tensor(ANSWER_TOKEN_IDS)
print(f"Answer token IDs: {list(zip(ANSWER_TOKENS, ANSWER_TOKEN_IDS.tolist()))}")

EOT_TOKEN_ID = tokenizer.convert_tokens_to_ids('<|im_end|>')

# ── Keyword detectors ─────────────────────────────────────────────────────────
_FOLLOWUP_RE = re.compile(
    r'controle|seguimento|retorno|acompanhamento|'
    r'reavalia[çc][aã]o|bi[oó]psia|pun[cç][aã]o|'
    r'(?<!\d)(\d{1,2})\s*meses',
    re.IGNORECASE,
)
_NO_PRIOR_RE = re.compile(
    r'anteriores n[aã]o dispon[ií]veis|primeira mamografia|'
    r'sem exames anteriores|imagens indispon[ií]veis',
    re.IGNORECASE,
)
_COMPLETE_RE = re.compile(
    r'à ultrassonografia|ao ultrassom|à us |compressão localizada|'
    r'magnificação|incidências adicionais realizadas',
    re.IGNORECASE,
)

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
CAT_STR = '\n'.join([f'    {k}: {v}' for k, v in BIRADS_CATEGORIES.items()])
ANSWER_MARKER = '<BIRADS>'


def create_prompt_user(report_text: str) -> str:
    has_followup = bool(_FOLLOWUP_RE.search(report_text))
    has_no_prior = bool(_NO_PRIOR_RE.search(report_text))
    is_complete  = bool(_COMPLETE_RE.search(report_text))

    followup_hint = (
        '    ⚠️ Linguagem de seguimento detectada → forte indicador de categoria 3.\n'
        if has_followup else
        '    ⚠️ Sem linguagem de seguimento → se benigno, prefira categoria 2.\n'
    )
    no_prior_hint = (
        '    ⚠️ Ausência de exames anteriores NÃO implica categoria 0.\n'
        if has_no_prior else ''
    )
    complete_hint = (
        '    ⚠️ Complementação já realizada e descrita → exame COMPLETO, não use 0.\n'
        if is_complete else ''
    )

    return (
        'Você é um radiologista especialista em mamografia.\n\n'
        'Tarefa: Classifique o laudo mamográfico abaixo em BI-RADS (0–6).\n'
        'Siga os 4 passos abaixo EM ORDEM.\n\n'

        'PASSO 1 — O exame está COMPLETO?\n'
        '  → Use 0 SOMENTE se o laudo solicita imagens que AINDA NÃO foram feitas.\n'
        f'{no_prior_hint}'
        f'{complete_hint}'
        '  → Se completo, continue para o Passo 2.\n\n'

        'PASSO 2 — HÁ ALGUM ACHADO?\n'
        '  → Se não há achados → use 1 e pare.\n'
        '  → Se há achado, continue para o Passo 3.\n\n'

        'PASSO 3 — O ACHADO É BENIGNO OU SUSPEITO?\n'
        '  Se BENIGNO:\n'
        '    • Use 2 (SEM seguimento): estável ≥2 anos, linfonodo/cisto confirmado,\n'
        '      calcificações vasculares/grosseiras/em pipoca estáveis.\n'
        f'{followup_hint}'
        '    • Use 3 (COM seguimento 6m): primeiro exame + achado benigno,\n'
        '      estabilidade <12 meses, assimetria persistente sem US correlação.\n\n'
        '  Se SUSPEITO:\n'
        '    • 4 = biópsia a considerar | 5 = biópsia obrigatória | 6 = malignidade confirmada\n\n'

        'PASSO 4 — VERIFICAÇÃO: 3≠2 | 3≠0 | 5≠4\n\n'

        f'Categorias:\n{CAT_STR}\n\n'
        f'Laudo:\n{report_text.strip()}\n\n'
        f'Responda APENAS com o número BI-RADS no formato: {ANSWER_MARKER}#Número</BIRADS>\n'
        f'Onde o número é um de: 0, 1, 2, 3, 4, 5, 6'
    )


def build_full_prompt(report_text: str, label: int = None) -> str:
    """
    Training  (label is not None):
        assistant turn = '<BIRADS>3</BIRADS><|im_end|>'
        After the causal shift, logits at (digit_pos - 1) predict the digit token.
        find_answer_position labels the digit token position → correct SALSA target.

    Inference (label is None):
        assistant turn ends at '<BIRADS>' (trailing EOT stripped).
        logits[:, -1, :] = model's next-token prediction = the digit.
    """
    user_content = create_prompt_user(report_text)
    messages = [{'role': 'user', 'content': user_content}]

    if label is not None:
        messages.append({
            'role': 'assistant',
            'content': f'{ANSWER_MARKER}{label}</BIRADS>'
        })
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    else:
        messages.append({
            'role': 'assistant',
            'content': ANSWER_MARKER
        })
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Strip trailing EOT so logits[:, -1, :] predicts the digit
        if text.endswith('<|im_end|>\n'):
            text = text[:-len('<|im_end|>\n')]
        return text


def find_answer_position(input_ids: list) -> int:
    """
    Find the position of the digit token in the training sequence.

    Training sequence tail:
        ... <BIRADS_tokens> <digit> </BIRADS_tokens> <|im_end|>

    We label the digit token position. After the causal shift:
        shift_logits[digit_pos - 1]  ← logit that predicts the digit
        mapped_labels[digit_pos - 1] ← remapped class index for the digit

    Strategy (matching original notebook):
        1. Find the last <|im_end|>.
        2. Walk backwards to find the first answer-digit token.
        3. That is the position to label.
    """
    answer_id_set = set(ANSWER_TOKEN_IDS.tolist())

    # Find last EOT token
    last_eot = -1
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == EOT_TOKEN_ID:
            last_eot = i
            break

    if last_eot == -1:
        return len(input_ids) - 2   # fallback

    # Walk backwards from just before EOT; search up to 20 tokens
    for i in range(last_eot - 1, max(last_eot - 20, -1), -1):
        if input_ids[i] in answer_id_set:
            return i

    return last_eot - 1             # fallback


# ── Dataset ───────────────────────────────────────────────────────────────────
class SALSADataset(Dataset):
    """
    Pre-tokenizes training prompts once at construction (same pattern as
    train_train.py's ReportDataset, but variable-length since we need the
    custom collator to handle per-sample label masks).

    Label placement:
        - labels[ans_pos] = digit_token_id  (the single supervised token)
        - everything else = -100
        - gce_tsallis_loss applies the causal shift so the loss lands on
          the logit that *precedes* the digit, i.e. the model reads <BIRADS>
          and is trained to predict the correct digit next.
    """
    def __init__(self, reports, targets):
        self.items = []
        for report, target in zip(reports, targets):
            full_text = build_full_prompt(report, label=target)
            enc       = tokenizer(
                full_text,
                return_tensors='pt',
                truncation=True,
                max_length=MAX_LEN,
                add_special_tokens=False,
            )
            input_ids = enc['input_ids'][0].tolist()

            labels  = [-100] * len(input_ids)
            ans_pos = find_answer_position(input_ids)
            labels[ans_pos] = input_ids[ans_pos]   # digit token ID

            self.items.append({
                'input_ids':      torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
                'labels':         torch.tensor(labels,    dtype=torch.long),
                'target':         target,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def salsa_collate(batch):
    """Left-pad all tensors in the batch to the longest sequence."""
    max_len = max(x['input_ids'].size(0) for x in batch)
    input_ids_padded, attention_mask_padded, labels_padded, targets = [], [], [], []
    for x in batch:
        pad_len = max_len - x['input_ids'].size(0)
        input_ids_padded.append(
            torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long),
                       x['input_ids']])
        )
        attention_mask_padded.append(
            torch.cat([torch.zeros(pad_len, dtype=torch.long), x['attention_mask']])
        )
        labels_padded.append(
            torch.cat([torch.full((pad_len,), -100, dtype=torch.long), x['labels']])
        )
        targets.append(x['target'])
    return {
        'input_ids':      torch.stack(input_ids_padded),
        'attention_mask': torch.stack(attention_mask_padded),
        'labels':         torch.stack(labels_padded),
        'targets':        torch.tensor(targets, dtype=torch.long),
    }


# ── Model ─────────────────────────────────────────────────────────────────────
class SALSANet(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            use_cache=False,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            trust_remote_code=True,
            attn_implementation='sdpa',
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj'],
            bias='none',
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        )
        self.backbone = get_peft_model(self.backbone, peft_config)
        self.backbone.print_trainable_parameters()

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


# ── GCE Tsallis loss (true SALSA) ─────────────────────────────────────────────
def gce_tsallis_loss(logits_full, labels, targets, class_weights, q=Q):
    """
    True SALSA loss, faithful to the original notebook's SALSATrainer.

    Causal shift (same as the notebook):
        shift_logits = logits[:, :-1, :]   → predictions for tokens 1..T
        shift_labels = labels[:, 1:]       → ground-truth for tokens 1..T

    After the shift, shift_labels at (digit_pos - 1) == digit_token_id.
    We remap digit_token_ids → class indices 0-6, slice logits to 7 classes,
    and apply the Tsallis GCE with hard-label targets + class weighting.
    """
    shift_logits = logits_full[:, :-1, :].contiguous()   # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()             # (B, T-1)

    device  = shift_logits.device
    ans_ids = ANSWER_TOKEN_IDS.to(device)                 # (7,)
    K       = ans_ids.numel()

    # Slice logits to the 7 answer-token columns
    shift_logits_K = shift_logits.index_select(dim=-1, index=ans_ids)  # (B, T-1, 7)

    # Build vocab → class-index remap (same approach as original notebook)
    remap = torch.full((shift_logits.size(-1),), -100, dtype=torch.long, device=device)
    remap[ans_ids] = torch.arange(K, dtype=torch.long, device=device)

    pad_mask          = shift_labels.eq(-100)
    shift_labels_safe = shift_labels.clamp_min(0)
    mapped_labels     = remap[shift_labels_safe]
    mapped_labels[pad_mask] = -100

    # Gather only the one labeled position per sequence
    nonpad_mask  = mapped_labels.ne(-100)
    keep         = nonpad_mask.nonzero(as_tuple=True)

    if keep[0].numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    logits_kept  = shift_logits_K[keep]                    # (N, 7)
    targets_kept = targets[keep[0]]                        # (N,)
    weights_kept = class_weights[targets_kept]             # (N,)

    probs       = torch.softmax(logits_kept.float(), dim=-1)
    t           = q - 1.0
    p_correct   = probs.gather(1, targets_kept.unsqueeze(1)).squeeze(1)
    loss_per_sample = (1.0 / t) * (1.0 - p_correct ** t)

    return (loss_per_sample * weights_kept).mean()


# ── Optimizer ─────────────────────────────────────────────────────────────────
def get_optimizer(model, learning_rate=2e-4, weight_decay=0.01):
    no_decay = ['bias', 'LayerNorm.weight']
    return torch.optim.AdamW(
        [
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'lr': learning_rate, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'lr': learning_rate, 'weight_decay': 0.0},
        ],
        lr=learning_rate, weight_decay=weight_decay,
    )


# ── Oversampling ──────────────────────────────────────────────────────────────
def oversample_rare(reports, targets, oversample_map, seed=SEED):
    extra_r, extra_t = [], []
    for cls, times in oversample_map.items():
        idxs = [i for i, t in enumerate(targets) if t == cls]
        for _ in range(times - 1):
            extra_r.extend([reports[i] for i in idxs])
            extra_t.extend([targets[i] for i in idxs])
        print(f'  Oversample BI-RADS {cls}: {len(idxs)} → {len(idxs)*times} samples')
    combined_r = reports + extra_r
    combined_t = targets + extra_t
    rng   = np.random.default_rng(seed)
    order = rng.permutation(len(combined_r))
    return [combined_r[i] for i in order], [combined_t[i] for i in order]


# ── SALSA inference ───────────────────────────────────────────────────────────
@torch.no_grad()
def salsa_predict(model, reports, batch_size=8):
    """
    Inference prompt ends at '<BIRADS>' (EOT stripped in build_full_prompt).
    logits[:, -1, :] = next-token prediction = the digit token.
    Restrict to 7 answer IDs and argmax → predicted BI-RADS class.
    """
    model.eval()
    all_preds, all_probs = [], []
    ans_ids = ANSWER_TOKEN_IDS.to(DEVICE)

    for i in range(0, len(reports), batch_size):
        batch_reports = reports[i:i + batch_size]
        prompts = [build_full_prompt(r, label=None) for r in batch_reports]
        enc = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=False,
        ).to(DEVICE)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(enc['input_ids'], enc['attention_mask'])
        last_logits = logits[:, -1, :]
        logits_7    = last_logits.index_select(-1, ans_ids)
        probs_7     = torch.softmax(logits_7.float(), dim=-1)
        preds       = probs_7.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs_7.cpu().tolist())

    return all_preds, all_probs


# ── Training loop ─────────────────────────────────────────────────────────────
def train_fold(fold, train_reports, train_targets, val_reports, val_targets):
    print(f'\n{"="*60}')
    print(f'FOLD {fold + 1} / {num_folds}  [Qwen3-8B SALSA]')
    print(f'{"="*60}')
    print(f'  train={len(train_targets)}  val={len(val_targets)}')

    print('Oversampling rare classes:')
    train_reports, train_targets = oversample_rare(
        train_reports, train_targets, OVERSAMPLE
    )
    print(f'  Training samples after oversampling: {len(train_targets)}')

    counts        = np.bincount(train_targets, minlength=NUM_CLASSES)
    weights       = len(train_targets) / (counts + 1e-6)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print('Class weights:')
    for c, w in enumerate(weights):
        print(f'  BI-RADS {c}: count={counts[c]:4d}  weight={w:.2f}')

    print('Tokenizing training data...')
    train_dataset = SALSADataset(train_reports, train_targets)
    print(f'  Done. {len(train_dataset)} samples tokenized.')

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True,
        drop_last=True, num_workers=0,
        collate_fn=salsa_collate,
    )

    model = SALSANet(model_path).to(DEVICE)
    print('Compiling model...')
    model = torch.compile(model, mode='default')
    print('Model compiled.')

    optimizer   = get_optimizer(model, learning_rate=2e-4, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum) * num_epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps,
    )
    scaler     = GradScaler()
    best_f1    = 0.0
    no_improve = 0

    for epoch in range(num_epochs):
        t0 = time.time()

        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        for step, batch in enumerate(tqdm(
                train_loader, desc=f'Fold {fold+1} Ep {epoch+1} Train')):

            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            targets_b      = batch['targets'].to(DEVICE)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss   = gce_tsallis_loss(
                    logits, labels, targets_b, class_weights, q=Q
                ) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            train_loss += loss.item() * grad_accum

        avg_loss = train_loss / len(train_loader)

        print('Running validation...')
        val_preds, _ = salsa_predict(model, val_reports, batch_size=batch_size * 2)

        f1      = f1_score(val_targets, val_preds, average='macro')
        elapsed = time.time() - t0
        print(f'\nFold {fold+1} | Epoch {epoch+1}/{num_epochs} | '
              f'Loss: {avg_loss:.4f} | F1-macro: {f1:.4f} | '
              f'Time: {elapsed/60:.1f}m')

        all_preds_np  = np.array(val_preds)
        all_labels_np = np.array(val_targets)

        cm = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(NUM_CLASSES)))
        print('\nConfusion Matrix (rows=true, cols=pred):')
        print('       ' + '  '.join(f'P{c}  ' for c in range(NUM_CLASSES)))
        for i, row in enumerate(cm):
            print(f'  T{i}  [ {" ".join(f"{v:4d}" for v in row)} ]')

        wrong_mask = all_preds_np != all_labels_np
        print(f'\nWrong predictions: {wrong_mask.sum()} / {len(all_labels_np)}')
        for c in range(NUM_CLASSES):
            class_wrong = wrong_mask & (all_labels_np == c)
            if class_wrong.sum() > 0:
                print(f'  BI-RADS {c}: {class_wrong.sum():3d} wrong')
            else:
                print(f'  BI-RADS {c}:   0 wrong  ✓')

        if f1 > best_f1:
            best_f1    = f1
            no_improve = 0
            model._orig_mod.backbone.save_pretrained(f'backbone_fold_{fold}_best')
            print(f'  ✓ New best — F1-macro: {best_f1:.4f}')
        else:
            no_improve += 1
            print(f'  No improvement ({no_improve}/{PATIENCE})')
            if no_improve >= PATIENCE:
                print(f'  Early stopping at epoch {epoch+1}')
                break

    print(f'\nFold {fold+1} complete. Best F1-macro: {best_f1:.4f}')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f'VRAM free: {torch.cuda.mem_get_info(0)[0]/1e9:.1f} GB')


if __name__ == '__main__':
    print('PyTorch  :', torch.__version__)
    print('GPU      :', torch.cuda.get_device_name(0))
    print(f'VRAM     : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

    torch.manual_seed(SEED)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_csv('/kaggle/input/datasets/rayenv2/train-clean/df_clean.csv')
    ext_df   = pd.read_csv('/kaggle/input/datasets/rayenv2/train-clean/train_external.csv')

    ext_rare = ext_df[ext_df['target'].isin([4, 5, 6])].copy()
    aug_df   = pd.concat([train_df, ext_rare], ignore_index=True)
    aug_df   = aug_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f'Total samples: {len(aug_df)}')
    print(aug_df['target'].value_counts().sort_index())

    all_reports = aug_df['report'].tolist()
    all_targets = aug_df['target'].tolist()

    skf    = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    splits = list(skf.split(all_reports, all_targets))

    print(f'\nStarting from fold {START_FOLD} — Qwen3-8B SALSA...')

    for fold in range(START_FOLD - 1, num_folds):
        train_idx, val_idx = splits[fold]

        fold_train_reports = [all_reports[i] for i in train_idx]
        fold_train_targets = [all_targets[i] for i in train_idx]
        fold_val_reports   = [all_reports[i] for i in val_idx]
        fold_val_targets   = [all_targets[i] for i in val_idx]

        train_fold(
            fold,
            fold_train_reports,
            fold_train_targets,
            fold_val_reports,
            fold_val_targets,
        )

    print('\nAll folds complete.')