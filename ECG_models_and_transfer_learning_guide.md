# ECG Models & Transfer Learning — Complete Detailed Guide

*Every model architecture and transfer-learning approach worth trying for the
PTB-XL 1D task, in detail, with code and honest pros/cons for your small-data
regime — plus a decision framework for which to actually use.*

---

## 0. Read this first — what "best model" means at YOUR data size

You have 200–2000 labeled patients, a ResNet that overfits (pooled 0.9979 vs CV
0.814), and unstable specificity. **At this scale, bigger/deeper is usually
worse, not better.** The honest ranking of approaches for your situation:

1. **Classical ML on engineered features (LightGBM)** — likely the single best
   model at n=200–2000. See the feature-engineering guide.
2. **Self-supervised pretraining → fine-tune a modest net** — the transfer path
   that genuinely helps, because pretraining uses all ~21k PTB-XL records.
3. **A small, heavily-regularized 1D-CNN** — simpler than your ResNet, less
   overfit.
4. **Deep architectures (ResNet/Inception/Transformer) from scratch** — only
   competitive once you have thousands of labeled records.

This guide covers all of them so you can choose deliberately, not by default.

---

# PART 1 — THE MODEL LANDSCAPE

## 1.1 Classical ML on features (the small-data champion)
LightGBM / XGBoost / logistic regression on the engineered feature matrix.
- **Pros:** best-in-class at hundreds–thousands of samples; tiny overfitting gap;
  interpretable (feature importances); handles missing values; trains in seconds.
- **Cons:** needs the feature-extraction pipeline; caps out if features miss
  something the raw signal contains.
- **Verdict for you:** **start here.** Detailed in the feature-engineering guide.

## 1.2 1D-CNN (the simple deep baseline)
Stacked Conv1D → BatchNorm → ReLU → pooling → dense.
- **Pros:** learns local morphology (QRS shape) automatically; simple; a *small*
  one (2 conv blocks) regularizes well.
- **Cons:** needs more data than classical ML to shine.
- **Verdict:** a small 1D-CNN is a better deep baseline than your ResNet at n=200.

```python
from tensorflow import keras
from tensorflow.keras import layers
def small_cnn(input_shape=(1000,12)):
    inp = keras.Input(input_shape); x = inp
    for f in (32, 64):                       # only 2 blocks — small on purpose
        x = layers.Conv1D(f, 7, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPool1D(2)(x); x = layers.Dropout(0.3)(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out)
```

## 1.3 1D-ResNet (your current model)
Conv blocks with skip connections.
- **Pros:** skip connections ease training of deeper nets; strong on large ECG
  datasets (it's the PTB-XL benchmark workhorse, e.g. `xresnet1d`).
- **Cons:** **over-parameterized for 200 records** — this is your overfitting.
- **Verdict:** keep as a comparison arm; shrink it (fewer blocks/filters) or only
  use it once you've scaled data.

## 1.4 InceptionTime / Inception-1D
Parallel conv kernels of different sizes (captures multi-scale morphology).
- **Pros:** strong general time-series classifier; multi-scale is well-suited to
  ECG (P-wave vs QRS vs T occur at different scales).
- **Cons:** more params; same small-data caveat.
- **Verdict:** worth trying at n≥1000; a good alternative to ResNet.

## 1.5 RNN / LSTM / GRU
Recurrent nets over the signal sequence.
- **Pros:** model temporal dependencies, rhythm.
- **Cons:** slow, harder to train, usually *worse* than 1D-CNNs on ECG
  morphology (the diagnostic info is local, not long-range). Prone to overfit at
  small n.
- **Verdict:** low priority; CNNs beat them here.

## 1.6 CNN + LSTM hybrid
CNN extracts per-window features → LSTM models their sequence.
- **Pros:** combines local morphology + rhythm.
- **Cons:** complex, data-hungry.
- **Verdict:** only if rhythm features matter and you have data.

## 1.7 Temporal Convolutional Network (TCN)
Dilated causal convolutions for long receptive fields without recurrence.
- **Pros:** captures long context, parallelizable, often beats LSTMs.
- **Cons:** another deep model needing data.
- **Verdict:** a reasonable deep alternative; not a priority at n=200.

## 1.8 Transformers / attention models
Self-attention over beats, leads, or patches.
- **Pros:** state-of-the-art on large ECG corpora; attention over leads can
  capture spatial patterns; pairs naturally with self-supervised pretraining.
- **Cons:** **very** data-hungry from scratch — will overfit badly at n=200
  unless pretrained.
- **Verdict:** only via pretraining (Part 2), never from scratch at your scale.

## 1.9 State-space models (S4 / Mamba-style)
Efficient long-sequence models.
- **Pros:** handle long signals efficiently; emerging strong results.
- **Cons:** newer, less tooling, data-hungry.
- **Verdict:** research direction, not for this deadline.

### Architecture summary for your data size

| Model | Data needed | Overfit risk at n=200 | Priority |
|---|---|---|---|
| LightGBM on features | low | very low | **1 (do first)** |
| Small 1D-CNN | low–med | low | 2 |
| SSL-pretrained encoder | uses 21k unlabeled | low (pretrained) | **2 (the transfer win)** |
| 1D-ResNet (shrunk) | med | medium | 3 |
| InceptionTime / TCN | med–high | medium–high | 4 (n≥1000) |
| LSTM / CNN-LSTM | high | high | low |
| Transformer / S4 | very high (or pretrain) | very high from scratch | only pretrained |

---

# PART 2 — TRANSFER LEARNING

## 2.1 The one principle that governs everything
**Transfer learning helps only when the pretraining source has MORE data than
your target.** Your target = 200–2000 labeled. Valid sources: all ~21k PTB-XL
records, or another large ECG corpus. Invalid: a model trained on *fewer* records
than your target (e.g. seeding a big model with your 200-record model — adds
nothing).

## 2.2 Self-supervised pretraining → fine-tune (RECOMMENDED)
Pretrain an encoder on all ~21k PTB-XL signals with **no labels**, then fine-tune
on your labeled subset. The encoder learns ECG morphology from thousands of
records; your scarce labels only place the decision boundary.

**Two main SSL objectives:**

**(a) Contrastive (CLOCS / SimCLR-style)** — two augmented views of the same
signal are positives, everything else negative (NT-Xent loss). Variants use
different *leads* or *time-segments* of the same patient as positives.

**(b) Masked autoencoding** — mask spans of the signal, train the model to
reconstruct them (BERT-for-signals). Good for transformers.

Detailed contrastive code is in the *transfer-learning implementation guide*
(encoder, NT-Xent, augmentation, pretraining loop). The asset is one file:
`ecg_ssl_encoder.h5`.

- **Pros:** uses the full corpus; learns genuine morphology; **best stability
  gain** for your specificity problem (encoder has seen thousands of normals).
- **Cons:** more implementation than classical ML; needs the patient-disjoint
  guard.
- **Verdict:** **the transfer approach to build** if you go beyond LightGBM.

## 2.3 Supervised pretraining on the broader PTB-XL task → fine-tune
Train on the full PTB-XL **multi-label** diagnostic task (all superclasses, all
patients), then fine-tune the backbone on your binary subset.
- **Pros:** uses PTB-XL's existing labels; simpler than SSL; rich supervision.
- **Cons:** commits the backbone to one labeling scheme; needs patient-disjoint
  pretrain/fine-tune split.
- **Verdict:** strong, simpler alternative to SSL.

```python
# 1) pretrain on full PTB-XL multi-label task (patients disjoint from eval set)
backbone = small_cnn_backbone()                 # returns features, no final layer
pretrain_model = add_multilabel_head(backbone, n_classes=5)  # NORM,MI,STTC,CD,HYP
pretrain_model.fit(X_pre, Y_pre_multilabel)     # all PTB-XL minus eval patients
backbone.save_weights("ecg_supervised_backbone.h5")

# 2) fine-tune on your binary subset (same patient-disjoint CV)
clf = add_binary_head(load_backbone("ecg_supervised_backbone.h5", trainable=False))
# phase 1 head only -> phase 2 unfreeze last block, low LR
```

## 2.4 Cross-dataset transfer
Pretrain on a *different* large ECG dataset (e.g. another public corpus), fine-
tune on PTB-XL.
- **Pros:** more data; tests generalization across cohorts.
- **Cons:** domain shift (different devices/populations); lead/sampling mismatch.
- **Verdict:** useful for robustness; mind input-format alignment.

## 2.5 Published pretrained ECG backbones
Load weights someone else pretrained on a large ECG corpus.
- **Pros:** zero pretraining cost if one matches your input (12-lead @ 100 Hz).
- **Cons:** matching weights for your exact architecture/format are often
  unavailable; verify what data they saw (leakage risk if it overlaps PTB-XL eval
  patients).
- **Verdict:** check, but don't count on a clean match.

## 2.6 ImageNet 2D transfer (AVOID for the final model)
Render ECGs as images, use a pretrained 2D CNN.
- **Verdict:** this is the path that caused your source confound. Avoid for the
  1D final model.

## 2.7 Fine-tuning strategies (how to adapt a pretrained backbone)

| Strategy | When |
|---|---|
| **Feature extraction** (freeze backbone, train head only) | smallest data; fastest; least overfit |
| **Progressive unfreezing** (head → last block → more, low LR) | your default; balances adaptation vs overfit |
| **Discriminative LR** (lower LR for early layers, higher for head) | when fine-tuning more layers |
| **Full fine-tuning** | only with lots of labeled data — avoid at n=200 |

```python
# Progressive unfreezing — the safe default
model, backbone = build_classifier(weights="ecg_ssl_encoder.h5", trainable=False)
# phase 1: head only, LR 1e-3
model.compile(keras.optimizers.Adam(1e-3), loss=bce); model.fit(...)
# phase 2: unfreeze last block, LR 1e-5, BatchNorm stays frozen
unfreeze_last_block(backbone)
model.compile(keras.optimizers.Adam(1e-5), loss=bce); model.fit(...)
```

**Always:** BatchNorm frozen during fine-tuning (the bug you fixed once), no
class weights on balanced data, threshold/calibration on a validation slice.

---

# PART 3 — COMBINING MODELS

## 3.1 Feature + deep fusion
Concatenate the engineered feature vector with the CNN/encoder's pooled embedding
before the final dense layer — combines clinical priors with learned features.

```python
sig_in = keras.Input((1000,12)); feat_in = keras.Input((n_features,))
emb = backbone(sig_in)                          # learned embedding
x = layers.Concatenate()([layers.GlobalAveragePooling1D()(emb), feat_in])
out = layers.Dense(1, activation="sigmoid")(layers.Dropout(0.3)(x))
fusion = keras.Model([sig_in, feat_in], out)
```

## 3.2 Stacking ensemble
Train LightGBM(features) and 1D-CNN(signal) separately; average their
probabilities, or train a small meta-learner on their outputs. Their errors are
uncorrelated (different representations), so the ensemble is steadier — directly
helps your specificity variance.

## 3.3 What NOT to do
- Don't seed a big model from your small model (Part 2.1).
- Don't ensemble many copies of the *same* architecture on the *same* features
  (correlated errors, false confidence).

---

# PART 4 — DECISION FRAMEWORK

```
How many LABELED patients do you have?

  ~200 (now)
    -> LightGBM on engineered features (Part 1.1)         [best ROI]
    -> + small 1D-CNN as comparison (Part 1.2)
    -> fix class weights, shrink net (your current bugs)

  ~2000 (after scaling — recommended)
    -> LightGBM on features  AND
    -> SSL-pretrained encoder fine-tuned (Part 2.2)        [transfer win]
    -> stack the two (Part 3.2)

  10k+ (future)
    -> Inception/ResNet/Transformer from scratch viable
    -> or large-scale SSL pretraining

ALWAYS, regardless of size:
  - patient-disjoint splits (GroupKFold by patient_id)
  - pretraining patients disjoint from eval patients
  - BatchNorm frozen in fine-tuning; no class weights on balanced data
  - out-of-fold predictions + bootstrap CIs
  - report improvement only if CI-supported vs baseline
```

---

# PART 5 — HONEST EXPECTATIONS & PRIORITY

- **At n=200, transfer learning is NOT your top lever** — LightGBM on features and
  fixing the class-weight bug are. Transfer learning adds capacity to an already-
  overfitting model unless paired with more data.
- **Transfer learning's real payoff is steadier specificity**, not a higher peak
  AUC — the pretrained encoder having seen thousands of normals stabilizes the
  normal-class boundary your small folds can't.
- **No model fixes label scarcity.** If specificity is still unstable after the
  best model + transfer learning, the honest conclusion is "need more labeled
  data," and that's a finding, not a failure.
- This is single-source PTB-XL — even a great model needs **external validation**
  before any clinical claim.

### Priority order
1. Fix the baseline bug (remove class weights, α=0.5, shrink net).
2. LightGBM on engineered features (feature-engineering guide) — likely your best
   single model.
3. Scale labeled data toward 2000 (patient-disjoint, single-source).
4. SSL pretraining on all PTB-XL → fine-tune (transfer-learning implementation
   guide) — the transfer approach that genuinely helps.
5. Stack the feature model + the deep model for uncorrelated errors.

### Install
```
pip install tensorflow lightgbm scikit-learn neurokit2 wfdb --break-system-packages
```

---

## The core idea

The instinct to reach for transfer learning and deep architectures is natural,
but at a few hundred labeled records the leverage is inverted: inject knowledge
(features) and reuse a large corpus (self-supervised pretraining) rather than
adding depth a tiny dataset can't support. Build the simple, knowledge-rich model
first; add the pretrained encoder when you've scaled the data; and let
patient-disjoint discipline and CI-supported comparison decide what actually
helped.
