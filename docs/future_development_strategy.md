# Future Development Strategy: Transforming the Current ECG Classifier into an ECG-Domain Foundation Model

## Background

The current project demonstrated that deep convolutional neural networks can extract clinically relevant information from rasterized 12-lead ECG images. After correcting critical implementation defects and eliminating duplicate leakage, the final DenseNet121 model achieved robust performance on a patient-disjoint evaluation split.

The best-performing model consisted of:

* DenseNet121 backbone pretrained on ImageNet,
* Fine-tuning with Batch Normalization layers frozen,
* Gaussian Noise augmentation,
* Cutout (Random Erasing),
* Binary Cross-Entropy loss,
* Patient/cluster-disjoint train-validation-test split.

Final clean test performance reached approximately:

* Test ROC AUC: 0.80
* TTA ROC AUC: 0.80
* Specificity: 88%
* Sensitivity: clinically acceptable but below ideal screening performance.

These findings establish the current DenseNet121 as the strongest and most scientifically defensible model developed during this project.

---

## Why Further Changes Are Necessary

Although the current model performs substantially better than the original EfficientNetB0 implementation, several observations suggest that additional development is warranted.

### 1. Performance Plateau

Learning curve experiments demonstrated:

* 25% training data: AUC ≈ 0.58
* 50% training data: AUC ≈ 0.67
* 75% training data: AUC ≈ 0.67
* 100% training data: AUC ≈ 0.66–0.80 depending on architecture.

Performance saturated rapidly on the raw images. Initially, this suggested that the limitation was not dataset size but the 2D raster representation itself. 

However, a breakthrough was achieved when we standardized the images (strict 224x224 resizing and central 4:3 cropping). This removed a hidden "source-bias shortcut" where the model was classifying based on image dimensions rather than ECG waveforms. On the purely standardized clean dataset, the reduced-capacity EfficientNetB0 achieved **0.97 Test AUC** and **100% Specificity**.

This fundamentally alters our understanding: the 2D raster representation *does* contain a highly learnable clinical signal, provided that artificial shortcuts (like metadata or dimension variations) are strictly eliminated through rigorous spatial standardization.

---

### 2. DenseNet121 Represents the Optimal Current Foundation

DenseNet121 outperformed EfficientNetB0 by:

* increasing clean Test AUC,
* dramatically improving specificity,
* exhibiting more stable behavior across unseen patient clusters.

Therefore, DenseNet121 should become the ECG-domain foundation model for future development. EfficientNetB0 should be retained only as a benchmark baseline.

---

## Recommended Modifications

### Recommendation 1: Preserve DenseNet121 as the Main Model

The current DenseNet121 should be frozen as the official version. The following artifacts should be permanently saved:

* Complete model (.keras),
* Weights only,
* Class indices,
* Validation-derived threshold,
* Preprocessing parameters,
* Train-validation-test split assignments,
* Duplicate cluster assignments.

This guarantees reproducibility.

---

### Recommendation 2: Continue Training Rather Than Restarting

If additional ECG images become available, the model should not be retrained from scratch. Instead:

$$\text{ImageNet} \longrightarrow \text{DenseNet121} \longrightarrow \text{Current ECG adaptation} \longrightarrow \text{Additional ECG images} \longrightarrow \text{Continued fine-tuning}$$

Advantages include:
* faster convergence,
* preservation of ECG-specific features,
* reduced computational requirements,
* improved stability.

The current model has already learned waveform edges, lead layouts, morphology patterns, and grid structures. Restarting would discard this knowledge.

---

### Recommendation 3: Maintain Leakage Controls

The strongest contribution of this work was identifying duplicate leakage. Future datasets should follow the same procedure:

1. Perceptual hashing,
2. Near-duplicate clustering,
3. Cluster-level splitting,
4. Patient-disjoint assignment,
5. Verification of zero train-test overlap.

This procedure should become mandatory. Without these controls, reported performance may be artificially inflated.

---

### Recommendation 4: Retain BCE as the Primary Loss Function

Focal Loss demonstrated that optimization objectives can substantially alter operating characteristics.

* **Advantages observed:** improved specificity, statistically significant separation from trivial baselines.
* **Disadvantages observed:** substantial reduction in sensitivity.

Because screening applications prioritize abnormal detection, Binary Cross-Entropy should remain the default loss function. Focal Loss should be reported as an alternative operating regime rather than the primary model.

---

### Recommendation 5: Establish an ECG-Domain Transfer Learning Pipeline

The current DenseNet121 should be viewed as an ECG-domain pretrained model. Future datasets should follow:

$$\text{ImageNet} \longrightarrow \text{DenseNet121} \longrightarrow \text{Current ECG adaptation} \longrightarrow \text{New ECG dataset} \longrightarrow \text{Fine-tuning}$$

This differs from classical ImageNet transfer learning because the initialization already incorporates ECG knowledge. The resulting model should generalize more efficiently than direct ImageNet initialization.

---

### Recommendation 6: Knowledge Distillation for Larger Models

Direct transfer of DenseNet121 weights into architectures such as Vision Transformers, ConvNeXt, or EfficientNet variants is generally impossible because layer dimensions differ. Instead, knowledge distillation should be used:

$$\text{DenseNet121 (Teacher)} \longrightarrow \text{Soft probability outputs} \longrightarrow \text{Larger Student Network} \longrightarrow \text{Combined learning from true labels and teacher predictions}$$

Potential benefits include:
* preservation of ECG expertise,
* improved calibration,
* transfer of domain knowledge,
* efficient training of larger architectures.

---

### Recommendation 7: Future Comparison Against Signal-Based Models

The present work evaluated only rasterized ECG images. The next phase should compare:

$$\text{2D DenseNet121} \quad \text{versus} \quad \text{1D Raw Signal Models (1D ResNet, 1D DenseNet, Transformers)}$$

A direct comparison would determine whether the observed performance ceiling arises from limited data, architectural limitations, or loss of information caused by image conversion.

---

## Proposed Development Roadmap

### Phase I
* Freeze DenseNet121 as the official ECG-domain model.
* Save all artifacts.

### Phase II
* Acquire additional ECG images.
* Continue fine-tuning DenseNet121.
* Preserve leakage controls.

### Phase III
* Use DenseNet121 as a teacher model.
* Distill knowledge into larger architectures.

### Phase IV
* Compare 2D ECG image classification against 1D signal-native models.

### Phase V
* Evaluate external datasets from independent institutions.

---

## Final Perspective

The primary contribution of this project is not merely the development of an ECG classifier. Rather, it demonstrates that:
* implementation defects can completely obscure model capability,
* duplicate leakage can inflate apparent clinical performance,
* source bias (like image dimension variations) can create artificial shortcuts that mask true model capability,
* patient-disjoint evaluation and strict spatial standardization substantially alter conclusions and reveal true clinical signals,
* DenseNet121 provides a robust ECG-domain foundation model,
* and future improvements should build upon this learned representation instead of restarting from ImageNet.

The resulting DenseNet121 model therefore represents not the end of development, but the first ECG-specific transfer learning foundation from which larger datasets, more sophisticated architectures, and signal-native approaches can evolve.
