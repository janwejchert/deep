import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def build_multiclass_resnet_1d():
    inputs = layers.Input(shape=(1000, 12), name='12_Lead_ECG_Input')
    
    x = layers.Conv1D(32, kernel_size=15, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Block 1
    shortcut = layers.Conv1D(64, kernel_size=1, padding='same')(x)
    x = layers.Conv1D(64, kernel_size=11, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, kernel_size=11, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Block 2
    shortcut = layers.Conv1D(128, kernel_size=1, padding='same')(x)
    x = layers.Conv1D(128, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, kernel_size=7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D(name='ECG_128d_Embedding')(x)
    
    # Multi-label classifier
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(5, activation='sigmoid', name='MultiLabel_5_Classes')(x)
    
    model = models.Model(inputs, outputs, name="Multiclass_1D_ResNet")
    return model

def main():
    metadata_path = 'data/subset_multiclass_metadata.csv'
    probs_path = 'models/clean_oof_multiclass_probs.npy'
    
    # 1. Generate Confusion Matrices
    if os.path.exists(probs_path) and os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        probs = np.load(probs_path)
        
        y_true = []
        for i, row in df.iterrows():
            y_true.append([row[f'label_{sc}'] for sc in SUPERCLASSES])
        y_true = np.array(y_true)
        
        y_pred = (probs >= 0.5).astype(int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, sc in enumerate(SUPERCLASSES):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'],
                        annot_kws={"size": 14})
            axes[i].set_title(f'{sc} Confusion Matrix', fontsize=14, fontweight='bold')
            axes[i].set_ylabel('True Label', fontsize=12)
            axes[i].set_xlabel('Predicted Label', fontsize=12)
            
        fig.delaxes(axes[-1]) # Remove the 6th empty subplot
        plt.tight_layout()
        cm_path = 'reports/figures/multiclass_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300)
        print(f"Saved {cm_path}")
        
    # 2. Generate Architecture Diagram
    model = build_multiclass_resnet_1d()
    try:
        tf.keras.utils.plot_model(model, to_file='reports/figures/multiclass_architecture.png', 
                                  show_shapes=True, show_layer_names=True, dpi=300)
        print("Saved reports/figures/multiclass_architecture.png")
    except Exception as e:
        print("Could not plot model with Graphviz. Falling back to simple saving.")
        print(e)
        
if __name__ == '__main__':
    main()
