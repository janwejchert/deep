import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing import image_dataset_from_directory

def main():
    dataset_dir = '/Users/felipedeleon/Desktop/Deep Ler,Project/dataset/Imagenes eco/gwbz3fsgp8-2'
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} not found.")
        return

    batch_size = 32
    img_height = 128
    img_width = 128

    print("Loading training dataset...")
    train_ds = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    print("Loading validation dataset...")
    val_ds = image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Building an Optimized Custom CNN for ECGs
    # Removed invalid augmentations (flips/rotations). 
    # Added BatchNormalization and Dropout for better generalization.
    model = models.Sequential([
        layers.InputLayer(input_shape=(img_height, img_width, 3)),
        layers.Rescaling(1./255),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # Prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    # Model Callbacks
    model_path = '/Users/felipedeleon/Desktop/Deep Ler,Project/ecg_model.h5'
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        model_path, save_best_only=True, monitor='val_accuracy', mode='max'
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        patience=5, monitor='val_accuracy', restore_best_weights=True
    )
    
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6
    )

    epochs = 25 # Increased epochs since we have EarlyStopping
    print(f"Training model for up to {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    print(f"Training complete. Best model saved to {model_path}")

    # Save class names for the app
    with open('/Users/felipedeleon/Desktop/Deep Ler,Project/class_names.txt', 'w') as f:
        for name in class_names:
            f.write(name + '\n')

if __name__ == '__main__':
    main()
