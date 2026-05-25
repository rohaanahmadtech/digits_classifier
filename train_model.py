import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DigitClassifier:
    def __init__(self, dataset_path='dataset', img_size=(28, 28)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.model = None
        self.class_names = [str(i) for i in range(10)]
        
    def load_and_preprocess_data(self):
        """Load images from folders and preprocess them"""
        images = []
        labels = []
        
        print("Loading dataset...")
        for digit in range(10):
            folder_path = os.path.join(self.dataset_path, str(digit))
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} not found!")
                continue
                
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            
            print(f"Loading {len(image_files)} images for digit {digit}")
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not load {img_path}")
                    continue
                
                # Convert to grayscale
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize image
                img = cv2.resize(img, self.img_size)
                
                # Normalize pixel values to [0, 1]
                img = img.astype('float32') / 255.0
                
                # Invert colors if needed (assuming black text on white background)
                # Most handwritten digits are black on white, but we want white on black for better contrast
                # Comment this line if your digits are already white on black
                img = 1.0 - img
                
                # Add channel dimension
                img = np.expand_dims(img, axis=-1)
                
                images.append(img)
                labels.append(digit)
        
        if len(images) == 0:
            raise ValueError("No images found in dataset! Please check your dataset path and image formats.")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"Total images loaded: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Labels distribution: {np.bincount(labels)}")
        
        return images, labels
    
    def augment_data(self, images, labels):
        """Apply data augmentation to improve model performance"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1
        )
        
        augmented_images = []
        augmented_labels = []
        
        # Keep original images
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        
        # Generate augmented images (1 per original image for speed)
        for i in range(len(images)):
            img = images[i].reshape(1, self.img_size[0], self.img_size[1], 1)
            label = labels[i]
            
            # Generate 1 augmented version per image
            aug_iter = datagen.flow(img, batch_size=1)
            aug_img = next(aug_iter)[0]
            augmented_images.append(aug_img)
            augmented_labels.append(label)
                    
        return np.array(augmented_images), np.array(augmented_labels)
    
    def build_model(self):
        """Build improved CNN model for digit classification"""
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def train(self, epochs=30, batch_size=32, use_augmentation=True):
        """Train the model"""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Apply data augmentation
        if use_augmentation:
            print("Applying data augmentation...")
            X_train_aug, y_train_aug = self.augment_data(X_train, y_train)
            X_train = np.concatenate([X_train, X_train_aug])
            y_train = np.concatenate([y_train, y_train_aug])
            print(f"Augmented training set size: {len(X_train)}")
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, 10)
        y_test_cat = to_categorical(y_test, 10)
        
        # Build model
        self.model = self.build_model()
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        self.model = keras.models.load_model('models/best_model.h5')
        
        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=1)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))
        
        # Save final model
        os.makedirs('models', exist_ok=True)
        self.model.save('models/digit_classifier_final.h5')
        print("\nModel saved to models/digit_classifier_final.h5")
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred_classes)
        
        return history, test_accuracy
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    square=True)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()

def main():
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists('dataset'):
        print("Error: 'dataset' folder not found!")
        print("Please create a 'dataset' folder with subfolders 0-9 containing digit images.")
        return
    
    # Initialize and train classifier
    classifier = DigitClassifier()
    
    try:
        # Train the model
        history, accuracy = classifier.train(epochs=30, batch_size=32, use_augmentation=True)
        print(f"\n✅ Training completed successfully with accuracy: {accuracy:.4f}")
        print(f"📁 Model saved in 'models' directory")
        print(f"📊 Training history saved as 'models/training_history.png'")
        print(f"📈 Confusion matrix saved as 'models/confusion_matrix.png'")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your dataset folder structure is correct:")
        print("   dataset/")
        print("   ├── 0/")
        print("   │   ├── image1.jpg")
        print("   │   └── ...")
        print("   ├── 1/")
        print("   └── ...")
        print("2. Ensure images are in JPEG or PNG format")
        print("3. Check that images can be read by OpenCV")

if __name__ == "__main__":
    main()