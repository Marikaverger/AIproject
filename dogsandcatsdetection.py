
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Set the path to your training and test data
train_dir = 'training_set/training_set'
test_dir = 'test_set/test_set'

# Set the dimensions of your images
img_width, img_height = 128, 128

# Use VGG16 as a pre-trained base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of augmented data for training and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
)


# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping]
)

# Save the model
model.save('dog_cat_classifier3.h5')

# Evaluate the model
eval_result = model.evaluate(test_generator, steps=len(test_generator))
print("\nEvaluation Result:")
print(f"Test Loss: {eval_result[0]:.4f}")
print(f"Test Accuracy: {eval_result[1]*100:.2f}%")

# Make predictions on the test set
Y_pred = model.predict(test_generator)
y_pred = (Y_pred > 0.5).astype(int)

# Calculate metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:")
print(confusion_matrix(test_generator.classes, y_pred))

print("\nClassification Report:")
print(classification_report(test_generator.classes, y_pred, target_names=['Cat', 'Dog']))
