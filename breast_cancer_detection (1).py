# Breast Cancer Detection with Keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib

# Load dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Features and labels
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Scale the data
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')  # FIXED: softmax instead of sigmoid
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Evaluate
loss, accuracy = model.evaluate(X_test_std, Y_test)
print("Test Accuracy:", accuracy)

# Predict on new data
input_data = (11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
              0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769,
              12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563)

input_df = pd.DataFrame([input_data], columns=X.columns)
input_std = scaler.transform(input_df)

prediction = model.predict(input_std)
prediction_label = np.argmax(prediction)

print("Prediction Probabilities:", prediction)
print("Predicted Class:", prediction_label)

if prediction_label == 0:
    print("The tumor is Malignant")
else:
    print("The tumor is Benign")

# Save the model and scaler
model.save('breast_cancer_model.h5')  # FIXED: Proper way to save Keras model
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler saved successfully!")
