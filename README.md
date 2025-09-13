# Audiobooks Classification - Customer bahavior prediction with TensorFlow

This project is a simple neural network built with **TensorFlow / Keras** to classify audiobook customers based on prepared training, validation, and test datasets.

## 📂 Dataset
The project uses three `.npz` files containing preprocessed data:
- `Audiobooks_data_train.npz`
- `Audiobooks_data_validation.npz`
- `Audiobooks_data_test.npz`

Each file contains:
- `inputs` – features (float32)
- `targets` – labels (int32)

⚠️ The datasets are not included in this repo (due to size / licensing).  
You need to place them in the same folder as the notebook or script.

## 🧠 Model Architecture
The model is defined using `tf.keras.Sequential` and consists of:
- `Flatten` layer (input shape: `(28, 28, 1)`)
- `Dense` hidden layer with **ReLU** activation (50 units)
- `Dense` hidden layer with **ReLU** activation (50 units)
- `Dense` output layer with **Softmax** activation (10 classes)

## ⚙️ Training Configuration
- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Crossentropy  
- **Metrics:** Accuracy  
- **Batch size:** 100  
- **Epochs:** up to 100  
- **Early stopping:** patience = 2  

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/audiobooks-classification.git
   cd audiobooks-classification
2. Place the dataset files (Audiobooks_data_train.npz, Audiobooks_data_validation.npz, Audiobooks_data_test.npz) in the project folder.
python train.py
3. 📊 Evaluation

After training, the model is evaluated on the test dataset:
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print("Test accuracy:", test_accuracy)

4. 🛠 Requirements

Python 3.7+

TensorFlow 2.x

NumPy

You can install dependencies manually:
pip install tensorflow numpy
Or use the included requirements.txt.
