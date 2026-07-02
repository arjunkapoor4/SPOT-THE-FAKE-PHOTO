train.py:
Loads the dataset, extracts features, trains the Extra Trees classifier, evaluates its performance, and saves the trained model (model.pkl).

predict.py:
Loads the trained model, extracts features from a given image, predicts the probability of it being a screen photo, and reports the prediction latency.

features.py:
Extracts a 22-dimensional feature vector from each image, including frequency-domain features, texture, sharpness, color statistics, edge density, glare information, and image metadata.

augment.py:
Provides data augmentation functions such as image flipping, rotation, zooming, brightness/contrast adjustment, and JPEG recompression to increase dataset diversity during training.

model.pkl:
Stores the trained Extra Trees model and the fitted StandardScaler.