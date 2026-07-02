Spot the Fake Photo – Approach and Results

For this assignment, I developed a lightweight machine learning pipeline to distinguish between a genuine photograph and a photograph of a screen (recaptured image). Instead of using a deep learning model, I adopted a handcrafted feature-based approach because it is computationally efficient, interpretable, and suitable for deployment on devices with limited computational resources.

A custom dataset containing 101 images was created, consisting of 50 real photographs and 51 photographs of screens captured under different lighting conditions and viewing angles. Each image was processed to extract a 22-dimensional feature vector. The extracted features include frequency-domain characteristics (high-frequency energy and moiré patterns), image sharpness, texture statistics, color statistics, edge density, glare information, and image metadata such as image dimensions, aspect ratio, file size, and bytes per pixel.

The extracted features were standardized and used to train an Extra Trees Classifier with 100 trees and balanced class weights. The model was evaluated using an 80/20 stratified train-test split and achieved 100% hold-out accuracy on the collected dataset. To further assess its generalization, I also tested the model on three unseen images that were not part of the training or testing dataset, and it correctly identified all three.

On my laptop (Intel Core i5 13th Gen CPU), the classifier itself performs inference in approximately 7.37 ms. The complete `predict.py` pipeline, including model loading, image preprocessing, feature extraction, and prediction, takes approximately 109–572 ms per image, with most images processed in around 100–200 ms. This makes the solution fast enough for near real-time execution and practical for on-device deployment.

Since the entire pipeline runs locally, the operational cost is effectively zero per image, requiring no cloud infrastructure or external API calls.

With additional development time, I would expand the dataset to include more devices, cameras, screen technologies, and lighting conditions to improve robustness. I would also further optimize the feature extraction pipeline and compare the handcrafted approach with lightweight deep learning models such as MobileNetV3 while maintaining low inference latency.
