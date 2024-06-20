# UNet Model for Road Segmentation

This repository contains the implementation of a UNet model for road segmentation using video frames. The project includes scripts for training the UNet model on image datasets and for performing inference on video files, overlaying the predicted masks on the original video frames.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Inference on Video](#inference-on-video)
- [Usage](#usage)
- [Requirements](#requirements)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Road segmentation is an essential task in autonomous driving, where the goal is to segment the road area from the rest of the image. This project leverages the UNet model, a type of convolutional neural network designed for image segmentation tasks. The trained model is capable of detecting road areas in video frames and overlaying the segmented areas on the original frames.

## Dataset

The dataset used for training the UNet model includes road images and their corresponding segmentation masks. The dataset should be organized into separate folders for training images and masks. Each mask should be a grayscale image where different pixel values represent different classes (e.g., road, background).

## Model Architecture

The UNet model consists of an encoder-decoder architecture with skip connections. The encoder captures the context of the input image, while the decoder enables precise localization. The skip connections between the encoder and decoder help in retaining spatial information, which is crucial for segmentation tasks.

### Encoder

- The encoder is composed of a series of convolutional layers with increasing depth, each followed by a max-pooling layer.
- Each convolutional block consists of two convolutional layers with ReLU activation and batch normalization.

### Decoder

- The decoder consists of a series of upsampling layers, each followed by a convolutional block.
- Skip connections from the encoder layers are concatenated with the decoder layers to retain spatial information.

### Output Layer

- The output layer uses a sigmoid activation function for binary segmentation tasks or a softmax activation function for multi-class segmentation tasks.

## Training the Model

To train the UNet model, a separate script is provided. The script loads the training images and masks, preprocesses them, and trains the UNet model using Keras. The training process includes data augmentation techniques to improve the model's generalization capability.

### Steps for Training

1. **Load Training Data**: Load the images and masks from the dataset.
2. **Preprocess Data**: Resize and normalize the images and masks.
3. **Define Model**: Construct the UNet model architecture.
4. **Compile Model**: Compile the model with an appropriate loss function and optimizer.
5. **Train Model**: Train the model using the preprocessed data, with optional validation.

## Inference on Video

The repository also includes a script for performing inference on video files. The script processes each frame of the video, runs the trained UNet model to predict the road segmentation mask, and overlays the mask on the original frame.

### Steps for Video Inference

1. **Load Model**: Load the pre-trained UNet model.
2. **Open Video**: Open the input video file.
3. **Process Frames**: For each frame in the video:
   - Preprocess the frame (resize and normalize).
   - Run inference to get the segmentation mask.
   - Postprocess the mask to match the original frame size.
   - Overlay the mask on the frame.
4. **Save Video**: Save the resulting video with the overlayed segmentation masks.

## Data
Thanks to https://www.cvlibs.net/datasets/kitti/  
Download from https://www.cvlibs.net/download.php?file=data_road.zip

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- tqdm
- ipywidgets (for Jupyter Notebook progress bar)

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to modify any section of this README to better fit your project's specific details or requirements.
