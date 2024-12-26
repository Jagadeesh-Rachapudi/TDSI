# Active Tamper Detection and Localization with Source Identification

This project focuses on audio regeneration using deep learning models. It includes scripts for training models, processing audio data, and embedding/extracting watermarks in audio files.

## Project Structure

### Files and Directories

- **AudioSealModel/**

  - `detector_model.pth`: Placeholder for the detector model weights.
  - `generator_model.pth`: Placeholder for the generator model weights.
  - `SaveModel.py`: Script for encoding and decoding watermarks in audio files.

- **models/**

  - `models.py`: Contains the definition of the `AudioRegenModel` class, which includes the `Encoder` and `Decoder` classes.
  - `train.py`: Script for training the audio regeneration model.

- **utils/**
  - `data_prcocessing.py`: Contains the `AudioSegmentDataset` class and `get_data_loader` function for loading and processing audio data.
  - `data_segrigation.py`: Script for splitting the dataset into training, validation, and test sets.
  - `segmentation.py`: Script for segmenting audio files into uniform segments.
  - `utils.py`: Contains utility functions (currently commented out).

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Torchaudio

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/Jagadeesh-Rachapudi/TDSI
   cd TDSL
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

#### Training the Model

To train the audio regeneration model, run the `train.py` script:

```sh
python src/models/train.py
This `README.md` file now includes the project name and instructions to start training using the `train.py` script.
This `README.md` file now includes the project name and instructions to start training using the `train.py` script.
```
