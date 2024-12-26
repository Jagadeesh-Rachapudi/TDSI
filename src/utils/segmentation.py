import os
import torch
import torchaudio
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import gc
import os
import random
import json  
from pathlib import Path
import matplotlib.pyplot as plt  
from torch.optim import Adam


def convert_audio_channels(wav: torch.Tensor, channels: int) -> torch.Tensor:
    """
    Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [C, T].
        channels (int): Expected number of channels as output.

    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [C, T].
    """
    if wav.shape[0] == channels:
        return wav
    if channels == 1:
        return wav.mean(dim=0, keepdim=True)
    elif wav.shape[0] == 1:
        return wav.expand(channels, -1)
    else:
        raise ValueError("Unsupported channel conversion")

def segment_and_save(
    input_base_dir: str = "../data_sets/train",
    output_base_dir: str = "../output_segments",
    segment_duration: float = 4.0,
    sample_rate: int = 16000,
    channels: int = 1,
    padding: bool = False
):
    """
    Segment audio files from the input directory into uniform 4-second segments and save the segments.

    Args:
        input_base_dir (str): Base directory containing input audio files (including subdirectories).
        output_base_dir (str): Base directory to save the segmented audio files.
        segment_duration (float): Duration of each segment in seconds.
        sample_rate (int): Target sample rate for the audio.
        channels (int): Target number of audio channels.
        padding (bool): Whether to pad shorter segments to the target duration.
    """
    print("Segmentation started...")

    # Define segment duration in frames
    segment_frames = int(segment_duration * sample_rate)
    
    # Iterate through subdirectories from the base directory
    for year_folder in sorted(os.listdir(input_base_dir)):
        year_path = os.path.join(input_base_dir, year_folder)
        if not os.path.isdir(year_path):
            continue  # Skip non-directory files
        
        output_dir = os.path.join(output_base_dir, year_folder)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for root, _, files in os.walk(year_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                    input_path = os.path.join(root, file)
                    audio, sr = torchaudio.load(input_path)

                    # Resample and convert channels if needed
                    if sr != sample_rate:
                        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio)
                    if audio.shape[0] != channels:
                        audio = convert_audio_channels(audio, channels)

                    # Check if the total duration is a multiple of 4 seconds
                    total_frames = audio.shape[1]
                    if total_frames % segment_frames != 0:
                        print(f"File {file} has a leftover segment of less than {segment_duration} seconds, truncating.")
                        total_frames = total_frames - (total_frames % segment_frames)
                        audio = audio[:, :total_frames]

                    # Uniform segmentation
                    for start_frame in range(0, total_frames, segment_frames):
                        end_frame = start_frame + segment_frames
                        segment = audio[:, start_frame:end_frame]

                        # Save the segment
                        output_filename = f"{Path(file).stem}_segment_{start_frame}.wav"
                        output_path = os.path.join(output_dir, output_filename)
                        torchaudio.save(output_path, segment, sample_rate)

    print("Segmentation ended.")

if __name__ == "__main__":
    # Define your input and output base directories
    input_base_directory = "/scratch/rachapudij.cair.iitmandi/project/raw_audios/en"  # Base path to the audio files
    output_base_directory = "/scratch/rachapudij.cair.iitmandi/project/segemented_data"  # Base path for output data

    # Call the function with your desired parameters
    segment_and_save(
        input_base_dir=input_base_directory,
        output_base_dir=output_base_directory,
        segment_duration=4.0,  
        sample_rate=16000,  # Target sample rate
        channels=1,  # Number of audio channels
        padding=False  # Whether to pad shorter segments
    )
