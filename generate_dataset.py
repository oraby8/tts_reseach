"""
Converts a dataset in LJSpeech format into audio tokens that can be used to train/fine-tune Soprano.
This script creates two JSON files for train and test splits in the provided directory.

Usage:
python generate_dataset.py --input-dir path/to/files

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
"""
import argparse
import pathlib
import random
import json
import os

import soundfile as sf
import torchaudio
import torch
from tqdm import tqdm
from nemo.collections.tts.models import AudioCodecModel

SAMPLE_RATE = 44100
SEED = 42
VAL_PROP = 0.1
VAL_MAX = 512

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    return parser.parse_args()

def main():
    args = get_args()
    input_dir = args.input_dir

    print("Loading model.")
    # Load Nvidia Audio Codec 44kHz
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    print("-----------------")
    encoder = AudioCodecModel.from_pretrained("nvidia/audio-codec-44khz").to(device).eval()
    print("Model loaded.")


    print("Reading metadata.")
    files = []
    with open(f'{input_dir}/metadata.txt', encoding='utf-8') as f:
        data = f.read().split('\n')
        for line in data:
            if not line.strip():
                continue
            filename, transcript = line.split('|', maxsplit=1)
            files.append((filename, transcript))
    print(f'{len(files)} samples located in directory.')

    print("Encoding audio.")
    dataset = []
    for sample in tqdm(files):
        filename, transcript = sample
        try:
            audio_path = f'{input_dir}/wavs/{filename}.wav'
            audio_data, sr = sf.read(audio_path)
            
            # audio_data is [T] or [T, C] and float64
            audio = torch.from_numpy(audio_data).float()
            
            # Convert to [C, T]
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.transpose(0, 1)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
            
        # Mix to mono if necessary
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        # audio is [1, T]. This matches [B, T] for B=1.
        
        audio = audio.to(device)
        audio_len = torch.tensor([audio.shape[1]], device=device)

        with torch.no_grad():
            encoded_tokens, encoded_len = encoder.encode(audio=audio, audio_len=audio_len)
            
        # encoded_tokens shape is [B, K, T] -> [1, 8, T]
        # We want to flatten this to a single sequence of tokens [t1_c1, t1_c2, ..., t1_c8, t2_c1, ...]
        # And apply offsets: codebook_i -> code + i * 1000
        
        # [B, K, T] -> [B, T, K]
        encoded_tokens = encoded_tokens.transpose(1, 2)
        
        # Create offsets: [0, 1000, 2000, ..., 7000]
        offsets = torch.arange(8, device=device) * 1000
        
        # Add offsets (broadcasting over T)
        encoded_tokens = encoded_tokens + offsets
        
        # Flatten: [B, T, K] -> [B, T * K]
        flat_tokens = encoded_tokens.reshape(encoded_tokens.shape[0], -1)
        
        # Convert to list
        tokens_list = flat_tokens.squeeze(0).tolist()
        
        dataset.append([transcript, tokens_list])

    print("Generating train/test splits.")
    random.seed(SEED)
    random.shuffle(dataset)
    num_val = min(int(VAL_PROP * len(dataset)) + 1, VAL_MAX)
    train_dataset = dataset[num_val:]
    val_dataset = dataset[:num_val]
    print(f'# train samples: {len(train_dataset)}')
    print(f'# val samples: {len(val_dataset)}')

    print("Saving datasets.")
    with open(f'{input_dir}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    with open(f'{input_dir}/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)
    print("Datasets saved.")


if __name__ == '__main__':
    main()
