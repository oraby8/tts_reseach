import argparse
import os
import io
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download and format Najdi dataset for Soprano")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory (default: ./najdi_dataset)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.txt")

    print(f"Downloading dataset to {output_dir}...")
    
    # Load dataset in streaming mode
    try:
        ds = load_dataset("AhmedBadawy11/UAE_100K", split="train", streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Cast audio column to not decode automatically, allowing us to handle bytes directly
    # This is crucial to avoid decoding errors with some backends and gives us raw bytes
    ds = ds.cast_column("audio", Audio(decode=False))

    print("Processing and saving files...")
    
    count = 0
    # Open metadata file for writing
    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        for item in tqdm(ds, desc="Processing samples"):
            try:
                # Audio processing
                audio_data = item['audio']
                original_path = audio_data.get('path', f'sample_{count}.wav')
                
                # Create a file_id (filename without extension)
                original_filename = os.path.basename(original_path)
                file_id = os.path.splitext(original_filename)[0]
                
                # If filename is empty or generic, ensure uniqueness
                if not file_id or file_id == "audio":
                    file_id = f"sample_{count:06d}"
                
                # Transcript
                transcript = item.get('text', '')
                if transcript is None:
                    transcript = ""
                
                # Clean transcript: remove newlines/returns as they break the line-based format
                transcript = transcript.replace('\n', ' ').replace('\r', '').strip()
                
                # Write to metadata: filename|transcript
                meta_f.write(f"{file_id}|{transcript}\n")
                
                # Get audio bytes
                audio_bytes = audio_data['bytes']
                
                # Decode with soundfile from bytes
                data, sr = sf.read(io.BytesIO(audio_bytes))
                
                # Save as WAV to the wavs directory
                # Using PCM_16 subtype ensures compatibility with scipy.io.wavfile used in generate_dataset.py
                output_wav_path = os.path.join(wavs_dir, f"{file_id}.wav")
                sf.write(output_wav_path, data, sr, subtype='PCM_16')
                
                count += 1
                
            except Exception as e:
                print(f"Error processing sample {count}: {e}")
                continue

    print(f"Done! Processed {count} samples.")
    print(f"Dataset saved to: {output_dir}")
    print(f"To generate tokens, run:\npython generate_dataset.py --input-dir {output_dir}")

if __name__ == "__main__":
    main()
