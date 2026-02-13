import argparse
import os
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def main():
    parser = argparse.ArgumentParser(description="Expand tokenizer vocabulary with Arabic data")
    parser.add_argument("--base-tokenizer", type=str, default="ekwek/Soprano-80M", help="Base tokenizer name")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to metadata.txt")
    parser.add_argument("--output-dir", type=str, default="soprano_tokenizer_extended", help="Output directory for new tokenizer")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size for the new language")
    args = parser.parse_args()

    print(f"Loading base tokenizer: {args.base_tokenizer}")
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    except Exception as e:
        print(f"Error loading base tokenizer: {e}")
        return

    print(f"Reading text from: {args.metadata_path}")
    texts = []
    try:
        with open(args.metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '|' in line:
                    _, text = line.strip().split('|', 1)
                    texts.append(text)
    except FileNotFoundError:
        print(f"Metadata file not found: {args.metadata_path}")
        return

    if not texts:
        print("No text found in metadata.")
        return
        
    print(f"Found {len(texts)} lines of text.")
    
    # Save texts to a temporary file for training
    corpus_file = "temp_corpus.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))

    print("Training new tokenizer on Arabic data...")
    # Train a new tokenizer using the tokenizers library
    # We use a BPE model, but without ByteLevel pre-tokenization since the base doesn't use it.
    # We'll use Whitespace pre-tokenizer.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.BPEDecoder() # Standard BPE decoder
    # No post processor needed for basic BPE usually, or compatible one.

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>", "<pad>"],
        # We don't use initial_alphabet from ByteLevel
    )
    
    try:
        tokenizer.train([corpus_file], trainer)
    except Exception as e:
        print(f"Error training tokenizer: {e}")
        if os.path.exists(corpus_file):
            os.remove(corpus_file)
        return

    print("Merging vocabularies...")
    # Get the vocabulary from the new tokenizer
    new_vocab = tokenizer.get_vocab()
    new_tokens_list = list(new_vocab.keys())
    
    print(f"Sample new tokens: {new_tokens_list[:10]}")
    
    # Add new tokens to base tokenizer
    num_added = base_tokenizer.add_tokens(new_tokens_list)
    
    print(f"Added {num_added} new tokens to the vocabulary.")
    print(f"Old vocab size: {len(base_tokenizer) - num_added}")
    print(f"New vocab size: {len(base_tokenizer)}")

    
    # Save the extended tokenizer
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Saving extended tokenizer to {args.output_dir}")
    base_tokenizer.save_pretrained(args.output_dir)
    
    # Clean up
    if os.path.exists(corpus_file):
        os.remove(corpus_file)
        
    print("Done.")

if __name__ == "__main__":
    main()
