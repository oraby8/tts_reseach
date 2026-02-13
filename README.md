

# train 

accelerate launch train.py --input-dir dataset --save-dir ./checkpoints --unfreeze-steps 1000

# inferance 
python inference.py --checkpoint-dir ./checkpoints/best_model --prompt "سلام عليكم و رحمة لله و بركاته" --output-file generated.wav