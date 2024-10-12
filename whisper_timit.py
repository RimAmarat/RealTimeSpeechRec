from datasets import load_dataset, Audio
from numpy.distutils.command.config import config
from transformers import WhisperProcessor

# Load the SpeechCommands dataset
dataset = load_dataset("speech_commands", 'v0.02', split="train")

# Resample the audio to 16kHz (since Whisper expects audio at 16kHz)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load Whisper processor for tokenization
processor = WhisperProcessor.from_pretrained("openai/whisper-small")


# Define a function to preprocess the data
def preprocess_data(batch):
    # Process audio
    audio = batch["audio"]["array"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

    # Encode the label (spoken word)
    with processor.as_target_processor():
        labels = processor(batch["label"], return_tensors="pt").input_ids

    # Return input features and labels
    return {"input_features": inputs.input_values[0], "labels": labels[0]}


# Preprocess the entire dataset
dataset = dataset.map(preprocess_data, remove_columns=["audio", "label", "file"])

# Create train and validation split
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

