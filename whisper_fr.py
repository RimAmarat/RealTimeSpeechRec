from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import os
import torch
import torchaudio

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU cache
torch.cuda.empty_cache()

# Set the directory where your dataset is located
train_dir = "train-clean-100"
dev_dir = "dev-clean"
test_dir = "test-clean"

# Load your local LibriSpeech dataset using torchaudio
train_set = torchaudio.datasets.LIBRISPEECH(train_dir, url="train-clean-100")
dev_set = torchaudio.datasets.LIBRISPEECH(dev_dir, url="dev-clean")
test_set = torchaudio.datasets.LIBRISPEECH(test_dir, url="test-clean")

# Initialize the processor
model_name = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(model_name)

# Preprocess function for each example
def preprocess_function(example):
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = example

    # Check if the waveform has more than one channel (e.g., stereo audio)
    if waveform.shape[0] > 1:
        # Convert stereo to mono by averaging across the channels
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Preprocess the input (audio) for Whisper, ensuring correct shape
    inputs = processor.feature_extractor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features

    # Tokenize the target (transcript/utterance) for Whisper
    labels = processor.tokenizer(utterance).input_ids

    return {
        "input_features": inputs.squeeze(0),  # Remove extra batch dimension
        "labels": torch.tensor(labels)
    }

# Define a custom PyTorch dataset class to preprocess the data
class LibriSpeechPreprocessed(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        return preprocess_function(example)

# Apply the preprocessing to the dataset
processed_trainset = LibriSpeechPreprocessed(train_set)
processed_devset = LibriSpeechPreprocessed(dev_set)
processed_testset = LibriSpeechPreprocessed(test_set)

# Load the Whisper model
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Define a data collator
def custom_data_collator(batch):
    # Stack the input features and labels into tensors
    input_features = torch.stack([item["input_features"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_features": input_features,
        "labels": labels
    }

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    per_device_train_batch_size=1,   # Adjust batch size based on available memory
    per_device_eval_batch_size=1,    # Adjust batch size based on available memory
    gradient_accumulation_steps=2,   # Optional: for simulating larger batch sizes
    fp16=True,                       # Enable mixed precision training
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
)

# Initialize the trainer with the datasets (not DataLoaders)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_trainset,  # Passing preprocessed dataset, not DataLoader
    eval_dataset=processed_devset,     # Same for eval dataset
    tokenizer=processor.tokenizer,     # Using the processor's tokenizer
    data_collator=custom_data_collator,
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Start fine-tuning the model
trainer.train()

# Save the fine-tuned model and processor
model.save_pretrained("./whisper_finetuned_model")
processor.save_pretrained("./whisper_finetuned_processor")