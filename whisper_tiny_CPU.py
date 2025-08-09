from transformers import WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration
import os
import gc
import torch
import torchaudio
import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cuda"):
    print("Using GPU")
else:
    print("Using CPU")

torchaudio.set_audio_backend("soundfile")

# Set the environment variable
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

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

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Clear cache more frequently
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Preprocess function for each example
def preprocess_function(example):
    waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = example

    # Check if the waveform has more than one channel (e.g., stereo audio)
    if waveform.shape[0] > 1:
        # Convert stereo to mono by averaging across the channels
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Preprocess the input (audio) for Whisper, ensuring correct shape
    inputs = torch.tensor(processor.feature_extractor(waveform.squeeze(0), sampling_rate=sample_rate).input_features)

    # Tokenize the target (transcript/utterance) for Whisper
    labels = processor.tokenizer(utterance).input_ids

    return {
        "input_features": inputs.squeeze(0).to(device),  # Remove extra batch dimension
        "labels": torch.tensor(labels).to(device),
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


# Slice data to test on a smaller dataset size
subset_train_set = torch.utils.data.Subset(train_set, indices=range(2000))
subset_dev_set = torch.utils.data.Subset(dev_set, indices=range(200))
subset_test_set = torch.utils.data.Subset(test_set, indices=range(200))

# Apply the preprocessing to the dataset
processed_trainset = LibriSpeechPreprocessed(subset_train_set)
processed_devset = LibriSpeechPreprocessed(subset_dev_set)
processed_testset = LibriSpeechPreprocessed(subset_test_set)

# Load the Whisper model
model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.get_encoder().requires_grad_(False)

model = model.to(device)

# Clear GPU cache
torch.cuda.empty_cache()
gc.collect()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Clear cache more frequently
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Define a data collator
def custom_data_collator(batch):
    try:
        # Stack the input features and labels into tensors
        input_features = torch.stack([item["input_features"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        return {
            "input_features": input_features.to(device),
            "labels": labels.to(device)
        }
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            raise e
        else:
            raise e


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    dataloader_pin_memory=False,   # Only CPU tensors can be pinned and we are using GPU tensors
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,   # Optional: for simulating larger batch sizes
    fp16=False,                       # Enable mixed precision training
    num_train_epochs=2,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    max_grad_norm=1.0,  #
)

# Initialize the trainer with the datasets (not DataLoaders)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_trainset,  # Passing preprocessed dataset, not DataLoader
    eval_dataset=processed_devset,     # Same for eval dataset
    processing_class=processor,     # Using the processor's tokenizer
    data_collator=custom_data_collator,
)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Clear cache more frequently
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Start fine-tuning the model
trainer.train()

# Save the fine-tuned model and processor
model.save_pretrained("./whisper_finetuned_model")
processor.save_pretrained("./whisper_finetuned_processor")


### EVALUATE ON TEST SET ###
# Define a function to compute metrics for evaluation
def compute_metrics(pred):
    wer_metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Decode the predictions and labels to text
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Compute Word Error Rate (WER) using Hugging Face WER metric
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}


# Initialize a new trainer for evaluation with compute_metrics function
eval_trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=processed_testset,
    processing_class=processor,
    data_collator=custom_data_collator,
    compute_metrics=compute_metrics
)

# Empty memory again before evaluation
torch.cuda.empty_cache()
gc.collect()


# Run evaluation on the test set
results = eval_trainer.evaluate()

# Print evaluation results
print(f"Test Set Results: {results}")