from datasets import Dataset, load_dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import multiprocessing
from itertools import chain

DATA_PATH = 'decompile_samples'
VOCAB_PATH = 'bpe_tokenizer/bpe-bytelevel-vocab.json'
MERGE_PATH = 'bpe_tokenizer/bpe-bytelevel-merges.txt'

print(torch.cuda.is_available())

# Create Roberta Tokenizer from BPE vocab and merge files
tokenizer = RobertaTokenizerFast(vocab_file=VOCAB_PATH, merges_file=MERGE_PATH)
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

# Load Dataset 
ds = load_dataset('text', data_dir='decompile_samples', split='test')
ds = ds.train_test_split(test_size=0.2)

# Preprocess the Dataset
num_proc = multiprocessing.cpu_count()

def group_texts(examples):
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=True
    )
    return tokenized_inputs

tokenized_datasets = ds.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc)

max_length = 512    
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
# shuffle dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34)

print(f"the training dataset contains in total {len(tokenized_datasets['train'])*max_length} tokens")

# Generate model
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability= 0.15
)

config = RobertaConfig(
    vocab_size=50_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)

print(model.num_parameters())

training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=24,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()

trainer.save_model("model2")