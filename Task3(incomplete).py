from transformers import BertTokenizer, BertForTokenClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = True

# Load your custom dataset
custom_dataset = pd.read_csv('ner_data.conll', sep='\t', header=None, names=['token', 'ner_label'])

# Split dataset into train, validation, and test sets
train_data, test_data = train_test_split(custom_dataset, test_size=0.2, random_state=42)
train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(custom_dataset['ner_label'].unique()))

# Drop rows with NaN values
train_data.dropna(inplace=True)
validation_data.dropna(inplace=True)

# Tokenize text data and create input features
def tokenize_text(text):
    tokenized_input = tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    return {
        'input_ids': tokenized_input['input_ids'],
        'attention_mask': tokenized_input['attention_mask'],
        'token_type_ids': tokenized_input.get('token_type_ids', None),  # Include token_type_ids if present in tokenizer output
    }

# Preprocess training and validation data
train_features = [tokenize_text(text) for text in train_data['token']]
validation_features = [tokenize_text(text) for text in validation_data['token']]

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_features,
    eval_dataset=validation_features,
)
trainer.train()


# Evaluation
trainer.evaluate(validation_data.values.tolist())

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')