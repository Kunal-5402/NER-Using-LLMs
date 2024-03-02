from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFaceHub
import pandas as pd
import warnings
import csv
import re
import os

warnings.filterwarnings("ignore")

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"


df = pd.read_csv('Processed_Data/final_data.csv')

input_text = ""

with open('input.txt', 'r') as file:
    input_text = file.read();

input_text = input_text + "\n\nInput text: " + df['combined_text'][100] + "\n\nOutput text:"


model = HuggingFaceHub(huggingfacehub_api_token=os.environ['HUGGINGFACEHUB_API_TOKEN'],
                        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        model_kwargs={"temperature":0.6, "max_new_tokens":1000})

response_text = model.predict(input_text, )


# Extracting the relevant section of the text after "Output text:"
relevant_section = re.search(r'Output text:(.+?)(?=Output text:|$)', response_text, re.DOTALL).group(1)

# Extracting the relevant lines containing entity and label
relevant_lines = re.findall(r'{"Entity": ".+?", "Label": ".+?"}', relevant_section)

# Extracting entities and labels and removing the extra "}"
entities_labels = [{"Entity": line.split('"Entity": ')[1].split(', "Label": ')[0].strip().strip('"'), 
                    "Label": line.split(', "Label": ')[1].strip().strip('"')[:-1]}  # Remove the extra "}"
                   for line in relevant_lines]

# Saving entities and labels as CSV
with open('entities_and_labels.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Entity', 'Label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for entity_label in entities_labels:
        writer.writerow(entity_label)


# Open the first CSV file and read its contents
with open('combined_file.csv', 'r', newline='') as file1:
    reader1 = csv.reader(file1)
    data1 = list(reader1)

# Open the second CSV file and read its contents
with open('entities_and_labels.csv', 'r', newline='') as file2:
    reader2 = csv.reader(file2)
    data2 = list(reader2)

# Combine the data from both CSV files
combined_data = data1 + data2

# Write the combined data to a new CSV file
with open('combined_file.csv', 'w', newline='') as combined_file:
    writer = csv.writer(combined_file)
    writer.writerows(combined_data)