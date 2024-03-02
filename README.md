# Named Entity Recognition(NER) Using LLMs

## Tasks
- **Task 1**: Feature Extraction from the dataset.
- **Task 2**: Using an open source LLM for entity and label extraction from the unstructured text data in features.
- **Task 3(incomplete in this project)**: Fine tuning LLM on the extracted entities for the NER task.
- **Python-based**: Entirely coded in Python.

## Brief explanation of how NER works

Named Entity Recognition (NER) is a natural language processing task that identifies and categorizes named entities (such as persons, organizations, locations) within text. It involves tokenizing text, extracting relevant features, and using machine learning models to classify words into predefined categories, enabling the extraction of key information from unstructured text data.

![NER Working!](NER_workflow.jpg)

## Installation
Ensure you have Python installed on your system. Then clone this repository:

```bash
git clone [repository-link]
cd [repository-directory]
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Additional Note
To test the program for task 1 and 2 are running fine, I have made a task file 3.1 for file conversion into CoNLL format, to run various tasks:

```bash
command: python task[#].py
```
