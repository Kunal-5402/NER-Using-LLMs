import pandas as pd

def extract_columns(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    extracted_data = []
    for line in lines:
        columns = line.split('\t')
        relevant_data = [col.strip() for col in columns]
        extracted_data.append(relevant_data)

    return extracted_data

input_file = 'Data/FLAT_RCL.txt'
extracted_data = extract_columns(input_file)

df = pd.DataFrame(extracted_data, columns=[
    'RECORD_ID', 'CAMPNO', 'MAKETXT', 'MODELTXT', 'YEARTXT', 'MFGCAMPNO', 'COMPNAME',
    'MFGNAME', 'BGMAN', 'ENDMAN', 'RCLTYPECD', 'POTAFF', 'ODATE', 'INFLUENCED_BY',
    'MFGTXT', 'RCDATE', 'DATEA', 'RPNO', 'FMVSS', 'DESC_DEFECT', 'CONSEQUENCE_DEFECT',
    'CORRECTIVE_ACTION', 'NOTES', 'RCL_CMPT_ID', 'MFR_COMP_NAME', 'MFR_COMP_DESC', 'MFR_COMP_PTNO'
])

output_csv_file = 'extracted_data.csv'
df.to_csv(output_csv_file, index=False)

df = pd.read_csv('extracted_data.csv', low_memory = False)

required_columns = ['DESC_DEFECT', 'CONSEQUENCE_DEFECT', 'CORRECTIVE_ACTION', 'NOTES']
required_data = df[required_columns]

required_data.to_csv('required_data.csv', index=False)

df2 = pd.read_csv('required_data.csv')

df2 = df2.apply(lambda x: x.astype(str).str.lower())

df2.to_csv('required_data.csv', index=False)

# Combine text from all columns into a single column
df2['combined_text'] = df2.apply(lambda row: ' '.join(map(str, row)), axis=1)

# Save the DataFrame to a CSV file
df2[['combined_text']].to_csv('final_data.csv', index=False)

d = pd.read_csv('final_data.csv')

# Remove duplicate rows
d = d.drop_duplicates()

# Save the modified DataFrame to the same file
d.to_csv('final_data.csv', index=False)

d = d.sample(500)
d.to_csv('final_data.csv', index=False)