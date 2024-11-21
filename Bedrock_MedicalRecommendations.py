# Synthea Dataset Processing with AI Integration

# Import necessary libraries
import os
import pandas as pd
import asyncio
import aiofiles
import json
from sklearn.preprocessing import LabelEncoder
import re
import boto3

# Step 1: Set up the working directory and load the dataset
print("Loading dataset...")
os.chdir(r'E:\Data\Dataset')  # Change directory to the dataset location
patients = pd.read_csv('patients.csv')
conditions = pd.read_csv('conditions.csv')
medications = pd.read_csv('medications.csv')
encounters = pd.read_csv('encounters.csv')
observations = pd.read_csv('observations.csv')
careplans = pd.read_csv('careplans.csv')

# Step 2: Filter the first 10 unique patients
print("Filtering dataset for the first 10 unique patients...")
patients = patients.rename(columns={'Id': 'PATIENT'})
ten_patients = patients['PATIENT'].unique()[:10]
patients = patients[patients['PATIENT'].isin(ten_patients)]
conditions = conditions[conditions['PATIENT'].isin(ten_patients)]
medications = medications[medications['PATIENT'].isin(ten_patients)]
encounters = encounters[encounters['PATIENT'].isin(ten_patients)]
observations = observations[observations['PATIENT'].isin(ten_patients)]
careplans = careplans[careplans['PATIENT'].isin(ten_patients)]

# Step 3: Rename columns for consistency
print("Renaming columns for consistency...")
medications = medications.rename(columns={'DESCRIPTION': 'MEDICATION_DESCRIPTION', 'REASONDESCRIPTION': 'MEDICATION_REASON'})
conditions = conditions.rename(columns={'DESCRIPTION': 'CONDITION_DESCRIPTION'})
encounters = encounters.rename(columns={'Id': 'ENCOUNTER', 'REASONDESCRIPTION': 'REASON_FOR_ENCOUNTER', 'DESCRIPTION': 'ENCOUNTER_DESCRIPTION'})
observations = observations.rename(columns={'DESCRIPTION': 'OBSERVATION_DESCRIPTION', 'DATE': 'OBSERVATION_DATE'})
careplans = careplans.rename(columns={'DESCRIPTION': 'CAREPLAN_DESCRIPTION', 'REASONDESCRIPTION': 'CAREPLAN_REASON'})

# Step 4: Select relevant columns
print("Selecting relevant columns...")
conditions = conditions[['PATIENT', 'ENCOUNTER', 'CONDITION_DESCRIPTION']]
medications = medications[['PATIENT', 'ENCOUNTER', 'MEDICATION_DESCRIPTION', 'MEDICATION_REASON']]
encounters = encounters[['PATIENT', 'ENCOUNTER', 'ENCOUNTER_DESCRIPTION', 'REASON_FOR_ENCOUNTER']]
observations = observations[['OBSERVATION_DATE', 'PATIENT', 'ENCOUNTER', 'OBSERVATION_DESCRIPTION', 'VALUE', 'CATEGORY']]
careplans = careplans[['PATIENT', 'ENCOUNTER', 'CAREPLAN_DESCRIPTION', 'CAREPLAN_REASON']]

# Step 5: Merge all datasets
print("Merging datasets...")
merged_data = pd.merge(medications, conditions, on=['PATIENT', 'ENCOUNTER'], how='outer')
merged_data = pd.merge(merged_data, observations, on=['PATIENT', 'ENCOUNTER'], how='outer')
merged_data = pd.merge(merged_data, careplans, on=['PATIENT', 'ENCOUNTER'], how='outer')

# Step 6: Handle missing values
print("Handling missing values...")
text_columns = ['PATIENT', 'ENCOUNTER', 'MEDICATION_DESCRIPTION', 'MEDICATION_REASON',
                'CONDITION_DESCRIPTION', 'OBSERVATION_DATE', 'OBSERVATION_DESCRIPTION', 
                'CATEGORY', 'CAREPLAN_DESCRIPTION', 'CAREPLAN_REASON']
for col in text_columns:
    merged_data[col] = merged_data[col].fillna('Unknown')
merged_data['VALUE'] = merged_data['VALUE'].fillna(0)

# Step 7: Encode patient IDs
print("Encoding patient IDs...")
le = LabelEncoder()
merged_data['PATIENT'] = le.fit_transform(merged_data['PATIENT'])

# Step 8: Save merged data to a CSV file
print("Saving merged data to CSV...")
output_csv_path = r'E:\Data\synthea_sample_data_csv_nov2021\csv\Finished\merged_data.csv'
merged_data.to_csv(output_csv_path, index=False)
print(f"CSV saved at {output_csv_path}")

# Step 9: Convert data to JSONL asynchronously
async def convert_to_jsonl(data, output_file_path):
    print("Converting data to JSONL format...")
    async with aiofiles.open(output_file_path, 'w') as outfile:
        for _, row in data.iterrows():
            json_entry = {
                "patient": row['PATIENT'],
                "encounter": row['ENCOUNTER'],
                "medication_description": row['MEDICATION_DESCRIPTION'],
                "medication_reason": row['MEDICATION_REASON'],
                "condition_description": row['CONDITION_DESCRIPTION'],
                "date": row['OBSERVATION_DATE'],
                "observation_description": row['OBSERVATION_DESCRIPTION'],
                "value": row['VALUE'],
                "category": row['CATEGORY'],
                "careplan_description": row['CAREPLAN_DESCRIPTION'],
                "careplan_reason": row['CAREPLAN_REASON']
            }
            await outfile.write(json.dumps(json_entry) + '\n')

output_jsonl_path = r'E:\Data\synthea_sample_data_csv_nov2021\csv\Finished\jsonl_merged_data.jsonl'
asyncio.run(convert_to_jsonl(merged_data, output_jsonl_path))
print(f"JSONL saved at {output_jsonl_path}")

# Step 10: Define functions to process data for AI input
def extract_date(value):
    """Extracts the date part (YYYY-MM-DD) from ISO 8601 datetime strings."""
    match = re.match(r'(\d{4}-\d{2}-\d{2})', value)
    return match.group(1) if match else value

def get_patient_data(patient_id, json_data):
    """Fetch records for a specific patient."""
    return [record for record in json_data if record['patient'] == patient_id]

def create_input_prompt(patient_records):
    """Create a prompt for the AI model."""
    prompt = f"Patient medical history for Patient ID: {patient_records[0]['patient']}:\n"
    for record in patient_records:
        details = []
        if record['encounter'] != "Unknown":
            details.append(f"- Encounter ID: {record['encounter']}")
        if record['condition_description'] != "Unknown":
            details.append(f"- Condition: {record['condition_description']}")
        if record['medication_description'] != "Unknown":
            details.append(f"- Medication: {record['medication_description']}")
        if record['medication_reason'] != "Unknown":
            details.append(f"- Medication Reason: {record['medication_reason']}")
        if record['observation_description'] != "Unknown" and record['value'] != "Unknown":
            details.append(f"- Observations: {record['observation_description']} ({record['value']})")
        if record['careplan_description'] != "Unknown":
            details.append(f"- Care Plan: {record['careplan_description']}")
        if record['date'] != "Unknown":
            details.append(f"- Date: {record['date']}")
        prompt += "\n".join(details) + "\n"
    prompt += "\nWhat is the recommended care plan or medication adjustment for this patient?"
    return prompt

# Step 11: Connect with Bedrock AI for recommendations
def invoke_bedrock_model(client, model_id, input_prompt):
    """Send prompt to the Bedrock AI model."""
    payload = {"messages": [{"role": "user", "content": input_prompt}]}
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType='application/json'
    )
    response_body = response['body'].read().decode('utf-8')
    return json.loads(response_body)['choices'][0]['message']['content']

print("Process complete. Data ready for AI integration.")



# Sample Run: Processing and AI Integration

# Load the JSONL data to simulate a run
import json

# Step 1: Read the processed JSONL data
print("Loading JSONL data for a sample run...")
jsonl_path = r'E:\Data\synthea_sample_data_csv_nov2021\csv\Finished\jsonl_merged_data.jsonl'
jsonl_data = []
with open(jsonl_path, 'r') as infile:
    for line in infile:
        jsonl_data.append(json.loads(line.strip()))

# Step 2: Fetch data for a specific patient
print("Fetching data for a specific patient...")
patient_id = 0  # Example patient ID
patient_records = get_patient_data(patient_id, jsonl_data)

if not patient_records:
    print(f"No records found for Patient ID: {patient_id}")
else:
    # Step 3: Create a prompt for the AI model
    print(f"Creating input prompt for Patient ID: {patient_id}...")
    input_prompt = create_input_prompt(patient_records)
    print("\nGenerated Input Prompt:\n")
    print(input_prompt)

    # Step 4: Connect with Bedrock AI model
    print("\nConnecting with Bedrock AI model to get recommendations...\n")
    
    # Replace 'your_model_id' and 'client' with your actual Bedrock model ID and initialized client
    client = boto3.client('bedrock-runtime')  # Ensure your AWS credentials are configured
    model_id = 'ai21.jamba-1-5-mini-v1:0'  # Replace with your actual model ID

    try:
        recommendation = invoke_bedrock_model(client, model_id, input_prompt)
        print("\nAI Model Recommendation:")
        print(recommendation)
    except Exception as e:
        print("An error occurred while connecting with the AI model:")
        print(e)

print("\nRun Complete.")
