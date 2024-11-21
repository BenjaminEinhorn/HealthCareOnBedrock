
import asyncio
import os
import pandas as pd
import json
import aiofiles



print(os.listdir())
os.chdir(r'E:\Data\Dataset')


# Load CSV files
patients = pd.read_csv('patients.csv')
conditions = pd.read_csv('conditions.csv')
medications = pd.read_csv('medications.csv')
encounters = pd.read_csv('encounters.csv')
observations = pd.read_csv('observations.csv')
careplans = pd.read_csv('careplans.csv')

patients.columns  #Rename Id to PATIENT
patients = patients.rename(columns={'Id':'PATIENT'})
conditions.head()
patients.head()
# Get a list of 10 unique patients
ten_patients = patients['PATIENT'].unique()[:10]  # Get the first 10 unique PATIENT IDs
ten_patients


# Filter each DataFrame by the selected 10 patients
patients = patients[patients['PATIENT'].isin(ten_patients)]
conditions = conditions[conditions['PATIENT'].isin(ten_patients)]
medications = medications[medications['PATIENT'].isin(ten_patients)]
encounters = encounters[encounters['PATIENT'].isin(ten_patients)]
observations = observations[observations['PATIENT'].isin(ten_patients)]
careplans = careplans[careplans['PATIENT'].isin(ten_patients)]

medications.columns
conditions.columns


# Rename 'DESCRIPTION' columns in medications and conditions
medications = medications.rename(columns={'DESCRIPTION': 'MEDICATION_DESCRIPTION'})
medications = medications.rename(columns={'REASONDESCRIPTION': 'MEDICATION_REASON'})
conditions = conditions.rename(columns={'DESCRIPTION': 'CONDITION_DESCRIPTION'})
encounters = encounters.rename(columns={'Id':'ENCOUNTER'})
encounters = encounters.rename(columns={'REASONDESCRIPTION':'REASON_FOR_ENCOUNTER'})
encounters = encounters.rename(columns={'DESCRIPTION': 'ENCOUNTER_DESCRIPTION'})
observations = observations.rename(columns={'DESCRIPTION': 'OBSERVATION_DESCRIPTION'})
observations = observations.rename(columns={'DATE': 'OBSERVATION_DATE'})
careplans = careplans.rename(columns={'DESCRIPTION': 'CAREPLAN_DESCRIPTION'})
careplans = careplans.rename(columns={'REASONDESCRIPTION': 'CAREPLAN_REASON'})




conditions = conditions[['PATIENT','ENCOUNTER','CONDITION_DESCRIPTION']]  #GOOD
medications = medications[['PATIENT','ENCOUNTER','MEDICATION_DESCRIPTION','MEDICATION_REASON']]
encounters = encounters[['PATIENT','ENCOUNTER','ENCOUNTER_DESCRIPTION','REASON_FOR_ENCOUNTER']]
observations = observations[['OBSERVATION_DATE','PATIENT','ENCOUNTER','OBSERVATION_DESCRIPTION','VALUE','CATEGORY']]
careplans = careplans[['PATIENT','ENCOUNTER','CAREPLAN_DESCRIPTION','CAREPLAN_REASON']]




# Merge medications and conditions on 'PATIENT' and 'ENCOUNTER'
merged_data = pd.merge(medications, conditions, on=['PATIENT', 'ENCOUNTER'], how='outer')

# Merge the observations DataFrame with the current merged_data
merged_data = pd.merge(merged_data, observations, on=['PATIENT', 'ENCOUNTER'], how='outer', suffixes=('', '_obs'))

# Merge the careplans DataFrame with the current merged_data
merged_data = pd.merge(merged_data, careplans, on=['PATIENT', 'ENCOUNTER'], how='outer', suffixes=('', '_cp'))

merged_data.columns



# Ensure you're working on a deep copy to avoid SettingWithCopyWarning
merged_data = merged_data.copy()

# Ensure you're working with a copy of the DataFrame to avoid SettingWithCopyWarning
merged_data = merged_data.copy()

# Define a list of the columns that should be filled with 'Unknown' (textual/categorical)
text_columns = ['PATIENT', 'ENCOUNTER', 'MEDICATION_DESCRIPTION', 'MEDICATION_REASON',
                'CONDITION_DESCRIPTION', 'OBSERVATION_DATE', 'OBSERVATION_DESCRIPTION', 'CATEGORY', 
                'CAREPLAN_DESCRIPTION', 'CAREPLAN_REASON']

# Loop through each column and fill NaNs with 'Unknown'
for col in text_columns:
    merged_data[col] = merged_data[col].fillna('Unknown')

# Fill NaN in the numeric column 'VALUE' with 0
merged_data['VALUE'] = merged_data['VALUE'].fillna(0)

#Not sure how much the model is doing math##

# Verify the changes
print(merged_data.head())



from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
le = LabelEncoder()

# Apply LabelEncoder to the 'PATIENT' column
merged_data['PATIENT'] = le.fit_transform(merged_data['PATIENT'])

# Verify the transformation
print(merged_data[['PATIENT']].head())

merged_data.columns

############################################################################
############################################################################




print("DataFrame saved to merged_data.csv successfully.")
# Save the entire DataFrame to the specified CSV file path
file_path = r'E:\Data\synthea_sample_data_csv_nov2021\csv\Finished\merged_data.csv'
merged_data.to_csv(file_path, index=False)

print(f"DataFrame saved to {file_path} successfully.")


import asyncio
import aiofiles
import json
import pandas as pd
#ASYNC LOADING
# Assuming merged_data is already defined as a pandas DataFrame

# Async function to convert data to JSONL format
async def convert_to_jsonl(data, output_file_path):
    async with aiofiles.open(output_file_path, 'w') as outfile:
        for _, row in data.iterrows():
            json_entry = {
                "patient": row['PATIENT'],
                "encounter": row['ENCOUNTER'],
                "medication_description": row['MEDICATION_DESCRIPTION'],
                "medication_reason": row['MEDICATION_REASON'],
                "condition_description": row['CONDITION_DESCRIPTION'],
                "date": row.get('OBSERVATION_DATE', None),  # Use .get() to handle missing column
                "observation_description": row['OBSERVATION_DESCRIPTION'],
                "value": row['VALUE'],
                "category": row['CATEGORY'],
                "careplan_description": row['CAREPLAN_DESCRIPTION'],
                "careplan_reason": row['CAREPLAN_REASON']
            }
            await outfile.write(json.dumps(json_entry) + '\n')

# Define the output path
output_file_path = r'E:\Data\synthea_sample_data_csv_nov2021\csv\Finished\jsonl_merged_data.jsonl'

# Run the async function to write data to JSONL file
asyncio.run(convert_to_jsonl(merged_data, output_file_path))

print(f"Data saved to {output_file_path}")

# Optional: Read the JSONL file back to check its content
jsonl_data = []
with open(output_file_path, 'r') as infile:
    for line in infile:
        jsonl_data.append(json.loads(line.strip()))

# Display the data read back from JSONL file
jsonl_data  # This will contain the list of JSON objects
jsonl_data[0]



#Regex Processing. Converting numerical strings to floats. Normalizing date.


import re

def extract_date(value):
    """Extracts just the date (YYYY-MM-DD) from an ISO 8601 datetime string."""
    match = re.match(r'(\d{4}-\d{2}-\d{2})', value)
    return match.group(1) if match else value  # Return the date part if matched, else return original

# Process jsonl_data entries
for entry in jsonl_data:
    #  Extract date part only
    entry['date'] = extract_date(entry['date'])
    
    # Convert `value` to a float if it's numeric, leave as is if not
    try:
        # Try converting `value` to a float
        entry['value'] = float(entry['value'])
    except ValueError:
        # If conversion fails, leave `value` as is (e.g., for strings like "No" or "Yes")
        pass


jjjson = jsonl_data






import boto3
# Initialize the Bedrock client
client = boto3.client('bedrock-runtime')

# Replace this with your actual model ID (e.g., 'amazon.titan-text-large')

model_id = 'ai21.jamba-1-5-mini-v1:0'

def get_patient_data(patient_id, json_data):
    # Create a list to hold all the records for the patient
    patient_records = []
    
    # Iterate over the list of patient data
    for patient_data in json_data:
        # Check if the 'patient' key matches the provided patient_id
        if patient_data['patient'] == patient_id:
            patient_records.append(patient_data)
    
    # Return all records for the patient, or None if no records are found
    return patient_records if patient_records else None

record = get_patient_data(0,jjjson)

def create_input_prompt(patient_records):
    # Initialize prompt with basic information
    prompt = f"Patient medical history for Patient ID: {patient_records[0]['patient']}:\n"
    
    for record in patient_records:
        # Skip fields with "Unknown" values to avoid clutter
        encounter_details = []
        
        # Add Encounter ID
        if record['encounter'] != "Unknown":
            encounter_details.append(f"- Encounter ID: {record['encounter']}")
        
        # Add Condition
        if record['condition_description'] != "Unknown":
            encounter_details.append(f"- Condition: {record['condition_description']}")
        
        # Add Medication
        if record['medication_description'] != "Unknown":
            encounter_details.append(f"- Medication: {record['medication_description']}")
        
        # Add Medication Reason
        if record['medication_reason'] != "Unknown":
            encounter_details.append(f"- Medication Reason: {record['medication_reason']}")
        
        # Add Observations
        if record['observation_description'] != "Unknown" and record['value'] != "Unknown":
            encounter_details.append(f"- Observations: {record['observation_description']} ({record['value']})")
        
        # Add Care Plan
        if record['careplan_description'] != "Unknown" or record['careplan_reason'] != "Unknown":
            careplan_info = f"- Care Plan: {record['careplan_description']}"
            if record['careplan_reason'] != "Unknown":
                careplan_info += f" ({record['careplan_reason']})"
            encounter_details.append(careplan_info)
        
        # Add Date
        if record['date'] != "Unknown":
            encounter_details.append(f"- Date: {record['date']}")
        
        # Add encounter details to the prompt if any detail exists
        if encounter_details:
            prompt += "\n" + "\n".join(encounter_details) + "\n"
    
    # Add final question to the prompt
    prompt += "\nThis is a collection of records for an individual patient. You are a doctor. What is the recommended care plan or medication adjustment for this patient? "
    
    return prompt


practiceprompt = create_input_prompt(record)
print(practiceprompt)


def invoke_bedrock_model(client, model_id, input_prompt):
    # Use the correct format with 'role' and 'content'
    payload = {
        "messages": [
            {"role": "user", "content": input_prompt}
        ]
    }
    
    # Invoke the model
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType='application/json'
    )
    
    # Read and parse the StreamingBody content
    response_body = response['body'].read().decode('utf-8')  # Read the response body and decode to string
    response_json = json.loads(response_body)  # Now parse the JSON
    
    # Print the entire response to debug
    print("Full response:", response_json)
    
    # Return the actual content from the assistant's message
    return response_json['choices'][0]['message']['content']


practiceprompt
# Call the function with the practice prompt
recommendation = invoke_bedrock_model(client, model_id, practiceprompt)

# Print the model's recommendation
print("Medical Recommendation:", recommendation)
