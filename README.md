# Synthea Dataset Processing and Bedrock Model Integration

## Overview

This project processes a synthetic healthcare dataset generated using Synthea, a tool for creating high-quality, realistic synthetic patient data. It filters patient records, merges related healthcare information, and exports the data in both CSV and JSONL formats. The processed data can be utilized to generate patient-specific prompts for AI models, with a focus on recommendations for medical care plans or medication adjustments. The Bedrock AI service is used to analyze the data and provide clinical recommendations.

## Features

- **Data Loading**: Reads multiple CSV files (`patients.csv`, `conditions.csv`, etc.) containing synthetic healthcare data.
- **Data Filtering**: Filters records for the first 10 unique patients for focused analysis.
- **Data Transformation**: Cleans and normalizes the dataset:
  - Renames columns for consistency.
  - Merges data into a single dataset.
  - Handles missing data by filling textual fields with "Unknown" and numerical fields with 0.
  - Encodes patient IDs using `LabelEncoder`.
- **Exporting**: Exports the processed dataset in CSV and JSONL formats.
- **Asynchronous JSONL Writing**: Writes JSONL data asynchronously for efficiency.
- **Regex Processing**: Normalizes dates and ensures numeric values are correctly formatted.
- **AI Integration**: Prepares patient data prompts and uses Bedrock AI to generate clinical recommendations.

## Files in the Dataset

- `patients.csv`
- `conditions.csv`
- `medications.csv`
- `encounters.csv`
- `observations.csv`
- `careplans.csv`

## Prerequisites

- **Programming Language**: Python 3.8+
- **Dependencies**:
  - `pandas==2.2.2`
  - `aiofiles==23.2.1`
  - `boto3==1.35.42`
  - `sklearn==1.4.2`
  - `asyncio==4.6.0`
  - `regex==2024.4.16`
  - `json==2.0.9`

## Installation

To set up the environment, install the dependencies:

```bash
pip install pandas==2.2.2 aiofiles==23.2.1 boto3==1.35.42 scikit-learn==1.4.2 regex==2024.4.16
