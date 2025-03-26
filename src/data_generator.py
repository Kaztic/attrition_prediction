#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import google.generativeai as genai
import time
import csv
import io
import os

# Configure Gemini API Key
genai.configure(api_key="AIzaSyANO7hs2mZT9RmJotgemri_oMQGo8vJ0p4")

# Define the prompt for structured CSV output
PROMPT = """
Generate a highly realistic, logically consistent employee record for attrition prediction. 
Ensure that all fields are interdependent and reflect real-world career progression, job satisfaction, and attrition trends.

Return data in a **CSV format** with exactly two columns: 'Attribute' and 'Value'.
Each row should contain one attribute-value pair.

Example format:
Attribute,Value
EmployeeID,1001
Tenure,5
RoleHistory,Engineer
...

**Employee Profile: Career & Performance**
    - EmployeeID: Unique identifier.
    - Tenure: Number of years at the company (1-20). Ensure logical career progression.
    - RoleHistory: One of ['Engineer', 'Manager', 'Analyst', 'Director', 'Consultant'], with reasonable transitions based on tenure and career growth.
    - Promotions: Number of promotions received, based on tenure, past performance, and company policies.
    - LastPromotionYearsAgo: Number of years since last promotion (should be <= Tenure and dependent on Promotions).
    - PastPerformance: Performance rating between 0.5 and 1.8, correlating with skill relevance, feedback, and promotions.
    - SkillRelevance: Score between 0.5 and 1.0, indicating how well the employee's skills align with their job role.
    - TrainingParticipation: Number of training programs attended, influenced by ambition, company policies, and role.
    - EventActivity: Number of company events attended, reflecting engagement and workplace involvement.
    - FeedbackScore: Performance feedback score (1-5 scale), aligning with PastPerformance.
    - SentimentScore: Employee sentiment score (-1 to 1), based on work-life balance, performance, and attrition risk.

**Work Environment & Organizational Factors**
    - TeamAttritionRate: Attrition rate in the employee's team (0.1 - 0.5). Higher values if multiple exits have occurred.
    - ManagerAttrition: Whether the manager left recently (0 or 1), affecting employee stability and morale.
    - ProjectType: One of ['R&D', 'Standard', 'Client-facing'], influencing workload and job satisfaction.
    - LocationChanges: Number of times the employee relocated, affecting stability and work-life balance.
    - Feedback360: Score from 1 to 5 on 360-degree feedback (based on peer & leadership reviews).
    - RoleChanges: Number of role transitions (correlated with tenure, ambition, and company needs).
    - WorkLifeBalance: Work-life balance rating (1-5 scale), dependent on project type, workload, and company culture.
    - LeavePattern: Leave frequency (0.5 - 2.0 scale), linked to work stress, personal life, or dissatisfaction.
    - RecognitionCount: Number of recognitions received in the last 2 years (linked to performance & engagement).
    - AwardsReceived: Number of awards received in the last 2 years (correlated with RecognitionCount & Feedback360).

**Attrition Status & Reasons**
    - AttritionLabel: Whether the employee left the company (0 = stayed, 1 = left).
    - ExitReason: If attrition occurred, specify one of:
        - 'Better Offer' (often linked to high performance but low satisfaction).
        - 'Career Growth' (linked to stagnation, lack of promotions, or unmet ambitions).
        - 'Work-Life Balance' (due to excessive workload, high stress, or personal needs).
        - 'Manager Issue' (caused by a toxic environment, poor leadership, or lack of support).
        - 'Personal' (external life changes, health, or family reasons).
        - 'None' (if AttritionLabel = 0).

**Manager-Employee Conversations (Context-Based)**
    - ManagerDialogue: The manager's response based on the employee's situation:
        - If attrition occurs, provide a **realistic retention countermeasure** (e.g., salary adjustment, promotion, role change, flexible work).
        - If the employee stays, provide **constructive feedback** or motivation for career growth.
    - EmployeeResponse: The employee's response, logically based on their profile:
        - If they leave, explain **why the retention effort was insufficient**.
        - If they stay, share **career aspirations, expectations, or appreciation**.

Ensure that:
- **All data points must align logically**. For example:
    - High performance should lead to higher promotions or better offers.
    - Poor work-life balance should correlate with dissatisfaction and higher attrition risk.
    - Manager attrition should impact employee sentiment and retention likelihood.
- Ensure each field contributes meaningfully** to the overall story of the employee's career and decision-making process.
- Data is **structured and formatted correctly as a CSV** with exactly two columns.
- Each field value is logically consistent.
- Attrition reason aligns with the employee's history.
- The manager and employee dialogues align with the context.

**Return only the CSV output** with no explanations or extra text.
"""

# Function to generate employee data in CSV format
def generate_employee_record():
    """
    Uses Gemini LLM to generate structured employee data in CSV format.
    """
    model = genai.GenerativeModel("gemini-2.0-pro-exp")
    response = model.generate_content(PROMPT)
    
    try:
        csv_output = response.text.strip()
        return csv_output
    except:
        return None
    
def transpose_csv(input_file, output_file):
    """Transpose rows and columns in a CSV file."""
    with open(input_file, 'r', newline='') as infile:
        reader = list(csv.reader(infile))
        transposed = list(map(list, zip(*reader)))
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(transposed)

def transform_employee_data(input_file, output_file):
    """
    Transform the raw Gemini-generated CSV data into a structured format.
    """
    # Read the raw file content
    with open(input_file, "r", encoding="utf-8") as file:
        raw_content = file.readlines()
    
    # Remove any unwanted lines (like ```csv) and clean the data
    cleaned_lines = []
    for line in raw_content:
        # Skip empty lines and markdown code blocks
        if line.strip() and not line.startswith("```"):
            # Remove any markdown formatting
            line = line.replace("**", "").replace("*", "")
            cleaned_lines.append(line)
    
    # Save cleaned content to a temporary file
    cleaned_file_path = "cleaned_employee_data.csv"
    with open(cleaned_file_path, "w", encoding="utf-8") as file:
        file.writelines(cleaned_lines)
    
    try:
        # Read the cleaned CSV file without treating first row as header
        df = pd.read_csv(cleaned_file_path, header=None)
        
        # If the first row contains "Attribute,Value", skip it
        if df.iloc[0, 0].strip() == "Attribute":
            df = df.iloc[1:]
        
        # Reset index after potential row removal
        df = df.reset_index(drop=True)
        
        # Create a list to store structured employee data
        employees = []
        current_employee = {}
        
        for _, row in df.iterrows():
            if pd.notna(row[0]) and pd.notna(row[1]):  # Using column indices instead of names
                if row[0] == 'EmployeeID' and current_employee:
                    # Save previous employee record before starting a new one
                    employees.append(current_employee)
                    current_employee = {}
                
                # Add attribute-value pair to the current employee dictionary
                current_employee[row[0]] = row[1]
        
        # Append the last employee record
        if current_employee:
            employees.append(current_employee)
        
        # Convert list of dicts to DataFrame
        structured_df = pd.DataFrame(employees)
        
        # Save the transformed data to a new CSV file without index
        structured_df.to_csv(output_file, index=False)
        print(f"Transformed data saved to: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        print("Raw data preview:")
        print("".join(cleaned_lines[:10]))  # Print first 10 lines for debugging
        raise


# Number of samples to generate
num_samples = 3  # Adjust as needed

# Generate data
records = []
for i in range(num_samples):
    print(f"Generating record {i+1}/{num_samples}...")
    csv_data = generate_employee_record()
    
    if csv_data:
        records.append(csv_data)
    
    time.sleep(1)  # To avoid rate limits

# Combine all generated CSV data
if records:
    full_csv_data = "\n".join(records)

    # Save to CSV file correctly formatted
    output_file = "gemini_generated_employee_data.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        reader = csv.reader(io.StringIO(full_csv_data))
        writer = csv.writer(file)
        
        for row in reader:
            writer.writerow(row)
    
    print(f"Raw data saved to: {os.path.abspath(output_file)}")

    input_file = output_file  # Input generated file
    output_file = "structured_employee_data.csv"  # Output transformed file
    transform_employee_data(input_file, output_file)

    print("✅ Realistic employee dataset successfully generated and saved as CSV!")
else:
    print("❌ No data was generated. Check API access and try again.") 