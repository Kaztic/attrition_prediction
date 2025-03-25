import pandas as pd
import numpy as np
import random

# Define number of samples
num_samples = 100000

# Generate unique random employee IDs
employee_ids = random.sample(range(100000, 999999), num_samples)

# Possible roles and their probabilities
roles = ["Engineer", "Manager", "Analyst", "Director", "Consultant"]
role_probabilities = [0.4, 0.2, 0.2, 0.1, 0.1]

# Possible project types and their probabilities
project_types = ["R&D", "Standard", "Client-facing"]
project_probabilities = [0.3, 0.5, 0.2]

# Possible exit reasons and probabilities
exit_reasons = ["Better Offer", "Career Growth", "Work-Life Balance", "Manager Issue", "Personal", "None"]
exit_probabilities = [0.3, 0.25, 0.2, 0.15, 0.08, 0.02]

# Define performance distribution
performance_probs = [0.05, 0.1, 0.15, 0.25, 0.2, 0.15, 0.07, 0.03]  # Must sum to 1
performance_values = [0.5, 0.8, 1.0, 1.2, 1.3, 1.5, 1.7, 1.8]

# Generate synthetic data
data = []
for i in range(num_samples):
    employee_id = employee_ids[i]

    # Generate tenure
    first_10_probs = [0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.08, 0.05, 0.05, 0.02]
    remaining_prob = 1.0 - sum(first_10_probs)
    last_10_probs = [remaining_prob / 10] * 10
    tenure_probs = first_10_probs + last_10_probs
    tenure = np.random.choice(range(1, 21), p=tenure_probs)

    role_history = np.random.choice(roles, p=role_probabilities)
    promotions = max(0, tenure // 3 if tenure > 3 else 0)
    past_performance = np.random.choice(performance_values, p=performance_probs)
    skill_relevance = round(np.random.normal(loc=0.8, scale=0.1), 2)  # Most employees have relevant skills
    training_participation = max(0, int(np.random.poisson(lam=3)))  # Training frequency follows a Poisson distribution
    event_activity = np.random.choice(range(6), p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])  # Event participation likelihood
    feedback_score = round(np.random.normal(loc=3.5, scale=1.0), 2)  # Average feedback rating
    sentiment_score = round(np.random.uniform(-1.0, 1.0), 2)  # Sentiment analysis score varies
    team_attrition_rate = round(np.random.uniform(0.1, 0.5), 2)  # Team attrition percentage
    manager_attrition = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% chance that manager left
    project_type = np.random.choice(project_types, p=project_probabilities)
    location_changes = np.random.choice(range(6), p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])  # Location stability is common
    feedback_360 = round(np.random.normal(loc=3.5, scale=1.0), 2)  # 360-degree feedback
    role_changes = np.random.choice(range(4), p=[0.6, 0.25, 0.1, 0.05])  # Most employees have stable roles
    work_life_balance = round(np.random.normal(loc=3.0, scale=1.0), 2)  # Work-life balance ratings
    leave_pattern = round(np.random.uniform(0.5, 2.0), 2)  # Leave pattern metric varies
    attrition_label = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% attrition rate
    exit_reason = np.random.choice(exit_reasons, p=exit_probabilities)
    
    # Assign manager dialogue and employee response logically
    if attrition_label == 1:
        if exit_reason == "Better Offer":
            manager_dialogue = "We value you. Would a salary adjustment or promotion make you reconsider?"
            employee_response = "I have a better offer with more growth potential."
        elif exit_reason == "Career Growth":
            manager_dialogue = "Are there specific growth opportunities you feel are missing? We can explore options."
            employee_response = "I need more challenging projects and a clear career path."
        elif exit_reason == "Work-Life Balance":
            manager_dialogue = "Would flexible work arrangements help? We can adjust workloads."
            employee_response = "I've been struggling with long hours, and I need a better balance."
        elif exit_reason == "Manager Issue":
            manager_dialogue = "If there are leadership concerns, we can address them confidentially."
            employee_response = "I feel unsupported, and it's affecting my experience."
        elif exit_reason == "Personal":
            manager_dialogue = "I understand. Let us know if there's anything we can do to support you."
            employee_response = "Thank you, but this is a personal decision."
        else:
            manager_dialogue = "We'd love to have you stay. Is there anything we can do?"
            employee_response = "I’m still considering my options."
    else:
        manager_dialogue = "Great job! Keep up the good work. Let me know if you need support."
        employee_response = "Thank you! I’m looking forward to more opportunities here."
    
    data.append([
        employee_id, tenure, role_history, promotions, past_performance, skill_relevance,
        training_participation, event_activity, feedback_score, sentiment_score, team_attrition_rate,
        manager_attrition, project_type, location_changes, feedback_360, role_changes,
        work_life_balance, leave_pattern, attrition_label, exit_reason, manager_dialogue, employee_response
    ])

# Create DataFrame
columns = [
    "EmployeeID", "Tenure", "RoleHistory", "Promotions", "PastPerformance", "SkillRelevance",
    "TrainingParticipation", "EventActivity", "FeedbackScore", "SentimentScore", "TeamAttritionRate",
    "ManagerAttrition", "ProjectType", "LocationChanges", "Feedback360", "RoleChanges",
    "WorkLifeBalance", "LeavePattern", "AttritionLabel", "ExitReason", "ManagerDialogue", "EmployeeResponse"
]
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("synthetic_employee_data.csv", index=False)

print("Enhanced synthetic dataset generated successfully!")
