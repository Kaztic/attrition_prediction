feature_keyword_mapping = {
    'LeavePattern': ['absence', 'leave', 'time off', 'vacation', 'sick days', 'PTO', 'break', 'holiday', 'unavailable'],
    'ManagerAttrition': ['turnover', 'manager leaving', 'leadership change', 'resignation', 'attrition', 'replacement', 'transition'],
    'TeamAttritionRate': ['team turnover', 'resignations', 'team changes', 'staffing issues', 'team stability', 'departure'],
    'Promotions': ['promotion', 'career advancement', 'growth', 'raise', 'higher role', 'seniority', 'progression'],
    'ExitReason_Career Growth': ['stagnation', 'no growth', 'dead-end job', 'lack of promotion', 'career limitation', 'no advancement'],
    'FeedbackScore': ['feedback', 'reviews', 'performance rating', 'evaluation', 'manager feedback', 'peer review', 'comments'],
    'EngagementScore': ['motivation', 'satisfaction', 'collaboration', 'teamwork', 'enthusiasm', 'morale', 'participation'],
    'ExitReason_Better Offer': ['salary increase', 'better package', 'higher pay', 'offer elsewhere', 'compensation', 'financial growth'],
    'SkillRelevance': ['skills', 'upskilling', 'training', 'job fit', 'learning curve', 'industry trends'],
    'PastPerformance': ['past ratings', 'previous achievements', 'track record', 'historical performance', 'performance history'],
    'SentimentScore': ['happiness', 'mood', 'stress', 'frustration', 'job satisfaction', 'emotions', 'workplace atmosphere'],
    'Feedback360': ['peer feedback', 'manager evaluation', 'colleague review', 'cross-feedback', 'team assessment'],
    'ExitReason_Work-Life Balance': ['stress', 'burnout', 'workload', 'family time', 'personal life', 'overtime'],
    'ExitReason_Manager Issue': ['bad boss', 'conflict', 'management problem', 'leadership concerns', 'toxic environment'],
    'EventActivity': ['company events', 'team outings', 'social gatherings', 'participation', 'team bonding'],
    'WorkLifeBalance': ['well-being', 'stress level', 'time management', 'remote work', 'hours flexibility'],
    'RoleChanges': ['job switch', 'position change', 'new responsibilities', 'role transition', 'department change'],
    'LastPromotionYearsAgo': ['stagnation', 'career standstill', 'long time since promotion', 'overdue promotion'],
    'RecognitionCount': ['awards', 'appreciation', 'acknowledgment', 'rewards', 'commendation', 'shoutout'],
    'Tenure': ['years of service', 'experience', 'loyalty', 'long-term employee', 'veteran', 'commitment'],
    'ExitReason_Personal': ['health', 'family', 'relocation', 'personal issues', 'life changes'],
    'TrainingParticipation': ['learning', 'training', 'workshops', 'skill development', 'courses', 'certification'],
    'ProjectType_Client-facing': ['customer interaction', 'client handling', 'external stakeholders', 'service role'],
    'AwardsReceived': ['recognition', 'trophy', 'medal', 'certification', 'achievement', 'honor'],
    'ProjectType_Standard': ['regular project', 'business-as-usual', 'internal project', 'standard workflow'],
    'RoleHistory_Analyst': ['data analysis', 'research', 'reporting', 'analytical role'],
    'RoleHistory_Manager': ['team leader', 'management experience', 'supervision', 'project leadership'],
    'RoleHistory_Consultant': ['advisory role', 'consulting', 'strategic guidance', 'client solutions'],
    'LocationChanges': ['office transfer', 'relocation', 'change in worksite', 'move to new region'],
    'ProjectType_R&D': ['innovation', 'research', 'development', 'new technology', 'experimental projects'],
    'Region_REG2': ['regional assignment', 'specific region work', 'location-based duties'],
    'Region_REG3': ['regional assignment', 'specific region work', 'location-based duties'],
    'Business_Unit_BU10': ['specific business unit', 'BU10 assignment', 'department allocation'],
    'RoleHistory_Engineer': ['engineering background', 'technical expertise', 'product development'],
    'Business_Unit_BU8': ['specific business unit', 'BU8 assignment', 'department allocation'],
    'Region_REG4': ['regional assignment', 'specific region work', 'location-based duties'],
    'Business_Unit_BU2': ['specific business unit', 'BU2 assignment', 'department allocation'],
    'Business_Unit_BU1': ['specific business unit', 'BU1 assignment', 'department allocation'],
    'Region_REG1': ['regional assignment', 'specific region work', 'location-based duties'],
    'Business_Unit_BU7': ['specific business unit', 'BU7 assignment', 'department allocation'],
    'Business_Unit_BU4': ['specific business unit', 'BU4 assignment', 'department allocation'],
    'Business_Unit_BU6': ['specific business unit', 'BU6 assignment', 'department allocation'],
    'Business_Unit_BU3': ['specific business unit', 'BU3 assignment', 'department allocation'],
    'Business_Unit_BU9': ['specific business unit', 'BU9 assignment', 'department allocation'],
    'Business_Unit_BU5': ['specific business unit', 'BU5 assignment', 'department allocation'],
    'RoleHistory_Director': ['executive role', 'decision-making', 'corporate leadership'],
    'ExitReason_Manager Issue': ['conflict with manager', 'leadership problem', 'boss trouble'],
    'ProjectType_Client-facing': ['customer relations', 'external engagement', 'service industry'],
    'ProjectType_Consultant': ['strategic role', 'advisory projects', 'consulting assignments'],
    'RoleHistory_Manager': ['people management', 'team supervision', 'senior leadership']
}

retention_strategies = {
    'LeavePattern': [
        "Implement flexible work arrangements to reduce burnout and absenteeism. Introduce wellness programs to improve work-life balance and reduce excessive leaves."
    ],
    'ManagerAttrition': [
        "Develop leadership training and coaching programs to improve manager retention. Create succession planning initiatives to ensure smooth leadership transitions."
    ],
    'TeamAttritionRate': [
        "Enhance team bonding and engagement activities to reduce overall attrition. Conduct retention interviews to address team concerns proactively."
    ],
    'Promotions': [
        "Establish a structured career growth path with clear promotion criteria. Recognize high performers through accelerated promotion tracks."
    ],
    'ExitReason_Career Growth': [
        "Offer cross-functional opportunities to enhance career progression. Develop individual growth plans aligned with employee aspirations."
    ],
    'FeedbackScore': [
        "Implement real-time feedback mechanisms for continuous improvement. Introduce peer recognition systems to boost morale."
    ],
    'EngagementScore': [
        "Enhance employee involvement through participative decision-making. Encourage cross-team collaboration for a stronger workplace culture."
    ],
    'ExitReason_Better Offer': [
        "Offer competitive compensation packages to retain top talent. Provide performance-based bonuses to increase employee satisfaction."
    ],
    'SkillRelevance': [
        "Introduce continuous learning programs to keep skills updated. Develop skill-based mentorship opportunities within the company."
    ],
    'PastPerformance': [
        "Reward consistent high performers with special recognition and career progression plans. Use past performance trends to tailor development opportunities."
    ],
    'SentimentScore': [
        "Implement mental health support programs to address workplace stress. Encourage open communication to understand employee concerns."
    ],
    'ExitReason_Work-Life Balance': [
        "Introduce remote/hybrid work options to improve flexibility. Adjust workloads to maintain a healthier work-life balance."
    ],
    'ExitReason_Manager Issue': [
        "Conduct leadership training to improve managerial effectiveness. Implement anonymous feedback for addressing manager-related concerns."
    ],
    'WorkLifeBalance': [
        "Encourage flexible work hours to improve overall well-being. Offer wellness stipends for fitness and mental health programs."
    ],
    'TrainingParticipation': [
        "Provide incentives for completing skill development programs. Offer career coaching alongside training initiatives."
    ]
}
