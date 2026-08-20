"""
Realistc data generation
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 5000


# -  Demographics
age = rng.integers(17, 30,N)
gender = rng.choice(["Male","Female"], size=N,p=[0.4,0.6])


# -- Social economic
family_income_level = rng.choice(["Low","Medium","High"],N,p=[0.35, 0.45,0.20])
parental_education = rng.choice(["None","Secondary","Bachelor","Postgraduate"], size=N,p=[0.15, 0.40, 0.35, 0.10])
distance_from_home_km = np.round(rng.exponential(15,size=N),1)
has_scholarship = rng.choice([0,1],N,p=[0.7,0.3])
part_time_job = rng.choice([0,1],N,p=[0.55,0.45])

#-- Academic History
high_school_gpa = np.clip(rng.normal(3.0,0.5, N), 1.0, 4.0).round(2)
entrance_exam_score = np.clip(rng.normal(65,15,size=N), 20, 100).round(1)
prior_failures = rng.poisson(0.4,N).clip(0,5)

#-- School Engagement
attendance_rate =  np.clip(rng.normal(80,15,size=N),20,100).round(1)
study_hours_per_week = np.clip(rng.normal(10,5, N),0,40).round(1)
current_gpa = np.clip(rng.normal(2.8,0.7,N),0.0,5.0).round(2)
lms_login_freq_per_week = np.clip(rng.normal(8,4,N),0,40).round(0)
extracurricular_activities = rng.choice([0,1],N,p=[0.6,0.4])
counseling_visits = rng.poisson(0.3,N).clip(0,10)
semester = rng.integers(1,9,N)


# --  latent dropout risk

z = (
    -0.9
    + 1.6 * (1 - attendance_rate / 100)
    + 1.4 * (2.5 - current_gpa) / 2.5
    + 0.9 * (family_income_level == "Low").astype(float)
    - 0.5 * (family_income_level == "High").astype(float)
    + 0.35 * prior_failures
    - 0.02 * study_hours_per_week
    - 0.015 * lms_login_freq_per_week
    + 0.5 * part_time_job
    - 0.6 * has_scholarship
    - 0.25 * extracurricular_activities
    + 0.015 * distance_from_home_km.clip(0, 60)
    + 0.20 * counseling_visits
    - 0.10 * (semester >= 5).astype(float)  # later-semester students more committed
    - 0.25 * (parental_education == "Bachelor").astype(float)
    - 0.45 * (parental_education == "Postgraduate").astype(float)
    + rng.normal(0, 0.35, N)  # noise
)

prob_dropout = 1 / (1+np.exp(-z))
dropout = (rng.random(N) < prob_dropout).astype(int)

df = pd.DataFrame(
    {
        "student_id": [f"S{100000+i}" for i in range(N)],
        "age": age,
        "gender": gender,
        "family_income_level": family_income_level,
        "parental_education": parental_education,
        "distance_from_home_km": distance_from_home_km,
        "has_scholarship": has_scholarship,
        "part_time_job": part_time_job,
        "high_school_gpa": high_school_gpa,
        "entrance_exam_score": entrance_exam_score,
        "prior_failures": prior_failures,
        "semester": semester,
        "attendance_rate": attendance_rate,
        "study_hours_per_week": study_hours_per_week,
        "current_gpa": current_gpa,
        "lms_login_freq_per_week": lms_login_freq_per_week,
        "extracurricular_activities": extracurricular_activities,
        "counseling_visits": counseling_visits,
        "dropout": dropout,
    }
)

df.to_csv("data/students.csv", index=False)

print(df.shape)
print(df["dropout"].value_counts(normalize=True))
print(df.head())