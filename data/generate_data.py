"""
Realistc data generation
"""

import numpy as np
import pandas as pd

from notebooks.generate_data import counseling_visits

rng = np.random.default_rng(42)
N = 5000


# -  Demographics
age = rng.integers(17, 30,N)
gender = rng.choice(["Male","Female"], size=N,p=[0.4,0.6])


# -- Social economic
family_income_level = rng.integers("Low","Medium","High",N,p=[0.35, 0.45,0.20])
parental_education = rng.choice(["None","Secondary","Bachelor","Postgraduate"], size=N,p=[0.15, 0.40, 0.35, 0.10])
distance_from_home_km = np.round(rng.exponential(15,size=N),1)
has_scholarship = rng.choice([0,1],N,p=[0.7,0.3])
part_time_job = rng.choice([0,1],N,p=[0.55,0.45])

#-- Academic History
high_school_gpa = np.clip(rng.normal(3.0,0.5, N), 1.0, 4.0).round(2)
entrace_exam_score = np.clip(rng.normal(65,15,size=N), 20, 100).round(1)
prior_failures = rng.poisson(0.4,N).clip(0,5)

#-- School Engagement
attandance_rate =  np.clip(rng.normal(80,15,size=N),20,100).round(1)
study_hours_per_week = np.clip(rng.normal(10,5, N),0,40).round(1)
current_gpa = np.clip(rng.normal(2.8,0.7),0.0,4.0).round(2)
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