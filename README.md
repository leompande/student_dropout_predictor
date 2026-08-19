# AI Student Dropout Predictor

> The purpose of this project is to help advisors to flag out high rist student early, the system output probability of dropout given a student's profile


### 1. Problem Definition
#### Type:
Supervised binary classification

#### Target:
dropout (1=will dropout , 0=will persist)

#### Goal:
Given a student profile, output a probability of dropout so that advisors can flag high-risk students  early.

### 2. Data Collection
Real dropout datasets are sensitive/hard to obtain, so generate_data.py builds synthetic data mimics real data. Twi things have been considered 

 - Feature distributions match real populations - GPA follows  normal distribution, distance from home as exponential
 - The label is not random - it is generated from wieghted logistic formula over the features , plus Gaussian noise , so that relationship that exists in real dropout research hold here too: low attendance , low GPA, low income , and prior feailures , scholarships , high parental education and engagement  