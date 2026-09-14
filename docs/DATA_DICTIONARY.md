# Data Dictionary

The primary dataset is `data/mental_health_burnout_tech_2026.csv`.

## Identification and profile

| Field | Description | Modeling role |
|---|---|---|
| `employee_id` | Employee record identifier | Identifier; excluded from ML |
| `age` | Age | Candidate feature |
| `gender` | Gender category | Candidate feature; review for fairness before use |
| `country` | Country | Candidate feature; review for fairness before use |
| `job_role` | Job role | Candidate feature |
| `seniority_level` | Seniority category | Candidate feature |
| `years_experience` | Years of professional experience | Candidate feature |
| `years_at_company` | Years at current company | Candidate feature |
| `company_size` | Company size category | Candidate feature |
| `industry` | Industry category | Candidate feature |
| `work_mode` | Remote, hybrid, or on-site arrangement | Candidate feature |

## Work and wellbeing variables

| Field | Description | Modeling role |
|---|---|---|
| `salary_usd` | Salary value in USD | Candidate feature |
| `work_hours_per_week` | Weekly working hours | Candidate feature |
| `meetings_per_day` | Average meetings per day | Candidate feature |
| `team_size` | Team size | Candidate feature |
| `sleep_hours_per_night` | Average nightly sleep | Candidate feature |
| `exercise_days_per_week` | Exercise days per week | Candidate feature |
| `vacation_days_taken` | Vacation days taken | Candidate feature |
| `therapy_access` | Access indicator | Candidate feature; governance review required |
| `uses_therapy` | Usage indicator | Candidate feature; sensitive data |
| `ai_tools_daily` | Daily AI-tool usage indicator/value | Candidate feature |
| `manager_support_score` | Manager support rating | Candidate feature |
| `work_life_balance_score` | Work-life balance rating | Candidate feature |
| `job_satisfaction_score` | Job satisfaction rating | Candidate feature |
| `social_support_score` | Social support rating | Candidate feature |
| `deadline_pressure_score` | Deadline-pressure rating | Candidate feature |
| `autonomy_score` | Autonomy rating | Candidate feature |
| `stress_score` | Stress rating | Candidate feature |

## Mental-health and burnout fields

| Field | Description | Modeling role |
|---|---|---|
| `burnout_score` | Numeric burnout score | **Excluded to prevent target leakage** |
| `burnout_level` | Burnout category | **Target** |
| `phq9_score` | PHQ-9 score | Candidate feature; sensitive data |
| `phq9_category` | PHQ-9 category | Candidate feature; sensitive data |
| `gad7_score` | GAD-7 score | Candidate feature; sensitive data |
| `gad7_category` | GAD-7 category | Candidate feature; sensitive data |
| `seeks_mental_health_support` | Support-seeking indicator | Candidate feature; sensitive data |
| `job_change_intention` | Job-change intention indicator | Candidate feature |

## Important modeling notes

- `employee_id` is an identifier and should not be used as a predictive feature.
- `burnout_score` is excluded because it is closely tied to `burnout_level` and would cause target leakage.
- Sensitive wellbeing fields require careful governance and should not be used casually in employment decisions.
- Categorical variables require appropriate encoding before use in most scikit-learn estimators.
- The data dictionary describes the current dataset; future datasets should be validated against an explicit schema before modeling.
