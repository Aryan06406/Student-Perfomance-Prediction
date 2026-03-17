"""
University Student Performance Intelligence System — Final API
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import logging
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# 1. Pipeline class definitions
# ══════════════════════════════════════════════════════════════════════════════
STATIC = [
    'Age', 'Gender', 'SES_Quartile', 'ParentalEducation', 'SchoolType',
    'InternetAccess', 'Race_Asian', 'Race_Black', 'Race_Hispanic', 'Race_Other',
    'Race_Two-or-more', 'Race_White', 'Locale_Rural', 'Locale_Suburban', 'Locale_Town',
    'first_generation_college_flag'
]
HISTORICAL = [
    'Prev_GPA', 'Rolling_GPA_Mean', 'GPA_Trend', 'Performance_Volatility',
    'Cumulative_Academic_Risk', 'Risk_Acceleration',
    'Learning_Resilience', 'Engagement_Drift',
    'Academic_Shock', 'Severe_Academic_Shock'
]
BEHAVIORAL = [
    'AttendanceRate', 'StudyHours', 'Extracurricular', 'PartTimeJob',
    'ParentSupport', 'Romantic', 'FreeTime', 'GoOut',
    'study_environment_score', 'procrastination_tendency',
    'sleep_hours', 'exam_anxiety_score', 'motivation_survey_score',
    'stress_survey_score', 'self_efficacy_score', 'goal_clarity_survey_score',
    'Commute_Strain', 'Social_Distraction', 'Effort_Score',
    'Support_Score', 'Distraction_Index', 'Discipline_Score', 'Burnout_Risk',
    'Support_Gap', 'Cognitive_Load', 'Study_Efficiency',
    'Psychological_Resilience', 'Focus_Index', 'Commute_Burden'
]
DYNAMIC = ['Dynamic_Motivation', 'Dynamic_Confidence', 'Dynamic_Burnout']
IA1_SIGNALS = ['IA1_Score', 'Assignment_Ratio', 'Backlog_Count', 'Academic_Risk', 'Effort_Performance_Gap', 'Engagement_Index', 'High_Performer']
IA2_SIGNALS = ['IA2_Score', 'IA_Average', 'IA_Improvement', 'Performance_Delta', 'Backlog_Pressure']    
SHOCK = [
    'Shock_Magnitude', 'Shock_Event_academic_achievement', 'Shock_Event_bereavement', 'Shock_Event_breakup',
    'Shock_Event_competition_success', 'Shock_Event_family_issue', 'Shock_Event_financial_stress', 'Shock_Event_health_problem',
    'Shock_Event_none', 'Shock_Event_mentorship_start', 'Shock_Event_part_time_job_increase', 'Shock_Event_scholarship_award'
]
INTERVENTION = [
    'Intervention_Active', 'Intervention_Applied_academic_warning', 'Intervention_Applied_counseling', 
    'Intervention_Applied_mentorship', 'Intervention_Applied_peer_tutoring', 'Intervention_Applied_none',
]
STATE = [
    'Archetype_Prev_elite', 'Archetype_Prev_fading_star', 'Archetype_Prev_late_bloomer', 
    'Archetype_Prev_pragmatist', 'Archetype_Prev_vulnerable', 'Archetype_Prev_disengaged',
]
TARGET_GPA = 'Final_Sem_GPA'
TARGET_DROPOUT = 'Dropout_Flag'
TARGET_ARCHETYPE = 'Archetype'
ALWAYS_EXCLUDE = {
    'Student_ID', 'Semester_ID', 'Latent_Ability', TARGET_GPA, TARGET_DROPOUT, TARGET_ARCHETYPE,
    'Archetype_Transition', 'Rolling_GPA_Mean', 'GPA_Trend', 'Academic_Shock', 'Severe_Academic_Shock',
    'Engagement_Drift', 'Cumulative_Academic_Risk', 'Risk_Acceleration', 'Learning_Resilience', 'Performance_Volatility',
}

EXTRA_EXCLUDE = {
    ('before_ia', 'gpa'): {'IA1_Score', 'IA2_Score', 'IA_Average', 'Prev_GPA', 'Assignment_Ratio', 'AttendanceRate', 'High_Performer', 'Effort_Performance_Gap', 'Academic_Risk', 'Backlog_Count', 'Backlog_Pressure'},
    ('after_ia1', 'gpa'): {'IA2_Score', 'IA_Average', 'Prev_GPA', 'AttendanceRate', 'High_Performer', 'Effort_Performance_Gap', 'Academic_Risk'},
    ('after_ia2', 'gpa'): {'Prev_GPA', 'AttendanceRate', 'High_Performer', 'Effort_Performance_Gap', 'Academic_Risk'},
    ('before_ia', 'dropout'): {'IA1_Score', 'IA2_Score', 'IA_Average', 'Prev_GPA', 'Assignment_Ratio', 'AttendanceRate', 'High_Performer', 'Effort_Performance_Gap', 'Academic_Risk', 'Backlog_Count', 'Backlog_Pressure', *SHOCK},
    ('after_ia1', 'dropout'): {'IA2_Score', 'IA_Average', 'Prev_GPA', 'AttendanceRate', 'High_Performer', 'Effort_Performance_Gap', 'Academic_Risk'},
    ('after_ia2', 'dropout'): {'Prev_GPA', 'AttendanceRate', 'High_Performer', 'Effort_Performance_Gap', 'Academic_Risk'},
    ('before_ia', 'archetype'): ((set(STATIC) - {'first_generation_college_flag'}) | set(HISTORICAL) | set(INTERVENTION) | set(IA1_SIGNALS) | set(IA2_SIGNALS) | {'AttendanceRate'}),
    ('after_ia1', 'archetype'): ((set(STATIC) - {'first_generation_college_flag'}) | (set(HISTORICAL) - {'Prev_GPA'}) | set(INTERVENTION) | {'AttendanceRate', 'IA2_Score', 'IA_Average', 'Backlog_Pressure'} | (set(STATE) - {'Archetype_Prev_elite', 'Archetype_Prev_vulnerable'})),
    ('after_ia2', 'archetype'): ((set(STATIC) - {'first_generation_college_flag'}) | set(HISTORICAL) | set(STATE) | {'AttendanceRate', 'Prev_GPA', 'High_Performer', 'Effort_Performance_Gap', 'Academic_Risk', 'Backlog_Count'}),
}

BASE = STATIC + HISTORICAL + BEHAVIORAL + DYNAMIC + SHOCK + INTERVENTION + STATE
SNAPSHOT_POOL = {
    'before_ia' : BASE,
    'after_ia1' : BASE + IA1_SIGNALS,
    'after_ia2' : BASE + IA1_SIGNALS + IA2_SIGNALS,
}
ARCHETYPE_POOL = {
    'before_ia': (BEHAVIORAL + DYNAMIC + SHOCK + ['Prev_GPA'] + STATE),
    'after_ia1': (BEHAVIORAL + DYNAMIC + SHOCK + IA1_SIGNALS + ['Prev_GPA', 'Archetype_Prev_elite', 'Archetype_Prev_vulnerable']),
    'after_ia2': (BEHAVIORAL + DYNAMIC + SHOCK + INTERVENTION + IA2_SIGNALS + ['IA1_Score']),
}

class EncodingTransformer(BaseEstimator, TransformerMixin):
    ORDINAL_MAPS = {'Gender': {'Male': 0, 'Female': 1}, 'ParentalEducation': {'<HS': 0, 'HS': 1, 'SomeCollege': 2, 'Bachelors+': 3}, 'SchoolType': {'Public': 0, 'Private': 1}}
    OHE_PREFIXES = ['Locale', 'Race', 'Archetype_Prev', 'Intervention_Applied', 'Shock_Event']
    def fit(self, X, y=None):
        X = X.copy()
        self._ohe_cols = [c for c in self.OHE_PREFIXES if c in X.columns]
        if self._ohe_cols:
            dummies = pd.get_dummies(X[self._ohe_cols].astype(str), prefix=self._ohe_cols, drop_first=True)
            self._ohe_feature_names = list(dummies.columns)
        else:
            self._ohe_feature_names = []
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col, mapping in self.ORDINAL_MAPS.items():
            if col in X.columns: X[col] = X[col].map(mapping).fillna(-1).astype(int)
        if self._ohe_cols:
            present = [c for c in self._ohe_cols if c in X.columns]
            dummies = pd.get_dummies(X[present].astype(str), prefix=present, drop_first=True)
            dummies = dummies.reindex(columns=self._ohe_feature_names, fill_value=0)
            X = X.drop(columns=present, errors='ignore')
            X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
        return X

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        df = X.copy()
        df['Academic_Performance'] = (df.get('TestScore_Math', pd.Series(0, index=df.index)) + df.get('TestScore_Science', pd.Series(0, index=df.index)) + df.get('TestScore_Reading', pd.Series(0, index=df.index))) / 3
        df['Engagement_Index'] = (df['AttendanceRate'] * 0.5 + df['StudyHours'] * 0.3 + df['Extracurricular'] * 0.2)
        df['Support_Score'] = (df['SES_Quartile'] + df['ParentalEducation'] + df['ParentSupport'])
        df['Distraction_Index'] = (df['Social_Distraction'] * 0.6 + df['FreeTime'] * 0.2 + df['Romantic'] * 0.2)
        df['Discipline_Score'] = (df['AttendanceRate'] * 0.6 + (df['StudyHours'] / 4) * 0.4)
        df['Burnout_Risk'] = (df['stress_survey_score'] * 0.5 - df['sleep_hours'] * 0.3 - df['motivation_survey_score'] * 0.2)
        df['Cognitive_Load'] = (df['StudyHours'] + df['PartTimeJob'] * 4 + df['Extracurricular'] * 2)
        df['Psychological_Resilience'] = (df['self_efficacy_score'] * 0.4 + df['motivation_survey_score'] * 0.3 - df['exam_anxiety_score'] * 0.3)
        df['Focus_Index'] = df['Engagement_Index'] - df['Distraction_Index']
        df['Commute_Burden'] = df['Commute_Strain'] * df['AttendanceRate']
        df['Support_Gap'] = df['Support_Score'] - df['Effort_Score']
        df['Study_Efficiency'] = df['Effort_Score'] / (df['StudyHours'] + 1)

        if 'IA1_Score' in df.columns and 'IA2_Score' in df.columns:
            df['IA_Average'] = (df['IA1_Score'] + df['IA2_Score']) / 2
            df['IA_Improvement'] = df['IA2_Score'] - df['IA1_Score']
            df['Academic_Risk'] = (df['Backlog_Count'] * 0.4 + (10 - df['Prev_GPA']) * 0.3 + (10 - df['IA_Average']) * 0.3)
            df['Effort_Performance_Gap'] = df['Effort_Score'] - df['IA_Average']
            df['Study_Efficiency'] = df['IA_Average'] / (df['StudyHours'] + 1) 
        elif 'IA1_Score' in df.columns:
            df['Academic_Risk'] = (df['Backlog_Count'] * 0.4 + (10 - df['Prev_GPA']) * 0.3 + (10 - df['IA1_Score']) * 0.3)
            df['Effort_Performance_Gap'] = df['Effort_Score'] - df['IA1_Score']
            df['Study_Efficiency'] = df['IA1_Score'] / (df['StudyHours'] + 1)

        if 'GPA' in df.columns: df['High_Performer'] = ((df['GPA'] > 8) & (df['AttendanceRate'] > 80)).astype(int)

        if 'Student_ID' in df.columns and 'Semester_ID' in df.columns:
            df = df.sort_values(['Student_ID', 'Semester_ID'])
            g  = df.groupby('Student_ID')
            if 'Final_Sem_GPA' in df.columns:
                df['Rolling_GPA_Mean'] = g['Final_Sem_GPA'].transform(lambda x: x.rolling(3, min_periods=1).mean())
                df['Performance_Volatility'] = g['Final_Sem_GPA'].transform(lambda x: x.rolling(3, min_periods=2).std())
                if 'Prev_GPA' in df.columns: df['GPA_Trend'] = df['Final_Sem_GPA'] - df['Prev_GPA']
            df['Engagement_Drift'] = g['Engagement_Index'].diff()
            if 'Academic_Risk' in df.columns:
                df['Cumulative_Academic_Risk'] = g['Academic_Risk'].cumsum()
                df['Risk_Acceleration'] = g['Academic_Risk'].diff()
            if 'Backlog_Count' in df.columns: df['Backlog_Pressure'] = g['Backlog_Count'].cumsum()
            if all(c in df.columns for c in ['IA_Improvement', 'GPA_Trend', 'Backlog_Count']):
                df['Learning_Resilience'] = (df['IA_Improvement'] + df['GPA_Trend'] - df['Backlog_Count'])
            if 'Rolling_GPA_Mean' in df.columns and 'Final_Sem_GPA' in df.columns:
                df['Academic_Shock'] = df['Rolling_GPA_Mean'] - df['Final_Sem_GPA']
                df['Severe_Academic_Shock'] = (df['Academic_Shock'] > 1).astype(int)
            temporal_cols = ['Engagement_Drift', 'Performance_Volatility', 'Risk_Acceleration', 'Cumulative_Academic_Risk']
            for col in temporal_cols:
                if col in df.columns: df[col] = (df.groupby('Student_ID')[col].transform(lambda x: x.ffill().bfill()))

        df.drop(columns=['TestScore_Math', 'TestScore_Reading', 'TestScore_Science'], inplace=True, errors='ignore')
        return df

class NumericImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self._num_cols = X.select_dtypes(include='number').columns.tolist()
        self._medians  = X[self._num_cols].median()
        return self
    def transform(self, X, y=None):
        X = X.copy()
        for col in self._num_cols:
            if col in X.columns: X[col] = X[col].fillna(self._medians.get(col, 0))
        return X

class SnapshotSelector(BaseEstimator, TransformerMixin):
    def __init__(self, snapshot: str, task: str):
        self.snapshot, self.task = snapshot, task
    def fit(self, X, y=None):
        pool = (ARCHETYPE_POOL[self.snapshot] if self.task == 'archetype' else SNAPSHOT_POOL[self.snapshot])
        exclude = set(ALWAYS_EXCLUDE) | EXTRA_EXCLUDE.get((self.snapshot, self.task), set())
        self.cols_ = [c for c in pool if c in X.columns and c not in exclude]
        return self
    def transform(self, X, y=None): return X[self.cols_].copy()

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load models
# ══════════════════════════════════════════════════════════════════════════════
TRACK = 'engineering'
MODELS_ROOT = os.path.join(BASE_DIR, 'models')
target_configs = {
    'archetype': os.path.join(MODELS_ROOT, 'archetype'),
    'dropout': os.path.join(MODELS_ROOT, 'dropout', TRACK),
    'gpa': os.path.join(MODELS_ROOT, 'gpa', TRACK)
}
TECHNICAL_DIR = os.path.join(MODELS_ROOT, 'technical')

try:
    PIPELINES = {}
    for target, directory in target_configs.items():
        if not os.path.exists(directory): continue
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                key = filename.replace('.pkl', '')
                PIPELINES[key] = joblib.load(os.path.join(directory, filename))
    
    le_archetype = joblib.load(os.path.join(TECHNICAL_DIR, 'le_archetype.pkl'))
    OPTIMAL_THRESHOLDS = joblib.load(os.path.join(TECHNICAL_DIR, 'optimal_thresholds.pkl'))
except Exception as e:
    logging.error(f"Model loading failed: {e}")

VALID_SNAPSHOTS = {'before_ia', 'after_ia1', 'after_ia2'}
VALID_TASKS     = {'gpa', 'dropout', 'archetype'}
MAX_SEMESTERS   = 8

# ══════════════════════════════════════════════════════════════════════════════
# 3. Carry state
# ══════════════════════════════════════════════════════════════════════════════
SHOCK_MOT_DELTA = {'family_issue': -1.5, 'health_problem': -1.2, 'financial_stress': -1.0, 'breakup': -1.8, 'part_time_job_increase': -0.5, 'competition_success': +2.0, 'scholarship_award': +1.8, 'mentorship_start': +1.5, 'bereavement': -2.0, 'academic_achievement': +1.5, 'none': 0.0}
SHOCK_BUR_DELTA = {'family_issue': +0.8, 'health_problem': +0.6, 'financial_stress': +0.5, 'breakup': +0.4, 'part_time_job_increase': +0.3, 'competition_success': -0.3, 'scholarship_award': -0.2, 'mentorship_start': -0.4, 'bereavement': +1.0, 'academic_achievement': -0.2, 'none': 0.0}

def binary_social_to_continuous(high_social: bool, high_freetime: bool, romantic: bool) -> dict:
    return {'Social_Distraction': 0.70 if high_social else 0.25, 'FreeTime': 4.0 if high_freetime else 2.0, 'Romantic': 1 if romantic else 0}

def advance_carry(carry: dict, sem_data: dict, prev_gpa: float, cur_gpa: float, sem_id: int) -> dict:
    social = float(sem_data.get('Social_Distraction', 0.4))
    intv   = int(sem_data.get('Intervention_Active', 0))
    shock  = sem_data.get('Shock_Event', 'none')
    gpa_d  = cur_gpa - prev_gpa if sem_id > 1 else 0.0
    mot    = carry['motivation']

    mot_delta = (0.10 * gpa_d - 0.05 * float(social > 0.55) * 0.5 + 0.20 * intv + SHOCK_MOT_DELTA.get(shock, 0.0))
    conf_delta = (0.10 * float(gpa_d > 0) - 0.08 * float(gpa_d < -0.5) + intv * 0.05)
    bur_delta  = (0.05 * float(sem_id > 5) + 0.05 * float(social > 0.5) + SHOCK_BUR_DELTA.get(shock, 0.0) - 0.10 * (mot / 10.0))

    return {
        'motivation': float(np.clip(mot + mot_delta, 1.0, 10.0)),
        'confidence': float(np.clip(carry['confidence'] + conf_delta, 1.0, 10.0)),
        'burnout'   : float(np.clip(carry['burnout'] + bur_delta, 0.0, 10.0)),
        'prev_gpa'  : cur_gpa,
        'archetype' : carry.get('archetype', 'pragmatist'),
    }

def bootstrap_carry(static: dict, history: list, rng_noise_std: float = 0.0) -> dict:
    carry = {
        'motivation': float(static.get('motivation_survey_score', 6.0)),
        'confidence': float(static.get('self_efficacy_score', 6.0)),
        'burnout'   : 0.0, 'prev_gpa'  : 0.0, 'archetype' : 'pragmatist',
    }
    for i, h in enumerate(history):
        sem_id, cur_gpa = i + 1, float(h['final_gpa'])
        sem_data = {**h, **binary_social_to_continuous(bool(h.get('high_social', False)), bool(h.get('high_freetime', False)), bool(h.get('Romantic', False)))}
        carry = advance_carry(carry, sem_data, carry['prev_gpa'], cur_gpa, sem_id)
        if rng_noise_std > 0:
            carry['motivation'] = float(np.clip(carry['motivation'] + np.random.normal(0, rng_noise_std), 1, 10))
            carry['confidence'] = float(np.clip(carry['confidence'] + np.random.normal(0, rng_noise_std * 0.5), 1, 10))
    return carry

def build_feature_dict(static: dict, carry: dict, current: dict, sem_id: int) -> dict:
    f = {}
    f.update(static)
    f.update(current)
    f['Dynamic_Motivation'] = carry['motivation']
    f['Dynamic_Confidence'] = carry['confidence']
    f['Dynamic_Burnout']    = carry['burnout']
    f['Prev_GPA']           = carry['prev_gpa']
    f['Archetype_Prev']     = carry['archetype']
    f['Semester_ID']        = sem_id
    f['Student_ID']         = static.get('Student_ID', 'UNKNOWN')
    f['GPA']                = current.get('GPA', carry['prev_gpa'])
    return f

# ══════════════════════════════════════════════════════════════════════════════
# 5. Core prediction
# ══════════════════════════════════════════════════════════════════════════════
def run_prediction(snapshot: str, task: str, model_name: str, features: dict) -> dict:
    key = f"{snapshot}__{task}__{model_name}"
    if key not in PIPELINES: raise KeyError(f"Pipeline '{key}' not found.")
    df, pipe = pd.DataFrame([features]), PIPELINES[key]

    if task == 'gpa':
        pred_gpa = float(np.clip(pipe.predict(df)[0], 0, 10))
        gpa_risk = max(0.0, (5.0 - pred_gpa) / 5.0)
        risk_tier = _risk_tier(gpa_risk)
        drivers = _risk_drivers(features)
        return {
            'task': 'gpa', 'snapshot': snapshot, 'model': model_name,
            'GPA_Predicted': round(pred_gpa, 2), 'GPA_Risk_Score': round(gpa_risk, 3),
            'Risk_Tier': risk_tier, 'Intervention_Level': _intervention_level(risk_tier),
            'Key_Risk_Drivers': drivers, 'Advisory_Notes': [ADVISORY_MAP[d] for d in drivers if d in ADVISORY_MAP],
            'Focus_Areas': list({FOCUS_MAP[d] for d in drivers if d in FOCUS_MAP}),
        }
    elif task == 'dropout':
        threshold = OPTIMAL_THRESHOLDS.get(f"{snapshot}__dropout__{model_name}", {}).get('threshold', 0.30)
        probs = pipe.predict_proba(df)[0]
        drop_prob = float(probs[1])
        total_risk = 0.65 * drop_prob + 0.35 * max(0.0, (5.0 - float(features.get('Prev_GPA', 5.0))) / 5.0)
        risk_tier = _risk_tier(total_risk)
        drivers = _risk_drivers(features)
        return {
            'task': 'dropout', 'snapshot': snapshot, 'model': model_name,
            'Dropout_Probability': round(drop_prob, 4), 'Dropout_Predicted': bool(int(drop_prob >= threshold)),
            'Threshold_Used': round(threshold, 4), 'Composite_Risk_Score': round(total_risk, 3),
            'Risk_Tier': risk_tier, 'Intervention_Level': _intervention_level(risk_tier),
            'Key_Risk_Drivers': drivers, 'Advisory_Notes': [ADVISORY_MAP[d] for d in drivers if d in ADVISORY_MAP],
        }
    elif task == 'archetype':
        probs = pipe.predict_proba(df)[0]
        arch_str = le_archetype.inverse_transform([pipe.predict(df)[0]])[0]
        prob_dict = {le_archetype.inverse_transform([i])[0]: round(float(p), 4) for i, p in enumerate(probs)}
        drivers = _risk_drivers(features)
        return {
            'task': 'archetype', 'snapshot': snapshot, 'model': model_name,
            'Archetype_Predicted': arch_str, 'Archetype_Probabilities': prob_dict,
            'Archetype_Profile': _archetype_profile(arch_str), 'Confidence': round(float(max(probs)), 3),
            'Key_Risk_Drivers': drivers, 'Advisory_Notes': [ADVISORY_MAP[d] for d in drivers if d in ADVISORY_MAP],
        }

def predict_all_tasks(snapshot: str, features: dict) -> dict:
    best = {'gpa': 'lgbm', 'dropout': 'lgbm_recall', 'archetype': 'lgbm'}
    out = {}
    for task, model in best.items():
        try: out[task] = run_prediction(snapshot, task, model, features)
        except Exception as e: out[task] = {'error': str(e)}

    gpa = out.get('gpa', {}).get('GPA_Predicted')
    drop = out.get('dropout', {}).get('Dropout_Probability')
    arch = out.get('archetype', {}).get('Archetype_Predicted')

    comp = round(0.5 * drop + 0.5 * max(0, (5 - gpa) / 5), 3) if (gpa is not None and drop is not None) else None
    tier = _risk_tier(comp) if comp is not None else 'Unknown'

    out['summary'] = {
        'GPA_Predicted': gpa, 'Dropout_Probability': drop, 'Archetype': arch,
        'Composite_Risk_Score': comp, 'Overall_Risk_Tier': tier,
        'Overall_Intervention': _intervention_level(tier) if tier != 'Unknown' else 'UNKNOWN',
    }
    return out

# ══════════════════════════════════════════════════════════════════════════════
# 6. Carry-forward helpers for simulation endpoints
# ══════════════════════════════════════════════════════════════════════════════
ARCHETYPE_DEFAULTS = {
    'elite': {'AttendanceRate':0.92,'StudyHours':4.5,'Effort_Score':0.80,'Social_Distraction':0.20,'Assignment_Ratio':0.95,'exam_anxiety_score':3.0,'stress_survey_score':4.5,'motivation_survey_score':8.5,'sleep_hours':7.5,'FreeTime':2,'Romantic':0,'GoOut':2,'Extracurricular':1,'PartTimeJob':0},
    'fading_star': {'AttendanceRate':0.85,'StudyHours':3.5,'Effort_Score':0.65,'Social_Distraction':0.40,'Assignment_Ratio':0.85,'exam_anxiety_score':5.0,'stress_survey_score':6.0,'motivation_survey_score':6.0,'sleep_hours':6.8,'FreeTime':3,'Romantic':0,'GoOut':3,'Extracurricular':0,'PartTimeJob':0},
    'late_bloomer': {'AttendanceRate':0.80,'StudyHours':3.0,'Effort_Score':0.85,'Social_Distraction':0.30,'Assignment_Ratio':0.80,'exam_anxiety_score':4.5,'stress_survey_score':5.5,'motivation_survey_score':8.0,'sleep_hours':7.2,'FreeTime':3,'Romantic':0,'GoOut':2,'Extracurricular':1,'PartTimeJob':0},
    'disengaged': {'AttendanceRate':0.65,'StudyHours':1.8,'Effort_Score':0.45,'Social_Distraction':0.70,'Assignment_Ratio':0.60,'exam_anxiety_score':6.5,'stress_survey_score':7.0,'motivation_survey_score':3.5,'sleep_hours':6.0,'FreeTime':4,'Romantic':1,'GoOut':4,'Extracurricular':0,'PartTimeJob':1},
    'pragmatist': {'AttendanceRate':0.78,'StudyHours':2.7,'Effort_Score':0.65,'Social_Distraction':0.40,'Assignment_Ratio':0.75,'exam_anxiety_score':4.5,'stress_survey_score':5.8,'motivation_survey_score':6.5,'sleep_hours':6.8,'FreeTime':3,'Romantic':0,'GoOut':3,'Extracurricular':0,'PartTimeJob':0},
    'vulnerable': {'AttendanceRate':0.70,'StudyHours':2.2,'Effort_Score':0.55,'Social_Distraction':0.45,'Assignment_Ratio':0.68,'exam_anxiety_score':7.0,'stress_survey_score':7.5,'motivation_survey_score':5.5,'sleep_hours':6.5,'FreeTime':3,'Romantic':1,'GoOut':3,'Extracurricular':0,'PartTimeJob':1},
    'dropped_out': {'AttendanceRate':0.0,'StudyHours':0.0,'Effort_Score':0.0,'Social_Distraction':0.0,'Assignment_Ratio':0.0,'exam_anxiety_score':0.0,'stress_survey_score':0.0,'motivation_survey_score':0.0,'sleep_hours':0.0,'FreeTime':0,'Romantic':0,'GoOut':0,'Extracurricular':0,'PartTimeJob':0},
}

def impute_future_semester(base_static: dict, base_current: dict, carry: dict, sem_id: int) -> dict:
    """Builds the base structure for a future semester without resetting behaviors."""
    current = base_current.copy()
    
    # Reset semester-specific event flags so they don't 'stick' forever
    current['Shock_Event'] = 'none'
    current['Shock_Magnitude'] = 0.0
    current['Intervention_Applied'] = 'none'
    current['Intervention_Active'] = 0
    current['Backlog_Count'] = 0
    current['GPA'] = carry['prev_gpa']

    return build_feature_dict(base_static, carry, current, sem_id)

def simulate_next_semester(static: dict, carry: dict, prev_features: dict, sem_id: int, mode: str = 'trajectory') -> dict:
    """Evolves the student's behavior dynamically over time."""
    current = prev_features.copy()

    # 1. Natural Drift (Senioritis): Attendance slowly drops by 1-2% after Sem 4
    drift_factor = 0.99 if sem_id > 4 else 1.0
    current['AttendanceRate'] = np.clip(current.get('AttendanceRate', 0.8) * drift_factor, 0.4, 1.0)

    # 2. Behavioral Feedback: Motivation boosts study hours, Burnout drops them
    gpa_improvement = carry.get('motivation', 5.0) / 10.0  # Scale 0.1 to 1.0
    burnout_penalty = carry.get('burnout', 0.0) / 20.0     # Penalty grows as burnout grows
    
    current['StudyHours'] = np.clip(
        current.get('StudyHours', 3.0) + (gpa_improvement * 0.4) - (burnout_penalty * 1.5),
        0.5, 15.0
    )
    motivation_factor = carry.get('motivation', 5) / 10.0
    current['StudyHours'] += (motivation_factor * 0.5)
    # 3. Difficulty Spike: Sems 5, 6, 7 require more effort to maintain the same GPA
    difficulty_spike = 1.05 if sem_id in [5, 6, 7] else 1.0
    current['Effort_Score'] = np.clip(current.get('Effort_Score', 0.7) / difficulty_spike, 0.2, 1.0)
    current['Effort_Score'] -= burnout_penalty

    # 4. Monte Carlo Variance: Inject cumulative noise into behaviors
    if mode == 'montecarlo':
        current['StudyHours'] += np.random.normal(0, 0.6)
        current['AttendanceRate'] += np.random.normal(0, 0.04)
        current['Effort_Score'] += np.random.normal(0, 0.05)
        
        # Ensure values stay in logical bounds after noise
        current['StudyHours'] = np.clip(current['StudyHours'], 0.0, 15.0)
        current['AttendanceRate'] = np.clip(current['AttendanceRate'], 0.1, 1.0)
        current['Effort_Score'] = np.clip(current['Effort_Score'], 0.1, 1.0)

    return current

def simulate_behavioral_drift(current, carry, sem_id):
    updated = current.copy()
    
    # Apply "Senioritis" drift: Attendance drops 1% per sem after Sem 4
    if sem_id > 4:
        updated['AttendanceRate'] *= 0.99 
    
    # Apply Motivation feedback: High motivation boosts Study Hours
    motivation_factor = carry.get('motivation', 5) / 10.0
    updated['StudyHours'] += (motivation_factor * 0.5)
    
    # Apply Burnout: High burnout reduces Effort_Score
    burnout_penalty = carry.get('burnout', 0) / 20.0
    updated['Effort_Score'] -= burnout_penalty
    
    return updated

def simulate_one_trajectory(static: dict, carry: dict, current_sem: int, current_features: dict, snapshot: str, noise_std: float = 0.0) -> list:
    trajectory = []
    c = carry.copy()

    # Base observation
    feats = build_feature_dict(static, c, current_features, current_sem)
    r_gpa  = run_prediction(snapshot, 'gpa', 'lgbm', feats)
    r_drop = run_prediction(snapshot, 'dropout', 'lgbm_recall', feats)
    r_arch = run_prediction(snapshot, 'archetype', 'lgbm', feats)

    # Before the loop (Observed Semester)
    trajectory.append({
        'semester': current_sem, 'stage': snapshot, 'data_type': 'observed',
        'GPA_Predicted': r_gpa['GPA_Predicted'],
        'Dropout_Probability': r_drop['Dropout_Probability'],
        'Dropout_Predicted': r_drop['Dropout_Predicted'], 
        'Archetype': r_arch['Archetype_Predicted'],
        'Risk_Tier': r_gpa['Risk_Tier'], 
        'Archetype_Probabilities': r_arch['Archetype_Probabilities'],
        'StudyHours': current_features.get('StudyHours', 0),
        'AttendanceRate': current_features.get('AttendanceRate', 0),
        'Effort_Score': current_features.get('Effort_Score', 0)
    })

    c = advance_carry(c, current_features, c['prev_gpa'], r_gpa['GPA_Predicted'], current_sem)
    c['archetype'] = r_arch['Archetype_Predicted']

    # Initialize the "evolving" features with the current semester's features
    evolved_current = current_features.copy()

    for fut_sem in range(current_sem + 1, MAX_SEMESTERS + 1):
        if c['archetype'] == 'dropped_out':
            trajectory.append({'semester': fut_sem, 'stage': 'N/A', 'data_type': 'simulated', 'status': 'dropped_out', 'GPA_Predicted': 0.0, 'Dropout_Probability': 1.0, 'Dropout_Predicted': True, 'Archetype': 'dropped_out', 'Risk_Tier': 'Critical Risk', 'Archetype_Probabilities': {}})
            break

        # 1. Evolve behavioral stats for the NEXT semester
        mode = 'montecarlo' if noise_std > 0 else 'trajectory'
        evolved_current = simulate_next_semester(static, c, evolved_current, fut_sem, mode)

        # 2. Build the final dictionary to pass to the model
        feats = impute_future_semester(static, evolved_current, c, fut_sem)

        # 3. Predict using the newly evolved features
        r_gpa  = run_prediction('before_ia', 'gpa', 'lgbm', feats)
        r_drop = run_prediction('before_ia', 'dropout', 'lgbm_recall', feats)
        r_arch = run_prediction('before_ia', 'archetype', 'lgbm', feats)

        # Inside the loop (Simulated Semesters)
        trajectory.append({
            'semester': fut_sem, 'stage': 'before_ia', 'data_type': 'simulated',
            'GPA_Predicted': r_gpa['GPA_Predicted'],
            'Dropout_Probability': r_drop['Dropout_Probability'],
            'Dropout_Predicted': r_drop['Dropout_Predicted'], 
            'Archetype': r_arch['Archetype_Predicted'],
            'Risk_Tier': r_gpa['Risk_Tier'], 
            'Archetype_Probabilities': r_arch['Archetype_Probabilities'],
            'StudyHours': evolved_current.get('StudyHours', 0),
            'AttendanceRate': evolved_current.get('AttendanceRate', 0),
            'Effort_Score': evolved_current.get('Effort_Score', 0)
        })

        c = advance_carry(c, feats, c['prev_gpa'], r_gpa['GPA_Predicted'], fut_sem)
        c['archetype'] = r_arch['Archetype_Predicted']

    return trajectory

def trajectory_summary(traj: list) -> dict:
    gpas, drops, archs = [r['GPA_Predicted'] for r in traj], [r['Dropout_Probability'] for r in traj], [r['Archetype'] for r in traj]
    return {
        'semesters_forecast': len(traj), 'GPA_trend': [round(g, 2) for g in gpas], 'avg_GPA': round(float(np.mean(gpas)), 2),
        'min_GPA': round(float(np.min(gpas)), 2), 'max_GPA': round(float(np.max(gpas)), 2),
        'max_dropout_risk': round(float(max(drops)), 4), 'archetype_trajectory': archs,
        'final_archetype': archs[-1], 'dropout_flagged_any': any(r.get('Dropout_Predicted', False) for r in traj),
        'overall_risk': _risk_tier(max(drops)),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 7. What-If scenario definitions
# ══════════════════════════════════════════════════════════════════════════════
_SET = {'PartTimeJob', 'Intervention_Active', 'Intervention_Applied', 'Shock_Event', 'Shock_Magnitude'}

SCENARIO_DELTAS = {
    'peer_tutoring': {'StudyHours': +1.0, 'Effort_Score': +0.10, 'Backlog_Count': -1, 'Intervention_Active': 1, 'Intervention_Applied': 'peer_tutoring'},
    'counseling': {'stress_survey_score': -1.5, 'exam_anxiety_score': -1.2, 'motivation_survey_score': +1.0, 'Dynamic_Burnout': -0.5, 'sleep_hours': +0.5, 'Intervention_Active': 1, 'Intervention_Applied': 'counseling'},
    'mentorship': {'motivation_survey_score': +1.2, 'goal_clarity_survey_score': +1.0, 'self_efficacy_score': +0.8, 'StudyHours': +0.5, 'Intervention_Active': 1, 'Intervention_Applied': 'mentorship'},
    'academic_warning': {'AttendanceRate': +0.05, 'StudyHours': +0.8, 'Effort_Score': +0.10, 'Intervention_Active': 1, 'Intervention_Applied': 'academic_warning'},
    'reduce_parttime': {'PartTimeJob': 0, 'StudyHours': +1.5, 'Dynamic_Burnout': -0.3, 'Effort_Score': +0.08},
    'scholarship_award': {'motivation_survey_score': +1.8, 'self_efficacy_score': +1.0, 'stress_survey_score': -1.0, 'Shock_Event': 'scholarship_award', 'Shock_Magnitude': 1.8},
    'mentorship_start': {'motivation_survey_score': +1.5, 'goal_clarity_survey_score': +1.2, 'Shock_Event': 'mentorship_start', 'Shock_Magnitude': 1.5},
    'family_issue': {'AttendanceRate': -0.10, 'StudyHours': -0.8, 'motivation_survey_score': -1.5, 'Dynamic_Burnout': +0.8, 'Shock_Event': 'family_issue', 'Shock_Magnitude': 1.5},
    'health_problem': {'AttendanceRate': -0.12, 'StudyHours': -1.0, 'motivation_survey_score': -1.2, 'Dynamic_Burnout': +0.6, 'Shock_Event': 'health_problem', 'Shock_Magnitude': 1.2},
    'financial_stress': {'StudyHours': -0.5, 'stress_survey_score': +1.5, 'motivation_survey_score': -1.0, 'Dynamic_Burnout': +0.5, 'Shock_Event': 'financial_stress', 'Shock_Magnitude': 1.0},
    'breakup': {'StudyHours': -0.6, 'motivation_survey_score': -1.8, 'sleep_hours': -0.8, 'Dynamic_Burnout': +0.4, 'Shock_Event': 'breakup', 'Shock_Magnitude': 1.8},
    'bereavement': {'AttendanceRate': -0.15, 'StudyHours': -1.2, 'motivation_survey_score': -2.0, 'Dynamic_Burnout': +1.0, 'stress_survey_score': +2.0, 'Shock_Event': 'bereavement', 'Shock_Magnitude': 2.0},
    'increase_parttime': {'PartTimeJob': 1, 'StudyHours': -0.8, 'Dynamic_Burnout': +0.5, 'AttendanceRate': -0.05, 'Shock_Event': 'part_time_job_increase', 'Shock_Magnitude': 0.5},
}

SCENARIO_DIRECTION = {
    'peer_tutoring': 'positive', 'counseling': 'positive', 'mentorship': 'positive', 'academic_warning': 'positive', 'reduce_parttime': 'positive', 'scholarship_award': 'positive', 'mentorship_start': 'positive',
    'family_issue': 'negative', 'health_problem': 'negative', 'financial_stress': 'negative', 'breakup': 'negative', 'bereavement': 'negative', 'increase_parttime': 'negative',
}

def apply_scenario(features: dict, scenario: str) -> dict:
    f = features.copy()
    for field, value in SCENARIO_DELTAS[scenario].items():
        if field in _SET: f[field] = value
        else: f[field] = float(f.get(field, 0)) + value
    return f

# ══════════════════════════════════════════════════════════════════════════════
# 8. Risk helpers
# ══════════════════════════════════════════════════════════════════════════════
def _risk_tier(score: float) -> str:
    if score < 0.25: return 'Low Risk'
    if score < 0.50: return 'Moderate Risk'
    if score < 0.75: return 'High Risk'
    return 'Critical Risk'

def _intervention_level(tier: str) -> str:
    return {'Low Risk':'NONE','Moderate Risk':'MONITOR','High Risk':'SUPPORT','Critical Risk':'URGENT'}.get(tier,'UNKNOWN')

def _risk_drivers(row: dict) -> list:
    drivers = []
    if float(row.get('AttendanceRate', 1)) < 0.75: drivers.append('Low Attendance')
    if int(row.get('Backlog_Count', 0)) > 2: drivers.append('High Backlog')
    if float(row.get('sleep_hours', 8)) < 5: drivers.append('Sleep Deprivation')
    if float(row.get('stress_survey_score', 0)) > 7: drivers.append('High Stress')
    if float(row.get('motivation_survey_score', 10)) < 4: drivers.append('Low Motivation')
    if float(row.get('StudyHours', 5)) < 2: drivers.append('Low Study Hours')
    if float(row.get('Effort_Score', 5)) < 0.3: drivers.append('Low Effort')
    if float(row.get('exam_anxiety_score', 0)) > 7: drivers.append('High Exam Anxiety')
    if float(row.get('Prev_GPA', 10)) < 5: drivers.append('Poor Prior GPA')
    if float(row.get('Dynamic_Burnout', 0)) > 5: drivers.append('High Burnout')
    return drivers

ADVISORY_MAP = {'Low Attendance': 'Improve attendance — target ≥80%', 'High Backlog': 'Clear pending backlogs before end of term', 'Sleep Deprivation': 'Establish consistent sleep schedule (7–8 hrs)', 'High Stress': 'Engage counselling or stress management support', 'Low Motivation': 'Goal-setting session recommended with mentor', 'Low Study Hours': 'Increase structured study time to ≥3 hrs/day', 'Low Effort': 'Review assignment submission and engagement patterns', 'High Exam Anxiety': 'Refer to academic skills workshop or therapy', 'Poor Prior GPA': 'Academic remediation or tutoring recommended', 'High Burnout': 'Reduce cognitive load — drop optional commitments'}
FOCUS_MAP = {'Low Attendance': 'Attendance', 'High Backlog': 'Academic Workload', 'Sleep Deprivation': 'Wellbeing', 'High Stress': 'Mental Health', 'Low Motivation': 'Motivation', 'Low Study Hours': 'Study Habits', 'Low Effort': 'Engagement', 'High Exam Anxiety': 'Mental Health', 'Poor Prior GPA': 'Academic Support','High Burnout': 'Wellbeing'}

def _archetype_profile(arch: str) -> dict:
    return {
        'elite': {'description':'High performer, consistent, self-driven', 'risk_level':'low', 'watch_for':'Burnout from overcommitment'},
        'fading_star': {'description':'Previously strong, currently declining', 'risk_level':'medium', 'watch_for':'GPA drop acceleration'},
        'late_bloomer': {'description':'Improving trajectory, gaining momentum', 'risk_level':'low-medium','watch_for':'Sustaining improvement'},
        'pragmatist': {'description':'Stable, goal-oriented, meets requirements', 'risk_level':'low', 'watch_for':'Disengagement if goals shift'},
        'vulnerable': {'description':'At-risk, multiple compounding stressors', 'risk_level':'high', 'watch_for':'Dropout, mental health crisis'},
        'disengaged': {'description':'Withdrawn from academic process', 'risk_level':'high', 'watch_for':'Escalation to dropout'},
        'dropped_out': {'description':'Has discontinued studies', 'risk_level':'critical','watch_for':'Re-engagement pathway'},
    }.get(arch, {'description':'Unknown','risk_level':'unknown','watch_for':''})

# ══════════════════════════════════════════════════════════════════════════════
# 9. Request parsing helpers
# ══════════════════════════════════════════════════════════════════════════════
def parse_student_request(data: dict):
    snapshot = data.get('snapshot')
    if snapshot not in VALID_SNAPSHOTS: raise ValueError(f"Invalid snapshot '{snapshot}'")
    static, history, current = data.get('static', {}), data.get('history', []), data.get('current', {})
    if not static: raise ValueError("'static' block is required")
    if not current: raise ValueError("'current' block is required")
    sem_id = len(history) + 1
    carry = bootstrap_carry(static, history)
    return static, carry, current, snapshot, sem_id

# ══════════════════════════════════════════════════════════════════════════════
# 10. Routes
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/predict/full_profile', methods=['POST'])
def full_profile():
    try:
        # Step 1: Get and validate input
        data = request.get_json(force=True)
        if not data:
            raise ValueError("Request body is empty")
        
        # Step 2: Parse student data
        try:
            static, carry, current, snapshot, sem_id = parse_student_request(data)
        except Exception as e:
            logging.error(f"Error parsing student request: {str(e)}")
            raise ValueError(f"Invalid student data: {str(e)}")
        
        # Step 3: Build features
        try:
            features = build_feature_dict(static, carry, current, sem_id)
        except Exception as e:
            logging.error(f"Error building features: {str(e)}")
            raise ValueError(f"Cannot build features: {str(e)}")
        
        # Step 4: Run predictions
        try:
            result = predict_all_tasks(snapshot, features)
        except Exception as e:
            logging.error(f"Error in predict_all_tasks: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
        
        # Step 5: Add carry state (with safe defaults)
        try:
            result['carry_state_computed'] = {
                'semester'           : sem_id,
                'Dynamic_Motivation' : round(carry.get('motivation', 5.0), 2),
                'Dynamic_Confidence' : round(carry.get('confidence', 5.0), 2),
                'Dynamic_Burnout'    : round(carry.get('burnout', 0.0), 2),
                'Archetype_Prev'     : carry.get('archetype', 'unknown'),
                'Prev_GPA'           : round(carry.get('prev_gpa', 5.0), 2),
            }
        except Exception as e:
            logging.error(f"Error building carry state: {str(e)}")
            # Don't fail on carry state, it's optional
            result['carry_state_computed'] = {
                'semester': sem_id,
                'Dynamic_Motivation': 5.0,
                'Dynamic_Confidence': 5.0,
                'Dynamic_Burnout': 0.0,
                'Archetype_Prev': 'unknown',
                'Prev_GPA': 5.0,
            }
        
        return jsonify(result)
        
    except ValueError as e:
        logging.error(f"ValueError in full_profile: {str(e)}")
        return jsonify({'error': str(e), 'type': 'validation'}), 400
    except KeyError as e:
        logging.error(f"KeyError in full_profile: {str(e)}")
        return jsonify({'error': f'Missing field: {str(e)}', 'type': 'missing_field'}), 400
    except Exception as e:
        logging.error(f"Unexpected error in full_profile: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'type': 'unknown', 'details': 'Check server logs'}), 500

@app.route('/predict/whatif', methods=['POST'])
def whatif():
    try:
        data = request.get_json(force=True)
        static, carry, current, snapshot, sem_id = parse_student_request(data)
        task = data.get('task', 'gpa')
        model_name = data.get('model_name') or ('lgbm_recall' if task == 'dropout' else 'lgbm')
        
        # Determine scenarios. Only include valid ones from SCENARIO_DELTAS
        scenarios = data.get('scenarios', list(SCENARIO_DELTAS.keys()))
        valid_scenarios = [s for s in scenarios if s in SCENARIO_DELTAS]

        base_features = build_feature_dict(static, carry, current, sem_id)
        baseline = run_prediction(snapshot, task, model_name, base_features)
        scenario_results = {}

        for sc in valid_scenarios:
            modified = apply_scenario(base_features, sc)
            pred = run_prediction(snapshot, task, model_name, modified)
            delta = {}
            if task == 'gpa':
                delta['GPA_Delta'] = round(pred['GPA_Predicted'] - baseline['GPA_Predicted'], 3)
            scenario_results[sc] = {
                **pred,
                'direction': SCENARIO_DIRECTION.get(sc, 'unknown'),
                'delta_vs_baseline': delta,
            }

        ranked = sorted(scenario_results.items(), key=lambda x: (x[1].get('delta_vs_baseline', {}).get('GPA_Delta', 0)), reverse=True)
        return jsonify({
            'baseline': baseline,
            'scenarios': scenario_results,
            'ranked_best_to_worst': [k for k, _ in ranked]
        })
    except (ValueError, KeyError) as e: return jsonify({'error': str(e)}), 400
    except Exception as e: return jsonify({'error': 'What-if failed', 'details': str(e)}), 500

@app.route('/predict/trajectory', methods=['POST'])
def trajectory():
    try:
        data = request.get_json(force=True)
        static, carry, current, snapshot, sem_id = parse_student_request(data)
        if sem_id >= MAX_SEMESTERS: return jsonify({'error': f'Already at final semester ({MAX_SEMESTERS})'}), 400
        traj = simulate_one_trajectory(static, carry, sem_id, current, snapshot)
        return jsonify({'current_semester': sem_id, 'forecast_through': MAX_SEMESTERS, 'summary': trajectory_summary(traj), 'trajectory': traj})
    except (ValueError, KeyError) as e: return jsonify({'error': str(e)}), 400

@app.route('/predict/montecarlo', methods=['POST'])
def montecarlo():
    try:
        data = request.get_json(force=True)
        static, carry, current, snapshot, sem_id = parse_student_request(data)
        n_runs = min(int(data.get('n_runs', 100)), 500)
        noise_std = float(data.get('noise_std', 0.2))

        if sem_id >= MAX_SEMESTERS: return jsonify({'error': f'Already at final semester ({MAX_SEMESTERS})'}), 400

        all_gpas, all_drops, all_archs = [], [], []
        for run in range(n_runs):
            noisy_carry = bootstrap_carry(static, data.get('history', []), rng_noise_std=noise_std)
            traj = simulate_one_trajectory(static, noisy_carry, sem_id, current, snapshot, noise_std=noise_std)
            all_gpas.append([r['GPA_Predicted'] for r in traj])
            all_drops.append([r['Dropout_Probability'] for r in traj])
            all_archs.append([r['Archetype'] for r in traj])

        max_len = MAX_SEMESTERS - sem_id + 1
        for lst in (all_gpas, all_drops, all_archs):
            for row in lst:
                while len(row) < max_len: row.append(row[-1])

        gpas_arr, drops_arr = np.array(all_gpas), np.array(all_drops)
        semesters, per_sem = list(range(sem_id, MAX_SEMESTERS + 1)), []

        for j, s in enumerate(semesters):
            gpa_col, drop_col = gpas_arr[:, j], drops_arr[:, j]
            arch_col = [all_archs[r][j] for r in range(n_runs)]
            per_sem.append({
                'semester': s,
                'GPA': {'mean': round(float(np.mean(gpa_col)), 2), 'p10': round(float(np.percentile(gpa_col, 10)), 2), 'p90': round(float(np.percentile(gpa_col, 90)), 2)},
                'Dropout_Probability': {'mean': round(float(np.mean(drop_col)), 4), 'p10': round(float(np.percentile(drop_col, 10)), 4), 'p90': round(float(np.percentile(drop_col, 90)), 4)}
            })

        return jsonify({
            'current_semester': sem_id, 'forecast_through': MAX_SEMESTERS, 'n_runs': n_runs,
            'summary': {
                'GPA_final_mean': round(float(np.mean(gpas_arr[:, -1])), 2),
                'GPA_final_std': round(float(np.std(gpas_arr[:, -1])), 2),
                'GPA_final_p10_p90': [round(float(np.percentile(gpas_arr[:, -1], 10)), 2), round(float(np.percentile(gpas_arr[:, -1], 90)), 2)],
                'dropout_risk_mean': round(float(np.mean(drops_arr[:, -1])), 4),
                'prob_dropout_any_sem': round(float(np.mean(np.any(drops_arr >= 0.30, axis=1))), 3),
            },
            'per_semester': per_sem,
        })
    except (ValueError, KeyError) as e: return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)