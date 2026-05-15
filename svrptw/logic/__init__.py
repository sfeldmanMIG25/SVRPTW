"""Logic axis: distilled dispatcher-acceptance judge (SPEC-6-LOGIC-01).

Teacher (Gemini 3.1 Flash Lite) → student (small MLP, <50ms) → ensemble (5 heads, σ).
Calibrated scalar in [0,1] = P(dispatcher ships it). See specs/SPEC-6-LOGIC-01-distilled-judge.md.
"""
from svrptw.logic.teacher import GeminiTeacher, OpenRouterCommitteeTeacher, TeacherLabel

__all__ = ["GeminiTeacher", "OpenRouterCommitteeTeacher", "TeacherLabel"]
