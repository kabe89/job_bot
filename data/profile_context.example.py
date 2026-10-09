"""Example profile context helper for JobBot.

Allows programmatically defining candidate profile context and constraints
that are injected into LLM prompting when tailoring resumes and cover letters.
"""
from typing import Dict, Any

PROFILE_CONTEXT: Dict[str, Any] = {
    "candidate_name": "Jane Doe",
    "primary_title": "Senior Computational Scientist / Software Engineer",
    "core_strengths": [
        "Distributed system design and cloud computing",
        "Machine learning pipelines (PyTorch, scikit-learn)",
        "Python, C++, SQL, Linux/Bash automation",
        "High-throughput data modeling and scientific computing"
    ],
    "target_roles": [
        "Computational Scientist",
        "Machine Learning Engineer",
        "Senior Software Engineer",
        "Scientific Data Engineer"
    ],
    "preferred_locations": ["Boston, MA", "New York, NY", "Remote"],
    "minimum_salary": 140000
}

def get_profile_context() -> str:
    """Return a formatted string summary of profile constraints."""
    lines = [f"# Candidate Context: {PROFILE_CONTEXT['candidate_name']}"]
    lines.append(f"Title: {PROFILE_CONTEXT['primary_title']}")
    lines.append("Core Strengths: " + ", ".join(PROFILE_CONTEXT["core_strengths"]))
    lines.append("Target Roles: " + ", ".join(PROFILE_CONTEXT["target_roles"]))
    lines.append("Preferred Locations: " + ", ".join(PROFILE_CONTEXT["preferred_locations"]))
    return "\n".join(lines)
