from app.models import BodySystem

CLINICAL_PATIENTS = [
    {
        "patient_id": "PT-001",
        "age": 65,
        "gender": "M",
        "chief_complaint": "Crushing chest pain radiating to left arm",
        "history": "Hypertension, Type 2 DM, chronic smoker",
        "vitals": {"HR": "110", "BP": "165/95", "RR": "24", "SpO2": "94%", "Temp": "98.6F"},
        "true_body_system": BodySystem.CARDIAC,
        "true_esi_level": 1,
        "true_triage_note": "Patient presents with classic symptoms of acute myocardial infarction (STEMI/NSTEMI). High risk due to cardiac history. Requires immediate EKG, IV access, and continuous cardiac monitoring in a resuscitation bay.",
    },
    {
        "patient_id": "PT-002",
        "age": 22,
        "gender": "F",
        "chief_complaint": "Twisted right ankle during soccer",
        "history": "None",
        "vitals": {"HR": "85", "BP": "110/70", "RR": "16", "SpO2": "99%", "Temp": "98.8F"},
        "true_body_system": BodySystem.MUSCULOSKELETAL,
        "true_esi_level": 4,
        "true_triage_note": "Patient with isolated right ankle injury. Vitals stable. Pain is 6/10. Requires one resource (X-ray). Can safely wait in the lobby.",
    },
    {
        "patient_id": "PT-003",
        "age": 45,
        "gender": "F",
        "chief_complaint": "Severe right lower quadrant abdominal pain for 12 hours",
        "history": "Appendectomy at age 12",
        "vitals": {"HR": "105", "BP": "130/80", "RR": "18", "SpO2": "98%", "Temp": "101.4F"},
        "true_body_system": BodySystem.GI,
        "true_esi_level": 2,
        "true_triage_note": "High-risk scenario: acute abdomen with systemic fever and tachycardia. Risk of bowel obstruction, perforation, or pelvic inflammatory disease. Place in acute bed, needs IV, labs, and CT scan.",
    },
    {
        "patient_id": "PT-004",
        "age": 8,
        "gender": "M",
        "chief_complaint": "Wheezing and shortness of breath",
        "history": "Asthma",
        "vitals": {"HR": "130", "BP": "100/60", "RR": "32", "SpO2": "90%", "Temp": "99.0F"},
        "true_body_system": BodySystem.RESPIRATORY,
        "true_esi_level": 2,
        "true_triage_note": "Pediatric asthma exacerbation with hypoxia (SpO2 90%) and tachypnea. Potential for rapid decompensation. Requires immediate nebulizer treatment, steroids, and continuous oxygen.",
    },
    {
        "patient_id": "PT-005",
        "age": 30,
        "gender": "M",
        "chief_complaint": "Prescription refill for Lisinopril",
        "history": "Hypertension",
        "vitals": {"HR": "75", "BP": "125/80", "RR": "14", "SpO2": "100%", "Temp": "98.2F"},
        "true_body_system": BodySystem.OTHER,
        "true_esi_level": 5,
        "true_triage_note": "Non-urgent request for medication refill. Vital signs are normal, asymptomatic. Zero resources required from the ED. Send to fast track.",
    },
    {
        "patient_id": "PT-006",
        "age": 75,
        "gender": "F",
        "chief_complaint": "Sudden onset left-sided weakness and slurred speech",
        "history": "Atrial fibrillation",
        "vitals": {"HR": "88", "BP": "180/100", "RR": "18", "SpO2": "95%", "Temp": "98.4F"},
        "true_body_system": BodySystem.NEUROLOGIC,
        "true_esi_level": 1,
        "true_triage_note": "Acute stroke alert. Sudden onset neurological deficits (FAST positive) and severe hypertension. Requires immediate activation of stroke protocol, STAT head CT, and neurology consult.",
    },
    {
        "patient_id": "PT-007",
        "age": 19,
        "gender": "M",
        "chief_complaint": "Sore throat and mild fever for 3 days",
        "history": "None",
        "vitals": {"HR": "90", "BP": "115/75", "RR": "16", "SpO2": "99%", "Temp": "100.2F"},
        "true_body_system": BodySystem.OTHER,
        "true_esi_level": 4,
        "true_triage_note": "Suspected pharyngitis or viral URI. Airway is intact, vitals stable. Requires point-of-care rapid strep swab (one resource).",
    },
    {
        "patient_id": "PT-008",
        "age": 55,
        "gender": "M",
        "chief_complaint": "Vomiting dark blood (coffee grounds)",
        "history": "Cirrhosis, heavy alcohol use",
        "vitals": {"HR": "125", "BP": "85/50", "RR": "22", "SpO2": "96%", "Temp": "97.8F"},
        "true_body_system": BodySystem.GI,
        "true_esi_level": 1,
        "true_triage_note": "Upper GI bleed with hemorrhagic shock (hypotension, marked tachycardia). Imminent threat to life. Move to resuscitation room immediately for massive transfusion protocol and GI consult.",
    },
    {
        "patient_id": "PT-009",
        "age": 60,
        "gender": "F",
        "chief_complaint": "Chronic back pain, worse today",
        "history": "Osteoarthritis",
        "vitals": {"HR": "80", "BP": "140/90", "RR": "16", "SpO2": "98%", "Temp": "98.6F"},
        "true_body_system": BodySystem.MUSCULOSKELETAL,
        "true_esi_level": 3,
        "true_triage_note": "Exacerbation of chronic back pain. Normal vitals. Requires multiple resources (pain meds, possible plain films, IM injection), but no high-risk features.",
    },
    {
        "patient_id": "PT-010",
        "age": 40,
        "gender": "F",
        "chief_complaint": "Shortness of breath for 1 week, leg swelling",
        "history": "Recent long-haul flight",
        "vitals": {"HR": "110", "BP": "120/80", "RR": "22", "SpO2": "93%", "Temp": "98.9F"},
        "true_body_system": BodySystem.RESPIRATORY,
        "true_esi_level": 2,
        "true_triage_note": "High risk for Pulmonary Embolism given history of travel, tachycardia, and hypoxemia. Needs immediate workup (CT Angio, D-Dimer, EKG). Room in acute area.",
    }
]

try:
    from datasets import load_dataset
    import random
    _MED_DS = list(load_dataset("medical_questions_pairs", split="train", streaming=True).take(50))
    for row in _MED_DS:
        CLINICAL_PATIENTS.append({
            "patient_id": f"PT-HF-{random.randint(1000,9999)}",
            "age": random.randint(18, 80),
            "gender": random.choice(["M", "F"]),
            "chief_complaint": row.get("question_1", "Medical issue"),
            "vitals": {"HR": "90", "BP": "120/80", "RR": "16", "SpO2": "98%", "Temp": "98.6F"},
            "history": row.get("question_2", "None"),
            "true_body_system": BodySystem.OTHER,
            "true_esi_level": 4,
            "true_triage_note": "Standard triage for generic complaint."
        })
except Exception:
    pass
