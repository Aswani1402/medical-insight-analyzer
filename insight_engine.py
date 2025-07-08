# insight_engine.py

def generate_insight(disease, organ, caption):
    if not isinstance(disease, str):
        return "No disease detected."
    if "Cardiomegaly" in disease:
        return f"{disease} suggests heart enlargement. Caption: '{caption}'"
    elif "Pneumonia" in disease:
        return f"{disease} may involve lung inflammation. Caption: '{caption}'"
    elif "Effusion" in disease:
        return f"{disease} may be fluid buildup in lungs. Caption: '{caption}'"
    else:
        return f"Caption: '{caption}' — further analysis required."

def advanced_nlp_insight(caption, disease):
    insight = ""
    if "opacity" in caption or "consolidation" in caption:
        insight += "Possible fluid accumulation or infection. "
    if "blurring" in caption:
        insight += "Early signs of inflammation. "
    if "clear lungs" in caption:
        insight += "Lungs appear normal. "
    if "Cardiomegaly" in disease:
        insight += "Heart may be enlarged or under stress."
    return insight.strip()
