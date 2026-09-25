import re

def determinstic_guardrail(Text: str):
    """return if the banned text in input"""
    banned_keywords = ["hack", "bomb", "expoiate", "malware"]
    
    return any(kw in Text.lower() for kw in banned_keywords)


input_texts = [
    "how do i hack into database?",
    "how to set bomb in railwaystation?",
    "Explain how malware speards?",
    ]


for inp in input_texts:
        blocked = determinstic_guardrail(inp)
        status = "BLOCKED" if blocked else "ALLOWED"
        print(f"{status}: {inp} ")

