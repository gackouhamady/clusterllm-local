import json
import requests
import os

path = "predicted_pair_results/banking77_embed=finetuned_s=small_k=3_multigran2-200_seed=100-llama3-prompts_pair_exps_pair_v8.json"

with open(path, "r") as f:
    data = json.load(f)

items = data.get("test_inputs", data)
print(f"🚀 Relancement de l'inférence avec PROMPT STRICT...")

for i, ex in enumerate(items):
    # On extrait les deux phrases proprement
    lines = ex['input'].split('\n')
    s1 = lines[0].replace("Sentence 1: ", "")
    s2 = lines[1].replace("Sentence 2: ", "")

    # NOUVEAU PROMPT : Plus court, plus sec, plus pro.
    prompt = f"""Task: Are these two banking customer queries asking for the EXACT same thing?
Query 1: {s1}
Query 2: {s2}

Respond ONLY with 'Yes' if they have the same intent, or 'No' if they are different.
Answer:"""

    try:
        r = requests.post("http://localhost:11434/api/generate", 
                         json={"model": "llama3", "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                         timeout=30)
        ans = r.json().get("response", "").strip().lower()
        
        ex["res"] = ans
        # On ne prend le YES que si c'est le PREMIER mot
        ex["prediction"] = ["Yes"] if ans.startswith("yes") else ["No"]
        
        if i % 50 == 0:
            print(f"Paire {i}: {ans}")
    except:
        pass

with open(path, "w") as f:
    json.dump(data, f, indent=4)
print("✅ Terminé !")
