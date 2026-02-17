import os
from safetensors.torch import load_file, save_file

# Chemin vers votre checkpoint malade
ckpt_folder = "./checkpoints/banking77_finetuned_instructor_qwen2.5_7b/checkpoint-738"
input_file = os.path.join(ckpt_folder, "model.safetensors")
output_file = os.path.join(ckpt_folder, "model_fixed.safetensors")

print(f"🔧 Chargement de : {input_file}")
state_dict = load_file(input_file)
new_state_dict = {}

fixed_count = 0
for key, value in state_dict.items():
    # On enlève le préfixe "0.auto_model."
    if key.startswith("0.auto_model."):
        new_key = key.replace("0.auto_model.", "")
        new_state_dict[new_key] = value
        fixed_count += 1
    # Cas rare : préfixe "0." simple
    elif key.startswith("0."):
        new_key = key.replace("0.", "")
        new_state_dict[new_key] = value
        fixed_count += 1
    else:
        # On garde les autres clés telles quelles
        new_state_dict[key] = value

print(f"✅ {fixed_count} clés réparées.")
print(f"💾 Sauvegarde dans : {output_file}")
save_file(new_state_dict, output_file)
print("Terminé !")
