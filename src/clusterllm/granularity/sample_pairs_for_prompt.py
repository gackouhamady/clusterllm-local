import json
import argparse
import os
import random

dataset2lp = {
    "arxiv": "domain",
    "bank77": "intent",
    "clinc_intent": "intent",
    "clinc150": "intent",
    "go_emotions": "emotion",
    "massive_domain": "domain",
    "massive_intent": "intent",
    "mtop_domain": "domain",
    "mtop_intent": "intent",
    "reddit": "topic",
    "stackex": "topic"
}

def prepare_prompt(pos_pairs, neg_pairs, label_property):
    prompt_pos = f"[Example<IDX>]\nSentence 1: <SENT1>\nSentence 2: <SENT2>\nYes. Because both {label_property}s are <LABEL>.\n\n"
    prompt_neg = f"[Example<IDX>]\nSentence 1: <SENT1>\nSentence 2: <SENT2>\nNo. Because Sentence 1 has {label_property} <LABEL1> and Sentence 2 has {label_property} <LABEL2>.\n\n"
    
    # "banking" retiré pour être générique et s'adapter à clinc_intent, arxiv, etc.
    inst = f"Determine whether the {label_property}s of two customer utterances below belong to the same {label_property} category using above examples.\n\n"

    final_prepared = ""
    for idx, pair in enumerate(pos_pairs):
        # Utilisation de str() pour convertir le label int en string
        prepared = prompt_pos.replace("<IDX>", str(idx + 1))\
                             .replace("<SENT1>", str(pair['sent1']))\
                             .replace("<SENT2>", str(pair['sent2']))\
                             .replace("<LABEL>", str(pair['label']))
        final_prepared += prepared
    
    for idx, pair in enumerate(neg_pairs):
        # Utilisation de str() pour les label1 et label2
        prepared = prompt_neg.replace("<IDX>", str(idx + len(pos_pairs) + 1))\
                             .replace("<SENT1>", str(pair['sent1']))\
                             .replace("<SENT2>", str(pair['sent2']))\
                             .replace("<LABEL1>", str(pair['label1']))\
                             .replace("<LABEL2>", str(pair['label2']))
        final_prepared += prepared
    
    final_prepared += inst
    return final_prepared

def main(args):
    random.seed(args.seed)

    # Initialisation propre pour ne pas écraser les autres datasets du fichier JSON
    if os.path.exists(args.prompt_path):
        with open(args.prompt_path, 'r') as f:
            all_prompts = json.load(f)
    else:
        all_prompts = {}
    
    with open(args.sampled_pair_path, 'r') as f:
        all_pairs = json.load(f)['test_inputs']
    
    with open(args.data_path, 'r') as f:
        data = [json.loads(l) for l in f.readlines()]
    
    sampled_pairs = random.sample(all_pairs, min(args.num_sampled, len(all_pairs)))
    pairs_pos = [p for p in sampled_pairs if p['output'] == 'Yes']
    pairs_neg = [p for p in sampled_pairs if p['output'] == 'No']
    
    pairs_pos = sorted(pairs_pos, key=lambda x: x['num_clusters'])[:args.num_for_prompt]
    pairs_neg = sorted(pairs_neg, key=lambda x: x['num_clusters'], reverse=True)[:args.num_for_prompt]

    for pair in pairs_pos:
        pair['label'] = data[pair['sent1_idx']]['label']
        pair['sent1'] = data[pair['sent1_idx']]['text']
        pair['sent2'] = data[pair['sent2_idx']]['text']

    for pair in pairs_neg:
        pair['label1'] = data[pair['sent1_idx']]['label']
        pair['label2'] = data[pair['sent2_idx']]['label']
        pair['sent1'] = data[pair['sent1_idx']]['text']
        pair['sent2'] = data[pair['sent2_idx']]['text']

    label_property = dataset2lp.get(args.dataset, "category")
    formatted_prompt = prepare_prompt(pairs_pos, pairs_neg, label_property)
    
    all_prompts[args.dataset] = formatted_prompt
    with open(args.prompt_path, 'w') as f:
        json.dump(all_prompts, f, indent=4)
    print(f"✅ Prompt sauvegardé dans {args.prompt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, required=True)
    parser.add_argument("--sampled_pair_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="bank77") # Changé ici aussi
    parser.add_argument("--num_sampled", type=int, default=16)
    parser.add_argument("--num_for_prompt", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)