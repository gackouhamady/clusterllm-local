import logging
import numpy as np
from transformers import AutoTokenizer
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# --- Metrics Functions (Unchanged) ---
def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': clustering_accuracy_score(y_true, y_pred)*100,
            'ARI': adjusted_rand_score(y_true, y_pred)*100,
            'NMI': normalized_mutual_info_score(y_true, y_pred)*100}

# --- Dictionary Enriched with Your 10 Specific Datasets ---
# Strategic mapping based on ClusterLLM paper appendices (Table 10)
DEFINITIONS = {
    'hkunlp/instructor-xl': {
        # ... (Kept legacy keys if needed, but added yours below) ...
        'bank77': 'Represent the bank purpose for retrieval: ',
        'clinc_intent': 'Represent the sentence for retrieving the purpose: ',
        'mtop_intent': 'Represent the sentence for retrieving the purpose: ',
        'mtop_domain': 'Represent a sentence: ',
        'massive_intent': 'Represent the sentence for retrieving the purpose: ',
        'massive_domain': 'Represent the scene for retrieval: ',
        'stackex': 'Represent the question for retrieval: ',
        'arxiv': 'Represent the science statement for retrieval: ',
        'reddit': 'represent a reddit community title: ',
        'go_emotions': 'Represent an emotion sentence for retrieval: ',
    },
    'hkunlp/instructor-large': {
        # --- YOUR 10 DATASETS (Strict Naming & Paper Prompts) ---
        
        # 1. Intent Discovery [cite: 637]
        'bank77': 'Represent the bank purpose for retrieval: ',
        'clinc_intent': 'Represent the sentence for retrieving the purpose: ',
        'mtop_intent': 'Represent the sentence for retrieving the purpose: ',
        'massive_intent': 'Represent the sentence for retrieving the purpose: ',
        
        # 2. Domain Discovery [cite: 614]
        'mtop_domain': 'Represent a sentence: ',
        'massive_domain': 'Represent the scene for retrieval: ',
        # Note: CLINC(D) usually shares prompt style with domain tasks
        
        # 3. Topic Mining [cite: 614]
        'stackex': 'Represent the question for retrieval: ',
        'arxiv': 'Represent the science statement for retrieval: ',
        'reddit': 'represent a reddit community title: ',
        
        # 4. Emotion Detection [cite: 614]
        'go_emotions': 'Represent an emotion sentence for retrieval: ',

        # --- Legacy / Other keys (kept for compatibility) ---
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieving duplicate sentences: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
        'RedditClustering': 'represent a reddit community title: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent the question for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
        'banking77': 'Represent the bank purpose for retrieval: ', # Legacy name
        'few_rel_nat': 'Represent the relation between two entities for retrieval: ',
        'ArxivClusteringS2S_coarse': 'Represent the science statement for retrieval: ',
        'ArxivClusteringS2S_fine': 'Represent the science statement for retrieval: ',
        'arxiv_fine': 'Represent the science statement for retrieval: ',
        'go_emotion': 'Represent an emotion sentence for retrieval: ', # Legacy name
        'few_nerd_nat': 'Represent the entity type for retrieval: ',
        "few_event": "Represent the event type for retrieval: ",
        "stackexchange_p2p": "Represent the question and answer for retrieving duplicate question and answers: ",
        "reddit_p2p": "represent a Reddit community passage: ",
        "massive_scenario": "Represent the scene for retrieval: ",
        "clinc": "Represent the sentence for retrieving the purpose: ",
        "clinc_domain": "Represent a sentence: ",
    },
    'hkunlp/instructor-base': {
        # Mapped similarly for consistency if you ever switch to base
        'bank77': 'Represent the bank purpose for retrieval: ',
        'clinc_intent': 'Represent the sentence for retrieving the purpose: ',
        'mtop_intent': 'Represent the sentence for retrieving the purpose: ',
        'mtop_domain': 'Represent a sentence: ',
        'massive_intent': 'Represent the sentence for retrieving the purpose: ',
        'massive_domain': 'Represent the scene for retrieval: ',
        'stackex': 'Represent the question for retrieval: ',
        'arxiv': 'Represent the science statement for retrieval: ',
        'reddit': 'represent a reddit community title: ',
        'go_emotions': 'Represent an emotion sentence for retrieval: ',
        
        # Legacy
        'TwentyNewsgroupsClustering': 'Represent the news comment for retrieval: ',
        'BiorxivClusteringS2S': 'Represent the biomedical statement for retrieval: ',
        'MedrxivClusteringS2S': 'Represent the medicine statement for retrieving duplicate sentences: ',
        'ArxivClusteringP2P': 'Represent the science passage for retrieval: ',
        'ArxivClusteringS2S': 'Represent the science statement for retrieval: ',
        'BiorxivClusteringP2P': 'Represent the Bio-medicine passage for retrieval: ',
        'MedrxivClusteringP2P': 'Represent the medicine paragraph for retrieval: ',
        'RedditClustering': 'represent a reddit community title: ',
        'RedditClusteringP2P': 'represent a Reddit community passage: ',
        'StackExchangeClustering': 'Represent the question for retrieval: ',
        'StackExchangeClusteringP2P': 'Represent the question and answer for retrieving duplicate question and answers: ',
        'banking77': 'Represent the bank purpose for retrieval: ',
        'few_rel_nat': 'Represent the relation between two entities for retrieval: ',
        'ArxivClusteringS2S_coarse': 'Represent the science statement for retrieval: ',
        'ArxivClusteringS2S_fine': 'Represent the science statement for retrieval: ',
        'arxiv_fine': 'Represent the science statement for retrieval: ',
        'synthesized_emotion': 'Represent the example according to the emotion for retrieval: ',
        'synthesized_style': 'Represent the example according to the style for retrieval: ',
        'synthesized_topic': 'Represent the example according to the topic for retrieval: ',
        'go_emotion': 'Represent an emotion sentence for retrieval: ',
        'few_nerd_nat': 'Represent the entity type for retrieval: ',
        "few_event": "Represent the event type for retrieval: ",
        "stackexchange_p2p": "Represent the question and answer for retrieving duplicate question and answers: ",
        "reddit_p2p": "represent a Reddit community passage: ",
        "massive_scenario": "Represent the scene for retrieval: ",
        "clinc": "Represent the sentence for retrieving the purpose: ",
        "clinc_domain": "Represent a sentence: ",
        "goal_language": "Represent a sentence: ",
        "goal_topic": "Represent a sentence: ",
    },
}

class ClusteringEvaluator(object):
    def __init__(self, sentences, labels, clustering_batch_size=500, limit=None, **kwargs):
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.args = kwargs['args']
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        new_sentences = []
        
        if self.args.prompt:
            print(f'with prompt: {self.args.prompt}')
            
            # --- FIXED LOGIC FOR ACCURATE PROMPT RETRIEVAL ---
            # 1. Get the dictionary for the specific model (e.g., hkunlp/instructor-large)
            model_definitions = DEFINITIONS.get(self.args.prompt, {})
            
            # 2. Retrieve instruction using strict dataset names (bank77, clinc_intent, etc.)
            # If the dataset name from DVC (self.args.task_name) matches a key, we get the optimized prompt.
            instruction = model_definitions.get(
                self.args.task_name, 
                "Represent the document for clustering: " # Fallback only if name mismatch
            )
            
            # Explicit Warning if fallback is used to avoid silent failures
            if self.args.task_name not in model_definitions:
                print(f"⚠️ Warning: Dataset '{self.args.task_name}' NOT found in DEFINITIONS. Using generic prompt. Check spelling!")
            else:
                print(f"✅ Dataset '{self.args.task_name}' found. Using prompt: '{instruction}'")

            for s in self.sentences:
                # Concatenate instruction + sentence
                full_input = instruction + s
                # Truncate if necessary (Instructor usually handles up to 512, but kept 256 per your logic)
                if len(self.tokenizer(full_input)['input_ids']) <= 256:
                    new_sentences.append([instruction, s, 0])
                else:
                    # If too long, maybe skip instruction or truncate? kept original logic:
                    new_sentences.append(['', s, 0])
        else:
            new_sentences = self.sentences

        # Encoding
        corpus_embeddings = np.asarray(model.encode(new_sentences))

        # Metric Calculation
        if self.labels is not None:
            label_ids, n_clusters = self._convert_label_to_ids(self.labels)
            all_measures = self._compute_metrics(corpus_embeddings, label_ids, n_clusters)
        else:
            all_measures = {}

        return all_measures, corpus_embeddings
    
    def eval_only(self, corpus_embeddings):
        if self.labels is not None:
            label_ids, n_clusters = self._convert_label_to_ids(self.labels)
            all_measures = self._compute_metrics(corpus_embeddings, label_ids, n_clusters)
        else:
            all_measures = {}

        return all_measures
    
    def _compute_metrics(self, corpus_embeddings, label_ids, n_clusters):
        """
        Centralized clustering logic.
        """
        all_measures = {'ACC': [], 'NMI': [], 'ARI': []}
        seeds = [100, 13, 21, 36, 42]
        
        for seed in seeds:
            if self.args.scale == "small":
                logger.info(f"Fitting K-Means model (seed: {seed})...")
                # Added n_init='auto' for sklearn compatibility
                preds = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto').fit_predict(corpus_embeddings)
            elif self.args.scale == "large":
                logger.info(f"Fitting MiniBatch K-Means model (seed: {seed})...")
                preds = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init='auto').fit_predict(corpus_embeddings)
            
            preds = np.asarray(preds)
            measures = clustering_score(label_ids, preds)
            for k in measures:
                all_measures[k].append(measures[k])

        final_results = {}
        for k in ['ACC', 'NMI', 'ARI']:
            final_results[f'{k}_mean'] = np.mean(all_measures[k])
            final_results[f'{k}_std'] = np.std(all_measures[k])
        
        return final_results

    def _convert_label_to_ids(self, labels):
        if labels and isinstance(labels[0], list):
            labels = [l[0] for l in labels]
        unique_labels = list(set(labels))
        n_clusters = len(unique_labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        label_ids = [label_map[l] for l in labels]
        return np.asarray(label_ids), n_clusters