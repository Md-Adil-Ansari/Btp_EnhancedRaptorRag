import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import logging
import random
import time
import urllib.request
import zipfile

from enhancedRaptor import RetrievalAugmentation, RetrievalAugmentationConfig, ClusterTreeConfig
from enhancedRaptor.QAModels import BaseQAModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class DummyQAModel(BaseQAModel):
    """A dummy QA model because we only care about retrieval accuracy for BEIR."""
    def answer_question(self, context, question, cite_sources=False):
        return "Not evaluated."

def download_scifact(out_dir="/tmp/scifact"):
    """Downloads and unzips SciFact dataset if missing."""
    if os.path.exists(os.path.join(out_dir, "corpus.jsonl")):
        return out_dir
        
    logging.info("Downloading BEIR SciFact dataset...")
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join("/tmp", "scifact.zip")
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("/tmp/")
        
    return out_dir

def load_beir_data(data_dir):
    """Loads ALL test queries that have exactly 1 relevant document."""
    corpus = {}
    queries = {}
    qrels = []

    logging.info("Loading corpus...")
    with open(os.path.join(data_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = doc

    logging.info("Loading queries...")
    with open(os.path.join(data_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    logging.info("Loading qrels...")
    # Read qrels. We only want queries that map to exactly 1 document.
    query_to_docs = {}
    with open(os.path.join(data_dir, "qrels", "test.tsv"), "r", encoding="utf-8") as f:
        next(f) # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                q_id, doc_id = parts[0], parts[1]
                if q_id not in query_to_docs:
                    query_to_docs[q_id] = []
                query_to_docs[q_id].append(doc_id)

    # Filter to queries with exactly 1 relevant doc
    selected_q_ids = sorted([q_id for q_id, docs in query_to_docs.items() if len(docs) == 1])
    
    selected_data = []
    for q_id in selected_q_ids:
        doc_id = query_to_docs[q_id][0]
        selected_data.append({
            "query_id": q_id,
            "query_text": queries[q_id],
            "doc_id": doc_id,
            "doc_title": corpus[doc_id].get("title", ""),
            "doc_text": corpus[doc_id].get("text", "")
        })
        
    return selected_data

def build_combined_corpus(selected_data):
    """Combines all selected documents into one string with markers."""
    combined_text = ""
    seen_docs = set()
    for item in selected_data:
        doc_id = item["doc_id"]
        if doc_id not in seen_docs:
            combined_text += f"\n\n=== DOCUMENT {doc_id} ===\n"
            combined_text += f"{item['doc_title']}\n{item['doc_text']}\n"
            seen_docs.add(doc_id)
    return combined_text

def main():
    data_dir = download_scifact()
    selected_data = load_beir_data(data_dir)
    num_samples = len(selected_data)
    logging.info(f"Total queries to evaluate: {num_samples}")
    
    combined_corpus_text = build_combined_corpus(selected_data)
    logging.info(f"Built combined corpus of {len(combined_corpus_text)} characters.")

    qa_model = DummyQAModel()

    # Create the 4 configs
    from enhancedRaptor.SummarizationModels import GemmaSummarizationModel
    summarization_model = GemmaSummarizationModel(model="gemma-3-27b-it")

    # 1. Normal Dense
    nd_config = RetrievalAugmentationConfig(
        tree_builder_config=ClusterTreeConfig(chunking_strategy="token", use_ice=False, max_tokens=100, summarization_model=summarization_model),
        qa_model=qa_model,
        tr_retriever_mode="dense"
    )
    # 2. Normal Hybrid
    nh_config = RetrievalAugmentationConfig(
        tree_builder_config=ClusterTreeConfig(chunking_strategy="token", use_ice=False, max_tokens=100, summarization_model=summarization_model),
        qa_model=qa_model,
        tr_retriever_mode="hybrid"
    )
    # 3. ICE Dense
    id_config = RetrievalAugmentationConfig(
        tree_builder_config=ClusterTreeConfig(chunking_strategy="semantic", chunking_threshold=0.8, use_ice=True, max_tokens=100, summarization_model=summarization_model),
        qa_model=qa_model,
        tr_retriever_mode="dense"
    )
    # 4. ICE Hybrid
    ih_config = RetrievalAugmentationConfig(
        tree_builder_config=ClusterTreeConfig(chunking_strategy="semantic", chunking_threshold=0.8, use_ice=True, max_tokens=100, summarization_model=summarization_model),
        qa_model=qa_model,
        tr_retriever_mode="hybrid"
    )

    os.makedirs("trees", exist_ok=True)
    
    # We only need to build 2 trees (Normal and ICE). Hybrid vs Dense is a retrieval-time setting.
    # Normal Tree
    if not os.path.exists("trees/beir_normal_tree.pkl"):
        logging.info("Building Normal RAPTOR Tree...")
        normal_ra = RetrievalAugmentation(config=nd_config)
        normal_ra.add_documents(combined_corpus_text)
        normal_ra.save("trees/beir_normal_tree.pkl")
    
    # ICE Tree
    if not os.path.exists("trees/beir_ice_tree.pkl"):
        logging.info("Building ICE RAPTOR Tree...")
        ice_ra = RetrievalAugmentation(config=id_config)
        ice_ra.add_documents(combined_corpus_text)
        ice_ra.save("trees/beir_ice_tree.pkl")

    # Initialize the 4 RA instances with the pre-built trees
    ra_normal_dense = RetrievalAugmentation(config=nd_config, tree="trees/beir_normal_tree.pkl")
    ra_normal_hybrid = RetrievalAugmentation(config=nh_config, tree="trees/beir_normal_tree.pkl")
    ra_ice_dense = RetrievalAugmentation(config=id_config, tree="trees/beir_ice_tree.pkl")
    ra_ice_hybrid = RetrievalAugmentation(config=ih_config, tree="trees/beir_ice_tree.pkl")

    results = []
    scores = {"normal_dense": 0, "normal_hybrid": 0, "ice_dense": 0, "ice_hybrid": 0}

    for i, item in enumerate(selected_data):
        logging.info(f"Evaluating Query {i+1}/{num_samples}: {item['query_text']}")
        q = item['query_text']
        target_marker = f"=== DOCUMENT {item['doc_id']} ==="
        
        q_result = {
            "query_id": item['query_id'],
            "doc_id": item['doc_id'],
            "query_text": q
        }
        
        for name, ra in [
            ("normal_dense", ra_normal_dense),
            ("normal_hybrid", ra_normal_hybrid),
            ("ice_dense", ra_ice_dense),
            ("ice_hybrid", ra_ice_hybrid)
        ]:
            try:
                context, _ = ra.retrieve(q, top_k=5)
                success = 1 if target_marker in context else 0
            except Exception as e:
                logging.error(f"{name} retrieval failed: {e}")
                success = 0
                
            q_result[f"{name}_success"] = success
            scores[name] += success
            
        results.append(q_result)
        
    report = {
        "summary": {
            "total_queries": num_samples,
            "accuracy_normal_dense": f"{scores['normal_dense'] / num_samples * 100:.1f}%",
            "accuracy_normal_hybrid": f"{scores['normal_hybrid'] / num_samples * 100:.1f}%",
            "accuracy_ice_dense": f"{scores['ice_dense'] / num_samples * 100:.1f}%",
            "accuracy_ice_hybrid": f"{scores['ice_hybrid'] / num_samples * 100:.1f}%"
        },
        "details": results
    }
    
    with open("beir_scifact_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info("="*50)
    logging.info("BEIR EVALUATION COMPLETE")
    for k, v in report["summary"].items():
        logging.info(f"{k}: {v}")
    logging.info("="*50)

if __name__ == "__main__":
    main()
