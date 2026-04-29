import os
import json
import logging
import time
import re
import glob
from enhancedRaptor import RetrievalAugmentation, RetrievalAugmentationConfig, ClusterTreeConfig
from enhancedRaptor.QAModels import GeminiQAModel
from enhancedRaptor.SummarizationModels import GemmaSummarizationModel
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def judge_answer(question, expected_answer, generated_answer, context, api_key):
    """Uses Gemini to judge the quality of a model's answer (strict rubric)."""
    prompt = f"""You are a STRICT expert grader for a Retrieval-Augmented Generation (RAG) system.

Question: {question}
Expected Baseline Answer: {expected_answer}

Model's Generated Answer:
{generated_answer}

Task: Score the Model's Generated Answer from 0 to 5 against the baseline.

Rubric:
- 5 = perfect semantic match: every key fact in the baseline appears, no contradictions, no fabricated details.
- 4 = matches all key facts but uses slightly different phrasing or adds minor neutral context.
- 3 = matches the main outcome but misses or muddles a secondary fact.
- 2 = partially correct: identifies the right scenario but contradicts itself, or hedges between two interpretations.
- 1 = mentions correct keywords but the overall conclusion contradicts the baseline.
- 0 = completely wrong, or returns an error message instead of an answer.

Critical failure modes — CAP THE SCORE AT 2 if any of these occur:
- The answer mixes BOTH a routine/low-stakes interpretation AND a high-stakes/catastrophic interpretation of the same procedure when the baseline is unambiguous about which one applies.
- The answer fabricates a "reveal", "red herring", "misinterpretation", or "however the text later clarifies" pattern that walks back the baseline outcome.
- The answer hedges with phrases like "though it could mean", "but actually", "in reality the consequence is..." when the baseline is firm.
- The answer pulls facts from the wrong paragraph (e.g., gives the dramatic outcome when the question describes a routine setting, or vice versa).

Output valid JSON ONLY. Inside the rationale string, DO NOT use double quotes (use single quotes or no quotes if you need to quote a phrase, e.g. 'dead zone' instead of "dead zone").

{{
    "correctness_score": <int 0-5>,
    "rationale": "<one sentence, no double quotes inside>"
}}
"""
    try:
        import time
        time.sleep(3)
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemma-3-27b-it",
            contents=prompt
        )
        
        text = response.text.strip()
        # Robustly parse JSON from markdown codeblocks if they exist
        import re
        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        else:
            json_match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()

        # Try strict JSON first, fall back to regex extraction if the LLM
        # emitted unescaped double-quotes inside the rationale string.
        result = None
        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            score_match = re.search(r'"correctness_score"\s*:\s*(\d+)', text)
            rat_match = re.search(r'"rationale"\s*:\s*"(.*)"\s*\}', text, re.DOTALL)
            if score_match:
                result = {
                    "correctness_score": int(score_match.group(1)),
                    "rationale": (
                        rat_match.group(1).strip()
                        if rat_match
                        else "Rationale unparseable (regex fallback)."
                    ),
                }

        if result is None:
            logging.error(f"Could not parse judge output. Raw text head: {text[:200]}")
            return 0, f"Failed to parse judge output: {text[:200]}"

        c_score = result.get('correctness_score', 0)
        rationale = result.get('rationale', "Failed to parse rationale.")
        return c_score, rationale

    except Exception as e:
        logging.error(f"Grader LLM failed: {e}")
        return 0, f"Error calling judge: {e}"

def setup_raptor_configs():
    summarization_model = GemmaSummarizationModel(model="gemma-3-27b-it")
    qa_model = GeminiQAModel(model="gemma-3-27b-it")
    
    # Normal Config
    normal_tb_config = ClusterTreeConfig(
        chunking_strategy="token",
        use_ice=False,
        max_tokens=100,
        summarization_model=summarization_model
    )
    normal_config = RetrievalAugmentationConfig(tree_builder_config=normal_tb_config, qa_model=qa_model)
    
    # ICE Config
    ice_tb_config = ClusterTreeConfig(
        chunking_strategy="semantic",
        chunking_threshold=0.8,
        use_ice=True,
        max_tokens=100,
        summarization_model=summarization_model
    )
    ice_config = RetrievalAugmentationConfig(tree_builder_config=ice_tb_config, qa_model=qa_model)
    
    return normal_config, ice_config

def evaluate_document(doc_id, text_path, json_path, normal_config, ice_config, api_key):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    with open(json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
        
    questions = qa_data.get("qa_pairs", [])
    if not questions:
        logging.warning(f"No questions found for {doc_id}")
        return []

    os.makedirs("trees", exist_ok=True)
    normal_tree_path = f"trees/{doc_id}_normal_tree.pkl"
    ice_tree_path = f"trees/{doc_id}_ice_tree.pkl"
    
    logging.info(f"\n{'='*50}\nEvaluating Document: {doc_id}\n{'='*50}")
    
    # Build or Load Normal Tree
    logging.info(f"Building/Loading Normal RAPTOR Tree for {doc_id}...")
    if not os.path.exists(normal_tree_path):
        normal_ra = RetrievalAugmentation(config=normal_config)
        normal_ra.add_documents(text)
        normal_ra.save(normal_tree_path)
    else:
        normal_ra = RetrievalAugmentation(config=normal_config, tree=normal_tree_path)
        
    # Build or Load ICE Tree
    logging.info(f"Building/Loading ICE RAPTOR Tree for {doc_id}...")
    if not os.path.exists(ice_tree_path):
        ice_ra = RetrievalAugmentation(config=ice_config)
        ice_ra.add_documents(text)
        ice_ra.save(ice_tree_path)
    else:
        ice_ra = RetrievalAugmentation(config=ice_config, tree=ice_tree_path)

    doc_results = []
    
    for idx, qa in enumerate(questions):
        question = qa['question']
        expected = qa['expected_answer']
        
        logging.info(f"\n--- Question {idx+1}: {question} ---")
        
        # Normal Raptor
        logging.info("Querying Normal RAPTOR...")
        try:
            n_context, _ = normal_ra.retrieve(question)
            normal_ans = normal_ra.qa_model.answer_question(n_context, question)
        except Exception as e:
            n_context = "ERROR"
            normal_ans = f"ERROR: {e}"
            
        time.sleep(3) # Rate limit protection
        
        # ICE Raptor
        logging.info("Querying ICE RAPTOR...")
        try:
            i_context, _ = ice_ra.retrieve(question)
            ice_ans = ice_ra.qa_model.answer_question(i_context, question)
        except Exception as e:
            i_context = "ERROR"
            ice_ans = f"ERROR: {e}"
            
        time.sleep(3)
        
        # Judging
        logging.info("Judging answers...")
        n_c, n_rationale = judge_answer(question, expected, normal_ans, n_context, api_key)
        i_c, i_rationale = judge_answer(question, expected, ice_ans, i_context, api_key)
        
        doc_results.append({
            "question": question,
            "expected_answer": expected,
            "normal_raptor": {
                "answer": normal_ans,
                "correctness": n_c,
                "rationale": n_rationale
            },
            "ice_raptor": {
                "answer": ice_ans,
                "correctness": i_c,
                "rationale": i_rationale
            }
        })
        
        logging.info(f"[Normal] Correctness: {n_c}/5")
        logging.info(f"[ICE]    Correctness: {i_c}/5")
        
    return doc_results

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable must be set!")

    user_input = input("Enter doc IDs to run (e.g., doc_2 doc_8 or doc_1,doc_3) or press Enter to run ALL: ").strip()
    
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        logging.error(f"Dataset directory '{dataset_dir}' not found.")
        return

    files_to_run = []
    if user_input:
        # Support comma-separated, space-separated, or mixed
        doc_ids = [d.strip() for d in user_input.replace(",", " ").split() if d.strip()]
        for doc_id in doc_ids:
            json_path = os.path.join(dataset_dir, f"{doc_id}.json")
            txt_path = os.path.join(dataset_dir, f"{doc_id}.txt")
            if os.path.exists(json_path) and os.path.exists(txt_path):
                files_to_run.append((doc_id, txt_path, json_path))
            else:
                logging.error(f"Could not find both {json_path} and {txt_path}. Skipping {doc_id}.")
    else:
        # Run ALL docs in the dataset folder
        for json_path in sorted(glob.glob(os.path.join(dataset_dir, "*.json"))):
            base_name = os.path.basename(json_path)
            doc_id = os.path.splitext(base_name)[0]
            txt_path = os.path.join(dataset_dir, f"{doc_id}.txt")
            if os.path.exists(txt_path):
                files_to_run.append((doc_id, txt_path, json_path))
            else:
                logging.warning(f"Found {json_path} but missing {txt_path}. Skipping.")

    if not files_to_run:
        logging.error("No valid dataset files found to process.")
        return

    normal_config, ice_config = setup_raptor_configs()
    
    report_path = "evaluation_report.json"
    
    # Load existing results if the report already exists (to preserve other docs)
    existing_results = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
                existing_results = existing_report.get("detailed_results", {})
                logging.info(f"Loaded existing report with {len(existing_results)} documents.")
        except Exception as e:
            logging.warning(f"Could not load existing report: {e}. Starting fresh.")

    for doc_id, txt_path, json_path in files_to_run:
        doc_res = evaluate_document(doc_id, txt_path, json_path, normal_config, ice_config, api_key)
        existing_results[doc_id] = doc_res  # Overwrite only re-run docs

    # Recompute summary across ALL documents in the merged results
    total_q = 0
    normal_total_c = 0
    ice_total_c = 0
    for doc_id, doc_res in existing_results.items():
        for res in doc_res:
            total_q += 1
            normal_total_c += res["normal_raptor"]["correctness"]
            ice_total_c += res["ice_raptor"]["correctness"]
    
    final_report = {
        "summary": {
            "total_documents": len(existing_results),
            "total_questions": total_q,
            "normal_raptor_avg_correctness": round(normal_total_c / total_q, 2) if total_q > 0 else 0,
            "ice_raptor_avg_correctness": round(ice_total_c / total_q, 2) if total_q > 0 else 0,
            "normal_raptor_accuracy_percentage": round((normal_total_c / (total_q * 5)) * 100, 2) if total_q > 0 else 0,
            "ice_raptor_accuracy_percentage": round((ice_total_c / (total_q * 5)) * 100, 2) if total_q > 0 else 0
        },
        "detailed_results": existing_results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
        
    logging.info("\n" + "="*50)
    logging.info("FINAL EVALUATION METRICS")
    logging.info(f"Total Questions Evaluated: {total_q}")
    logging.info(f"Normal RAPTOR Avg Correctness: {final_report['summary']['normal_raptor_avg_correctness']}/5 ({final_report['summary']['normal_raptor_accuracy_percentage']}%)")
    logging.info(f"ICE RAPTOR Avg Correctness   : {final_report['summary']['ice_raptor_avg_correctness']}/5 ({final_report['summary']['ice_raptor_accuracy_percentage']}%)")
    logging.info(f"Detailed report saved to {report_path}")
    logging.info("="*50)

if __name__ == "__main__":
    main()
