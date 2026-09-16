import os
import sys
import json
import socketserver
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, List

sys.path.insert(0, os.path.abspath("."))

from src.chem_utils import calculate_lexical_divergence, canonicalize_hill_notation, SharedSemanticEmbedder
from src.stage0_ground_truth import load_alias_dictionary
from src.stage1_mention_detection import MentionDetector
from src.stage2_entity_linker import EntityLinker
from src.stage3_canonical_graph import CanonicalGraphBuilder, run_stage3_graph_construction
from src.stage4_alias_labeler import AliasDivergenceLabeler
from src.stage5_retriever import RetrievalPipeline
from src.stage7_generator import RAGAnswerGenerator

class EntityAwareRAGService:
    corpus = []
    qa_dataset = []
    builder = None
    tagged_corpus = []
    detector = None
    linker = None
    pipeline = None
    labeler = None
    embedder = None
    generator = None

    @classmethod
    def initialize(cls):
        print("Initializing Entity-Aware RAG Engine...")
        with open("data/corpus.json", "r", encoding="utf-8") as f:
            cls.corpus = json.load(f)
        with open("data/qa_dataset.json", "r", encoding="utf-8") as f:
            cls.qa_dataset = json.load(f)

        alias_dict = load_alias_dictionary("data/alias_dictionary.json")

        # Fit shared semantic embedder on unified universe (identical to run_pipeline.py)
        training_texts = []
        for p in cls.corpus:
            training_texts.append(f"{p.get('title','')} {p.get('abstract','')} {p.get('text','')}")
        for ent_id, info in alias_dict.items():
            all_aliases = []
            for cat, names in info.get("categories", {}).items():
                all_aliases.extend(names)
            training_texts.append(f"{info.get('primary_name','')} {' '.join(all_aliases)}")
        for q in cls.qa_dataset:
            training_texts.append(q["query"])

        cls.embedder = SharedSemanticEmbedder(n_components=128, random_state=42)
        cls.embedder.fit(training_texts)

        cls.builder, cls.tagged_corpus = run_stage3_graph_construction(cls.corpus, alias_dict_path="data/alias_dictionary.json")
        cls.detector = MentionDetector(alias_dict=alias_dict, embedder=cls.embedder)
        
        # Load tuned grid search parameters if available
        alpha, beta, gamma, tau = 0.375, 0.50, 0.125, 0.30
        if os.path.exists("results/grid_search_results.json"):
            try:
                with open("results/grid_search_results.json") as f:
                    grid_p = json.load(f)
                    alpha = grid_p.get("alpha", alpha)
                    beta = grid_p.get("beta", beta)
                    gamma = grid_p.get("gamma", gamma)
                    tau = grid_p.get("tau", tau)
            except Exception:
                pass

        cls.linker = EntityLinker(
            alias_dict_path="data/alias_dictionary.json",
            corpus_path="data/corpus.json",
            embedder=cls.embedder,
            alpha=alpha, beta=beta, gamma=gamma, tau=tau
        )
        cls.pipeline = RetrievalPipeline(
            tagged_corpus=cls.tagged_corpus,
            entity_linker=cls.linker,
            graph_builder=cls.builder,
            embedder=cls.embedder,
            k_ent=20
        )
        cls.labeler = AliasDivergenceLabeler(cls.builder)
        cls.generator = RAGAnswerGenerator()
        print("Entity-Aware RAG Engine initialized successfully!")

    @classmethod
    def process_user_query(cls, query_text: str) -> Dict:
        mentions = cls.detector.extract_mentions(query_text)
        
        canonical_id = "NIL"
        link_confidence = 0.0
        if mentions:
            canonical_id, link_confidence = cls.linker.link_mention(mentions[0])
            
        if canonical_id == "NIL":
            resolved = cls.builder.map_mention_to_canonical(query_text)
            if resolved != "NIL":
                canonical_id = resolved
                link_confidence = 0.85

        mapped_name = "NIL"
        known_aliases = []
        if canonical_id != "NIL" and canonical_id in cls.linker.alias_dict:
            ent_info = cls.linker.alias_dict[canonical_id]
            mapped_name = ent_info.get("primary_name", canonical_id)
            for cat, a_list in ent_info.get("categories", {}).items():
                known_aliases.extend(a_list)

        cond3_results = cls.pipeline.retrieve_condition(query_text, condition=3, top_k=5)
        cond4_results = cls.pipeline.retrieve_condition(query_text, condition=4, top_k=5)
        cond5_results = cls.pipeline.retrieve_condition(query_text, condition=5, top_k=5)

        def format_passages(retrieved_tuples):
            res = []
            for rank, (pid, score) in enumerate(retrieved_tuples, 1):
                p_data = cls.pipeline.corpus_dict.get(pid, {})
                res.append({
                    "rank": rank,
                    "passage_id": pid,
                    "score": round(score, 4),
                    "title": p_data.get("title", ""),
                    "abstract": p_data.get("abstract", ""),
                    "text": p_data.get("text", ""),
                    "canonical_entity": p_data.get("canonical_entity", "NIL"),
                    "tagged_entities": p_data.get("tagged_entities", {})
                })
            return res

        top_p5 = format_passages(cond5_results)
        top_p3 = format_passages(cond3_results)
        top_p4 = format_passages(cond4_results)

        top_evidence_text = top_p5[0]["text"] if top_p5 else ""
        lex_sim = calculate_lexical_divergence(query_text, top_evidence_text)
        
        if lex_sim >= 0.70:
            bucket = "No-mismatch"
        elif 0.20 <= lex_sim < 0.70:
            bucket = "Partial-mismatch"
        else:
            bucket = "Full alias-mismatch" if canonical_id != "NIL" else "Partial-mismatch"

        gen_res = cls.generator.generate(query_text, top_p5, max_passages=3)
        synthesized_answer = gen_res["answer"]

        return {
            "query": query_text,
            "entity_linking": {
                "detected_surface_form": mentions[0].get("surface_form", query_text) if mentions else query_text,
                "canonical_id": canonical_id,
                "primary_name": mapped_name,
                "confidence": round(link_confidence, 4),
                "threshold_tau": cls.linker.tau,
                "known_aliases": list(set(known_aliases))
            },
            "divergence_analysis": {
                "lexical_similarity": round(lex_sim, 4),
                "bucket": bucket
            },
            "retrieval_comparison": {
                "condition_3_baseline": top_p3,
                "condition_4_entity_only": top_p4,
                "condition_5_full_system": top_p5
            },
            "synthesized_answer": synthesized_answer
        }

from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer

class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

class RAGRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/benchmark":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            with open("results/results_table.json") as f:
                table_results = json.load(f)
            with open("results/coverage_report.json") as f:
                cov_report = json.load(f)
            with open("results/grid_search_results.json") as f:
                grid_params = json.load(f)

            response_data = {
                "results_table": table_results,
                "coverage_report": cov_report,
                "grid_params": grid_params
            }
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
            return

        if parsed.path in ("/", "/index.html"):
            self.path = "/ui/index.html"
        elif not parsed.path.startswith("/ui/"):
            self.path = "/ui" + parsed.path

        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/query":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            req_json = json.loads(post_data.decode('utf-8'))
            
            user_query = req_json.get("query", "").strip()
            if not user_query:
                self.send_error(400, "Query cannot be empty")
                return

            result = EntityAwareRAGService.process_user_query(user_query)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
            return

        self.send_error(404, "Endpoint not found")

def start_server(port=8000):
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    EntityAwareRAGService.initialize()
    server_address = ('', port)
    httpd = ReusableHTTPServer(server_address, RAGRequestHandler)
    print("================================================================================")
    print(f"  Entity-Aware RAG Web UI Server running at: http://localhost:{port}")
    print(f"  (Also accessible at http://127.0.0.1:{port})")
    print("================================================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    start_server()
