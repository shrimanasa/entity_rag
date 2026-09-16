#!/usr/bin/env python3
import sys
import os

# Ensure local cache for matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch", "matplotlib")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.api import EntityAwareRAG

def print_result(res: dict):
    print("\n" + "=" * 80)
    print(f"  QUERY: {res['query']}")
    print("=" * 80)
    
    print("\n[1] ENTITY RESOLUTION:")
    print(f"  • Mentions Detected:    {res['detected_mentions'] if res['detected_mentions'] else 'None'}")
    print(f"  • Canonical Entity ID:  {res['canonical_entity_id']}")
    print(f"  • Primary Name:         {res['canonical_primary_name']}")
    print(f"  • Linking Confidence:   {res['linking_confidence']}")
    
    print("\n[2] ENTITY-AWARE RAG ANSWER (Condition 5 - Full System):")
    print(f"  \"{res['answer']}\"")
    print(f"  --> Cited Evidence Passages: {res['cited_passages']}")

    print("\n[3] STANDARD BASELINE RAG ANSWER (Condition 3 - BM25 + Dense):")
    print(f"  \"{res['baseline_answer']}\"")
    print(f"  --> Cited Evidence Passages: {res['baseline_cited_passages']}")

    if res['top_passages']:
        top_p = res['top_passages'][0]
        print("\n[4] TOP RETRIEVED PASSAGE EXCERPT:")
        print(f"  [{top_p['passage_id']}] {top_p.get('title', '')}")
        excerpt = top_p.get('text', '')[:250] + "..." if len(top_p.get('text', '')) > 250 else top_p.get('text', '')
        print(f"  {excerpt}")
    print("=" * 80 + "\n")

def main():
    print("Initializing Entity-Aware Materials-Science RAG System...")
    rag = EntityAwareRAG()
    rag.initialize()
    print("System Ready!\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        res = rag.ask(query)
        print_result(res)
    else:
        print("Entering interactive mode. Type 'exit' or 'quit' to quit.\n")
        sample_queries = [
            "What is the critical transition temperature of YBCO?",
            "What are the superconducting properties of 123 cuprate?",
            "Tell me about the crystal structure of rutile titania.",
            "What is the electrochemical performance of LiFePO4 cathode?",
            "Synthesis and physical properties of 2212 cuprate."
        ]
        print("Sample queries to try:")
        for sq in sample_queries:
            print(f"  - {sq}")
        print()

        while True:
            try:
                user_input = input("Enter material query > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                res = rag.ask(user_input)
                print_result(res)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

if __name__ == "__main__":
    main()
