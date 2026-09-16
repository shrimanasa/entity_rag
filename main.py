#!/usr/bin/env python3
"""
Main entrypoint for Entity-Aware Materials-Science RAG.
Provides both full experimental pipeline execution and interactive querying.
"""
import sys
import os
import argparse

# Ensure local cache for matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch", "matplotlib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline import run_end_to_end_pipeline
from query import print_result
from src.api import EntityAwareRAG

def main():
    parser = argparse.ArgumentParser(
        description="Entity-Aware Retrieval-Augmented Generation for Materials Science"
    )
    parser.add_argument(
        "--pipeline", "-p", action="store_true",
        help="Run the complete experimental pipeline (Stages 0-7) and generate reports"
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Run a specific materials science query through the Entity-Aware RAG system"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start an interactive query session in the terminal"
    )

    args = parser.parse_args()

    if args.pipeline:
        print("Starting full experimental pipeline (Stages 0 through 7)...")
        run_end_to_end_pipeline()
    elif args.query:
        print("Initializing EntityAwareRAG...")
        rag = EntityAwareRAG()
        rag.initialize()
        res = rag.ask(args.query)
        print_result(res)
    elif args.interactive:
        import query
        query.main()
    else:
        # Default behavior when run directly without arguments: run interactive or ask user
        print("=" * 80)
        print("  ENTITY-AWARE MATERIALS-SCIENCE RAG")
        print("=" * 80)
        print("Usage:")
        print("  python main.py --query \"What is the critical temperature of YBCO?\"")
        print("  python main.py --interactive   (or python query.py)")
        print("  python main.py --pipeline      (or python run_pipeline.py)")
        print("=" * 80)
        print("\nStarting interactive query session by default...\n")
        import query
        query.main()

if __name__ == "__main__":
    main()
