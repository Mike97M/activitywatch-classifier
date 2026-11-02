"""ActivityWatch Gemma Classifier CLI and Server Entry Point."""

import argparse
import sys
import uvicorn
from aw_client import AWClient
from classifier import FastClassifier
from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_TARGET_BUCKET_ID,
    WEB_HOST,
    WEB_PORT,
)
from gemma_oracle import GemmaOracle
from pipeline import ClassificationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="ActivityWatch Active Learning Classifier using Gemma E2B Oracle."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Serve UI command
    serve_parser = subparsers.add_parser("serve", help="Launch Web UI and API server")
    serve_parser.add_argument("--host", default=WEB_HOST, help="Host to bind server")
    serve_parser.add_argument("--port", type=int, default=WEB_PORT, help="Port to bind server")

    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run classification pipeline non-destructively")
    run_parser.add_argument(
        "--source",
        "-s",
        nargs="+",
        required=True,
        help="One or more source bucket IDs to clone from",
    )
    run_parser.add_argument(
        "--target",
        "-t",
        default=DEFAULT_TARGET_BUCKET_ID,
        help="Target bucket ID to write classified activities to",
    )
    run_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=100,
        help="Limit of events to clone per source bucket (0 for all)",
    )
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence cutoff threshold (0.0 to 1.0)",
    )
    run_parser.add_argument(
        "--backend",
        default="heuristic",
        choices=["heuristic", "ollama", "gemini_api", "openai_compatible", "llama_cpp"],
        help="Gemma backend to use for low-confidence events",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show ActivityWatch & Classifier status")

    args = parser.parse_args()

    # Default to serve if no command provided
    if not args.command or args.command == "serve":
        host = getattr(args, "host", WEB_HOST)
        port = getattr(args, "port", WEB_PORT)
        print(f"\n========================================================")
        print(f" ActivityWatch Gemma Classifier Web UI")
        print(f" Dashboard running at: http://localhost:{port}")
        print(f"========================================================\n")
        uvicorn.run("app:app", host=host, port=port, reload=False)
        return

    if args.command == "status":
        client = AWClient()
        try:
            info = client.get_info()
            buckets = client.list_buckets()
            print("ActivityWatch Server: CONNECTED")
            print(f"  URL: {client.base_url}")
            print(f"  Hostname: {info.get('hostname')}")
            print(f"  Version: {info.get('version')}")
            print(f"  Buckets ({len(buckets)}):")
            for b_id, b_meta in buckets.items():
                print(f"    - {b_id} (type: {b_meta.get('type')})")
        except Exception as e:
            print(f"ActivityWatch Server: OFFLINE or UNREACHABLE ({e})")

        clf = FastClassifier()
        stats = clf.get_stats()
        print("\nClassifier Status:")
        print(f"  Trained: {stats['is_trained']}")
        print(f"  Training samples: {stats['total_training_samples']}")
        print(f"  Categories: {stats['all_configured_categories']}")
        return

    if args.command == "run":
        print(f"Starting non-destructive pipeline...")
        print(f"  Source Buckets: {args.source}")
        print(f"  Target Bucket:  {args.target}")
        print(f"  Event Limit:    {args.limit}")
        print(f"  Confidence Cut: {args.threshold}")
        print(f"  Gemma Backend:  {args.backend}\n")

        oracle = GemmaOracle(backend=args.backend)
        pipeline = ClassificationPipeline(gemma_oracle=oracle, confidence_threshold=args.threshold)
        results = pipeline.run(
            source_bucket_ids=args.source,
            target_bucket_id=args.target,
            limit_per_bucket=args.limit,
            confidence_threshold=args.threshold,
        )

        print("\nPipeline Run Completed!")
        print(f"  Total Processed: {results['total_events_processed']}")
        print(f"  Events Saved to '{args.target}': {results['events_saved_to_new_bucket']}")
        print(f"  Fast Classifier calls: {results['fast_classifier_calls']}")
        print(f"  Gemma Oracle calls:    {results['gemma_oracle_calls']}")
        print(f"  Category Distribution: {results['category_distribution']}")


if __name__ == "__main__":
    main()
