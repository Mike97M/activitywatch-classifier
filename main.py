"""ActivityWatch Gemma Classifier CLI and Server Entry Point."""

import argparse
import sys
import uvicorn
from aw_client import AWClient
from classifier import FastClassifier
from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LOCAL_GGUF_NAME,
    DEFAULT_LOCAL_GGUF_REPO,
    DEFAULT_TARGET_BUCKET_ID,
    LIGHT_LOCAL_GGUF_NAME,
    MODELS_DIR,
    WEB_HOST,
    WEB_PORT,
)
from gemma_oracle import GemmaOracle, download_gemma_weights
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
        default="local",
        choices=["local", "heuristic", "ollama", "gemini_api", "openai_compatible", "llama_cpp"],
        help="Gemma backend to use for low-confidence events (default: local)",
    )
    run_parser.add_argument(
        "--model-path",
        default=None,
        help="Explicit path to a local Gemma GGUF file",
    )
    run_parser.add_argument(
        "--filter-afk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out AFK periods and only consider activities when user was active (default: True)",
    )
    run_parser.add_argument(
        "--afk-bucket",
        default=None,
        help="Explicit AFK bucket ID to filter against (auto-detected by default)",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show ActivityWatch & Classifier status")

    # View bucket command
    view_parser = subparsers.add_parser("view", help="View categorized activities and taxonomy distribution")
    view_parser.add_argument(
        "--target",
        "-t",
        default=DEFAULT_TARGET_BUCKET_ID,
        help="Target bucket ID to view (default: aw-classifier-categorized)",
    )
    view_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=25,
        help="Number of recent events to display (default: 25)",
    )

    # Download Gemma command
    dl_parser = subparsers.add_parser(
        "download-gemma", help="Download Gemma 4 E2B GGUF weights directly from Hugging Face"
    )
    dl_parser.add_argument(
        "--variant",
        choices=["standard", "light"],
        default="standard",
        help="Gemma 4 E2B variant: 'standard' (Q4_K_M ~3.1 GB) or 'light' (Q3_K_M ~2.5 GB)",
    )

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

    if args.command == "download-gemma":
        repo = DEFAULT_LOCAL_GGUF_REPO
        filename = (
            LIGHT_LOCAL_GGUF_NAME if args.variant == "light" else DEFAULT_LOCAL_GGUF_NAME
        )

        print(f"Downloading Gemma 4 E2B ({filename}) from {repo}...")
        print(f"Target directory: {MODELS_DIR}")
        path = download_gemma_weights(repo_id=repo, filename=filename)
        print(f"\nDownload complete! Model ready for local in-process inference at:\n  {path}")
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

        oracle = GemmaOracle()
        local_info = oracle.get_local_model_info()
        print("\nGemma E2B Local Engine:")
        print(f"  Local GGUF Model Available: {local_info['available']}")
        if local_info["available"]:
            print(f"  Model File: {local_info['filename']} ({local_info['size_gb']} GB)")
            print(f"  Model Path: {local_info['path']}")
        else:
            print("  No local GGUF found in models/.")
            print("  To download directly, run: uv run python main.py download-gemma")
        return

    if args.command == "run":
        print(f"Starting non-destructive pipeline...")
        print(f"  Source Buckets: {args.source}")
        print(f"  Target Bucket:  {args.target}")
        print(f"  Event Limit:    {args.limit}")
        print(f"  Confidence Cut: {args.threshold}")
        print(f"  AFK Guard:      {'Enabled' if args.filter_afk else 'Disabled'}")
        if args.afk_bucket:
            print(f"  AFK Bucket:     {args.afk_bucket}")
        print(f"  Gemma Backend:  {args.backend}\n")

        oracle = GemmaOracle(backend=args.backend, local_gguf_path=args.model_path)
        pipeline = ClassificationPipeline(gemma_oracle=oracle, confidence_threshold=args.threshold)
        results = pipeline.run(
            source_bucket_ids=args.source,
            target_bucket_id=args.target,
            limit_per_bucket=args.limit,
            confidence_threshold=args.threshold,
            filter_afk=args.filter_afk,
            afk_bucket_id=args.afk_bucket,
        )

        print("\nPipeline Run Completed!")
        if results.get("afk_filtering_enabled"):
            print(f"  AFK Buckets Used:        {', '.join(results.get('afk_buckets_used', [])) or 'None'}")
            print(f"  Raw Events Fetched:      {results.get('total_raw_events_fetched', 0)}")
            print(f"  Idle/AFK Events Dropped: {results.get('afk_events_filtered_out', 0)}")
        print(f"  Active Events Processed: {results['total_events_processed']}")
        print(f"  Events Saved to '{args.target}': {results['events_saved_to_new_bucket']}")
        print(f"  Fast Classifier calls:   {results['fast_classifier_calls']}")
        print(f"  Gemma Oracle calls:      {results['gemma_oracle_calls']}")
        print(f"  Category Distribution:   {results['category_distribution']}")
        return

    if args.command == "view":
        client = AWClient()
        target = args.target
        limit = args.limit
        print(f"\n========================================================")
        print(f" ActivityWatch Target Bucket Viewer")
        print(f" Bucket: {target} | Server: {client.base_url}")
        print(f"========================================================\n")
        try:
            binfo = client.get_bucket(target)
            if not binfo:
                print(f"Bucket '{target}' does not exist in ActivityWatch yet.")
                print(f"Run the classifier pipeline first: uv run python main.py run -s <source-bucket>")
                return

            events = client.get_events(target, limit=limit)
            if not events:
                print(f"Bucket '{target}' exists but contains no categorized events.")
                return

            # Compute category and subcategory breakdown
            cat_counts = {}
            subcat_counts = {}
            for ev in events:
                data = ev.get("data", {})
                cat = data.get("category", "Uncategorized")
                sub = data.get("subcategory", "General")
                full = data.get("full_category", f"{cat} > {sub}")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                subcat_counts[full] = subcat_counts.get(full, 0) + 1

            print(f"Category Distribution ({len(events)} loaded events):")
            for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (cnt / len(events)) * 100
                bar = "█" * int(pct // 5)
                print(f"  {cat:24} {cnt:3d} ({pct:5.1f}%) {bar}")

            print(f"\nSubcategory Breakdown:")
            for full, cnt in sorted(subcat_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {full:40} : {cnt} events")

            print(f"\nRecent Categorized Activities (Latest {min(len(events), limit)}):")
            print(f"{'Time':8} | {'App':15} | {'Category':20} | {'Subcategory':20} | {'Conf':5} | {'Classified By'}")
            print("-" * 92)
            for ev in events[:limit]:
                ts = (ev.get("timestamp") or "")[11:19]
                data = ev.get("data", {})
                app = (data.get("app") or "Unknown")[:15]
                cat = (data.get("category") or "Uncategorized")[:20]
                sub = (data.get("subcategory") or "General")[:20]
                conf = f"{data.get('confidence', 0.0):.2f}"
                by = data.get("classified_by", "")[:20]
                print(f"{ts:8} | {app:15} | {cat:20} | {sub:20} | {conf:5} | {by}")
            print()
        except Exception as e:
            print(f"Error inspecting target bucket '{target}': {e}")
        return


if __name__ == "__main__":
    main()
