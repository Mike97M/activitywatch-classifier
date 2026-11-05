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
    serve_parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable auto-reload on code change (default: False)",
    )

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
        "--include-audible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat tabs playing audio (movies/videos) as active, excluding music services (default: True)",
    )
    run_parser.add_argument(
        "--afk-bucket",
        default=None,
        help="Explicit AFK bucket ID to filter against (auto-detected by default)",
    )
    run_parser.add_argument(
        "--collect-training-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect novel PII-sanitized activities to expand training dataset (default: True)",
    )
    run_parser.add_argument(
        "--auto-commit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically commit updated training data to Git (default: False)",
    )
    run_parser.add_argument(
        "--time-range",
        choices=["24h", "today", "7d", "week", "30d", "month", "all"],
        default=None,
        help="Time window: 24h/day, today, 7d/week, 30d/month, all (overrides count limit)",
    )
    run_parser.add_argument(
        "--start",
        default=None,
        help="Custom start ISO-8601 timestamp (e.g. 2026-11-01T00:00:00Z)",
    )
    run_parser.add_argument(
        "--end",
        default=None,
        help="Custom end ISO-8601 timestamp (e.g. 2026-11-06T00:00:00Z)",
    )

    # Status command
    status_parser = subparsers.add_parser("status", help="Show ActivityWatch & Classifier status")

    # Commit training data command
    commit_parser = subparsers.add_parser(
        "commit-data", help="Stage and commit data/training_data.jsonl to Git"
    )
    commit_parser.add_argument(
        "--message",
        "-m",
        default=None,
        help="Commit message for training data",
    )

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
    view_parser.add_argument(
        "--time-range",
        choices=["24h", "today", "7d", "week", "30d", "month", "all"],
        default=None,
        help="Filter viewed activities by time window: 24h, today, 7d, 30d, all",
    )
    view_parser.add_argument(
        "--start",
        default=None,
        help="Custom start ISO-8601 timestamp",
    )
    view_parser.add_argument(
        "--end",
        default=None,
        help="Custom end ISO-8601 timestamp",
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

    # Sync categories command
    sync_parser = subparsers.add_parser(
        "sync-categories",
        help="Synchronize classifier taxonomy and learned rules into ActivityWatch native settings",
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
        uvicorn.run("app:app", host=host, port=port, reload=getattr(args, "reload", False))
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

    if args.command == "sync-categories":
        from category_sync import sync_taxonomy_to_activitywatch
        client = AWClient()
        clf = FastClassifier()
        print("Synchronizing classifier categories into ActivityWatch settings (/api/0/settings/classes)...")
        res = sync_taxonomy_to_activitywatch(client, clf)
        if res.get("success"):
            print(f"Success! Synchronized {res['classes_count']} category rules into ActivityWatch.")
            print(f"Categories: {', '.join(res['categories'])}")
            print("\nYou can now open http://localhost:5600/#/activity to view the category tree and sunburst!")
        else:
            print("Failed to sync categories to ActivityWatch.")
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

        import git_sync
        git_st = git_sync.get_training_data_status()
        print("\nGit Dataset Status:")
        print(f"  Dataset File:       {git_st['file']}")
        print(f"  Total Samples:      {git_st['sample_count']}")
        print(f"  Uncommitted:        {git_st['is_modified_or_untracked']} ({git_st['status_code']})")
        print(f"  Last Commit:        {git_st['last_commit']}")
        return

    if args.command == "commit-data":
        import git_sync
        print("Staging and committing data/training_data.jsonl...")
        res = git_sync.commit_training_data(message=args.message)
        if res.get("success"):
            if res.get("committed"):
                print(f"Success! [{res['commit_hash']}] {res['message']}")
            else:
                print(f"Clean: {res.get('message')}")
            print(f"Total dataset samples: {res.get('sample_count')}")
        else:
            print(f"Commit failed: {res.get('error') or res.get('message')}")
        return

    if args.command == "run":
        print(f"Starting non-destructive pipeline...")
        print(f"  Source Buckets: {args.source}")
        print(f"  Target Bucket:  {args.target}")
        print(f"  Event Limit:    {args.limit}")
        print(f"  Confidence Cut: {args.threshold}")
        print(f"  AFK Guard:      {'Enabled' if args.filter_afk else 'Disabled'}")
        if args.filter_afk:
            print(f"  Audible Media:  {'Active (except music)' if args.include_audible else 'Disabled'}")
        if args.afk_bucket:
            print(f"  AFK Bucket:     {args.afk_bucket}")
        if args.time_range:
            print(f"  Time Window:    {args.time_range}")
        if args.start:
            print(f"  Start Time:     {args.start}")
        if args.end:
            print(f"  End Time:       {args.end}")
        print(f"  Gemma Backend:  {args.backend}")
        print(f"  Auto-Expand:    {args.collect_training_data}")
        print(f"  Auto-Commit:    {args.auto_commit}\n")

        oracle = GemmaOracle(backend=args.backend, local_gguf_path=args.model_path)
        pipeline = ClassificationPipeline(gemma_oracle=oracle, confidence_threshold=args.threshold)
        results = pipeline.run(
            source_bucket_ids=args.source,
            target_bucket_id=args.target,
            limit_per_bucket=args.limit,
            time_range=args.time_range,
            start_time=args.start,
            end_time=args.end,
            confidence_threshold=args.threshold,
            filter_afk=args.filter_afk,
            include_audible=args.include_audible,
            afk_bucket_id=args.afk_bucket,
            collect_training_data=args.collect_training_data,
            auto_commit_git=args.auto_commit,
        )

        print("\nPipeline Run Completed!")
        if results.get("afk_filtering_enabled"):
            print(f"  AFK Buckets Used:        {', '.join(results.get('afk_buckets_used', [])) or 'None'}")
            if results.get("audible_intervals_used", 0) > 0:
                print(f"  Audible Media Intervals: {results.get('audible_intervals_used')}")
            print(f"  Raw Events Fetched:      {results.get('total_raw_events_fetched', 0)}")
            print(f"  Idle/AFK Events Dropped: {results.get('afk_events_filtered_out', 0)}")
        print(f"  Active Events Processed: {results['total_events_processed']}")
        print(f"  Events Saved to '{args.target}': {results['events_saved_to_new_bucket']}")
        print(f"  Fast Classifier calls:   {results['fast_classifier_calls']}")
        print(f"  Gemma Oracle calls:      {results['gemma_oracle_calls']}")
        if results.get("new_training_samples_added") is not None:
            print(f"  New Training Samples:    {results['new_training_samples_added']}")
        if results.get("git_commit"):
            gc = results["git_commit"]
            if gc.get("committed"):
                print(f"  Git Commit:              [{gc.get('commit_hash')}] {gc.get('message')}")
            else:
                print(f"  Git Commit:              {gc.get('message')}")
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

            from afk_filter import resolve_time_range
            v_start, v_end = resolve_time_range(
                time_range=args.time_range,
                start_time=args.start,
                end_time=args.end,
            )
            v_limit = 0 if (args.time_range and args.time_range != "count" and limit == 25) else limit
            events = client.get_events(target, limit=v_limit, start=v_start, end=v_end)
            if not events:
                time_desc = f" within '{args.time_range}'" if args.time_range else ""
                print(f"Bucket '{target}' exists but contains no categorized events{time_desc}.")
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
            print(f"{'Time':8} | {'App':12} | {'Category > Subcategory':30} | {'Conf':5} | {'Title / URL'}")
            print("-" * 105)
            for ev in events[:limit]:
                ts = (ev.get("timestamp") or "")[11:19]
                data = ev.get("data", {})
                app = (data.get("app") or "Unknown")[:12]
                cat = data.get("category") or "Uncategorized"
                sub = data.get("subcategory") or "General"
                full = f"{cat} > {sub}"[:30]
                conf = f"{data.get('confidence', 0.0):.2f}"
                url = data.get("url")
                title = data.get("title") or ""
                target_str = url if url else title
                if len(target_str) > 42:
                    target_str = target_str[:39] + "..."
                print(f"{ts:8} | {app:12} | {full:30} | {conf:5} | {target_str}")
            print()
        except Exception as e:
            print(f"Error inspecting target bucket '{target}': {e}")
        return


if __name__ == "__main__":
    main()
