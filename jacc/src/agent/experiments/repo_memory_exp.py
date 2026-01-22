#!/usr/bin/env python3
"""Experiment: Compare agent performance with/without memory.

Groups SWE-bench tasks by repo and sorts by creation time,
then runs them sequentially to measure memory effectiveness.

Usage:
    # Analyze dataset structure
    python -m agent.experiments.repo_memory_exp --analyze
    
    # Run experiment with memory
    python -m agent.experiments.repo_memory_exp --run --memory --repo django -o ./exp_memory
    
    # Run experiment without memory (baseline)
    python -m agent.experiments.repo_memory_exp --run --repo django -o ./exp_baseline
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def analyze_dataset(subset: str = "lite", split: str = "test"):
    """Analyze SWE-bench dataset: group by repo, sort by time."""
    
    dataset_mapping = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }
    
    dataset_path = dataset_mapping.get(subset, subset)
    logger.info(f"Loading: {dataset_path}, split: {split}")
    
    instances = list(load_dataset(dataset_path, split=split))
    logger.info(f"Total instances: {len(instances)}")
    
    # Group by repo
    by_repo = defaultdict(list)
    for inst in instances:
        # instance_id format: "owner__repo-issue_number"
        # e.g., "django__django-11039"
        instance_id = inst["instance_id"]
        parts = instance_id.split("-")
        repo = parts[0]  # "django__django"
        
        # Parse created_at if available
        created_at = inst.get("created_at", "")
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except:
                dt = None
        else:
            dt = None
        
        by_repo[repo].append({
            "instance_id": instance_id,
            "created_at": created_at,
            "datetime": dt,
            "problem_statement": inst.get("problem_statement", "")[:100],
        })
    
    # Sort each repo's issues by time
    for repo, issues in by_repo.items():
        by_repo[repo] = sorted(
            issues, 
            key=lambda x: x["datetime"] or datetime.min
        )
    
    # Print analysis
    print("\n" + "=" * 70)
    print("SWE-BENCH ANALYSIS BY REPO")
    print("=" * 70)
    
    # Sort repos by number of issues (descending)
    sorted_repos = sorted(by_repo.items(), key=lambda x: -len(x[1]))
    
    for repo, issues in sorted_repos:
        print(f"\n📁 {repo}: {len(issues)} issues")
        
        if issues[0]["datetime"]:
            first = issues[0]["created_at"][:10]
            last = issues[-1]["created_at"][:10]
            print(f"   Time range: {first} → {last}")
        
        # Show first 3 issues
        for i, issue in enumerate(issues[:3]):
            print(f"   [{i+1}] {issue['instance_id']}")
            print(f"       {issue['problem_statement'][:60]}...")
        
        if len(issues) > 3:
            print(f"   ... and {len(issues) - 3} more")
    
    print("\n" + "=" * 70)
    print(f"Total repos: {len(by_repo)}")
    print("=" * 70)
    
    return by_repo


def get_repo_instances(
    repo: str,
    subset: str = "lite",
    split: str = "test",
    limit: int | None = None,
) -> list[dict]:
    """Get instances for a specific repo, sorted by creation time."""
    
    dataset_mapping = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified", 
        "full": "princeton-nlp/SWE-bench",
    }
    
    dataset_path = dataset_mapping.get(subset, subset)
    instances = list(load_dataset(dataset_path, split=split))
    
    # Filter by repo
    filtered = []
    for inst in instances:
        instance_id = inst["instance_id"]
        if instance_id.startswith(repo) or repo in instance_id:
            # Parse created_at
            created_at = inst.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except:
                    dt = datetime.min
            else:
                dt = datetime.min
            
            inst["_datetime"] = dt
            filtered.append(inst)
    
    # Sort by time
    filtered = sorted(filtered, key=lambda x: x["_datetime"])
    
    if limit:
        filtered = filtered[:limit]
    
    logger.info(f"Found {len(filtered)} instances for repo: {repo}")
    
    return filtered


def run_experiment(
    repo: str,
    output_dir: Path,
    with_memory: bool = False,
    model: str = "gemini/gemini-2.5-flash",
    subset: str = "lite",
    split: str = "test",
    limit: int | None = None,
):
    """Run experiment on repo instances sequentially."""
    
    from agent import AgentConfig, run_agent
    from agent.models import get_model
    from agent.environments import get_environment
    from agent.memory import (
        get_memory_client, MemoryConfig, NoOpMemoryClient,
        run_async, close_async_runner
    )
    
    # Get sorted instances
    instances = get_repo_instances(repo, subset, split, limit)
    
    if not instances:
        logger.error(f"No instances found for repo: {repo}")
        return
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = AgentConfig.default()
    
    # Memory setup
    memory_client = None
    bank_id = f"exp_{repo.replace('__', '_')}"  # e.g., "exp_django_django"
    
    if with_memory:
        try:
            memory_config = MemoryConfig(bank_id=bank_id)
            memory_client = get_memory_client(memory_config, mode="direct")
            run_async(memory_client.initialize(), timeout=60)
            logger.info(f"✅ Memory enabled: bank_id={bank_id}")
        except Exception as e:
            logger.warning(f"⚠️ Memory init failed: {e}")
            memory_client = NoOpMemoryClient()
    
    # Results tracking
    results = {
        "repo": repo,
        "with_memory": with_memory,
        "bank_id": bank_id if with_memory else None,
        "model": model,
        "instances": [],
    }
    
    # Process each instance
    try:
        for i, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            problem_statement = instance["problem_statement"]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{len(instances)}] {instance_id}")
            logger.info(f"{'='*60}")
            
            # Create model and environment
            model_provider = get_model("api", model_name=model)
            
            # Get docker image
            from agent.run_swebench import get_swebench_docker_image
            docker_image = get_swebench_docker_image(instance)
            
            env = get_environment({
                "type": "docker",
                "image": docker_image,
                "timeout": 120,
                "cwd": "/testbed",
            })
            
            try:
                # Run agent
                final_state = run_agent(
                    problem_statement,
                    config,
                    model_provider,
                    env,
                    memory_client=memory_client,
                    instance_id=instance_id,
                )
                
                exit_status = final_state.get("exit_status", "unknown")
                steps = final_state.get("step_count", 0)
                cost = final_state.get("total_cost", 0)
                
                # Get patch
                diff_result = env.execute("git diff")
                patch = diff_result.get("output", "")
                
                # Save results
                result = {
                    "instance_id": instance_id,
                    "exit_status": exit_status,
                    "steps": steps,
                    "cost": cost,
                    "patch_length": len(patch),
                }
                results["instances"].append(result)
                
                # Save trajectory
                instance_dir = output_dir / instance_id
                instance_dir.mkdir(parents=True, exist_ok=True)
                
                traj = {
                    "instance_id": instance_id,
                    "exit_status": exit_status,
                    "steps": steps,
                    "cost": cost,
                    "messages": [
                        {"role": m["role"], "content": m["content"]}
                        for m in final_state.get("messages", [])
                    ],
                }
                (instance_dir / f"{instance_id}.traj.json").write_text(
                    json.dumps(traj, indent=2)
                )
                
                logger.info(f"✅ {instance_id}: {exit_status}, steps={steps}, cost=${cost:.4f}")
                
            except Exception as e:
                logger.error(f"❌ {instance_id}: {e}")
                results["instances"].append({
                    "instance_id": instance_id,
                    "exit_status": "error",
                    "error": str(e),
                })
            finally:
                env.cleanup()
    
    finally:
        # Cleanup
        if memory_client and hasattr(memory_client, 'close'):
            try:
                run_async(memory_client.close(), timeout=5)
                close_async_runner()
            except:
                pass
    
    # Save final results
    results_file = output_dir / "experiment_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Repo: {repo}")
    print(f"Memory: {'✅ Enabled' if with_memory else '❌ Disabled'}")
    print(f"Instances: {len(results['instances'])}")
    
    success = sum(1 for r in results["instances"] if r.get("exit_status") == "success")
    total_cost = sum(r.get("cost", 0) for r in results["instances"])
    total_steps = sum(r.get("steps", 0) for r in results["instances"])
    
    print(f"Success rate: {success}/{len(results['instances'])} ({100*success/len(results['instances']):.1f}%)")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total steps: {total_steps}")
    print(f"Results saved to: {results_file}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Memory effectiveness experiment")
    
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset structure")
    parser.add_argument("--run", action="store_true", help="Run experiment")
    
    parser.add_argument("--repo", type=str, default="django__django", help="Repo to filter")
    parser.add_argument("--memory", action="store_true", help="Enable memory")
    parser.add_argument("--subset", type=str, default="lite", help="SWE-bench subset")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, help="Limit number of instances")
    parser.add_argument("-m", "--model", type=str, default="gemini/gemini-2.5-flash")
    parser.add_argument("-o", "--output", type=str, default="./exp_output")
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_dataset(args.subset, args.split)
    elif args.run:
        run_experiment(
            repo=args.repo,
            output_dir=Path(args.output),
            with_memory=args.memory,
            model=args.model,
            subset=args.subset,
            split=args.split,
            limit=args.limit,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
