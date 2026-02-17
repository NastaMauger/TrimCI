"""
TrimCI Parallel Runner

Provides parallel execution of multiple TrimCI runs using spawn subprocesses.
This module MUST NOT import trimci at the top level to allow proper OMP control.
"""
import os
import time
from pathlib import Path
from multiprocessing import get_context

from .summary_logger import TrimCISummaryLogger


def _worker_single_run(args_tuple):
    """
    Worker function executed in a spawned subprocess.
    
    IMPORTANT: This function sets OMP_NUM_THREADS BEFORE importing trimci,
    ensuring the C++ OpenMP runtime respects the thread count.
    """
    (omp_threads, run_idx, h1, eri, n_alpha, n_beta, n_orb, 
     mol_name, args_dict, nuclear_repulsion, worker_folder) = args_tuple
    
    # 1. Set OMP_NUM_THREADS BEFORE any trimci import
    os.environ['OMP_NUM_THREADS'] = str(omp_threads)
    
    # 2. Now import trimci (C++ module loads here with correct OMP setting)
    from trimci.TrimCI_runner.trimci_driver import iterative_workflow
    from types import SimpleNamespace
    
    # 3. Reconstruct args namespace
    args = SimpleNamespace(**args_dict)
    
    # 4. Create worker-specific folder
    run_folder = os.path.join(worker_folder, f"run_{run_idx:03d}")
    Path(run_folder).mkdir(parents=True, exist_ok=True)
    
    # 5. Execute single run
    start_time = time.perf_counter()
    try:
        final_energy, dets, coeffs, details, run_args = iterative_workflow(
            h1, eri, n_alpha, n_beta, n_orb, mol_name,
            args, nuclear_repulsion, start_time, run_folder
        )
        elapsed = time.perf_counter() - start_time
        
        # Return serializable result
        return {
            'run_idx': run_idx,
            'success': True,
            'final_energy': final_energy,
            'dets': dets,
            'coeffs': coeffs,
            'details': details,
            'run_args': run_args,
            'elapsed': elapsed,
            'results_dir': run_folder
        }
    except Exception as e:
        return {
            'run_idx': run_idx,
            'success': False,
            'error': str(e),
            'results_dir': run_folder
        }


def run_trimci_main_calculation_parallel(
    h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion,
    folder=None,
    num_runs=None,
    n_parallel=None,
    omp_per_run=None
):
    """
    Parallel version of run_trimci_main_calculation.
    
    Executes multiple independent TrimCI runs in parallel using spawn subprocesses.
    Each subprocess has its own OMP_NUM_THREADS setting.
    
    Args:
        h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion:
            Same as run_trimci_main_calculation
        folder: Output directory (optional)
        num_runs: Total number of runs. Defaults to args.num_runs or 50.
        n_parallel: Number of parallel workers. Defaults to 8.
        omp_per_run: OMP threads per worker. If None, auto-calculated from
                     OMP_NUM_THREADS environment variable.
    
    Returns:
        Same as run_trimci_main_calculation:
        (best_energy, best_dets, best_coeffs, best_details, best_args)
    
    Example:
        # 128 cores total: 8 parallel workers × 16 OMP threads each
        result = run_trimci_main_calculation_parallel(
            h1, eri, n_alpha, n_beta, n_orb, mol_name, args, nuclear_repulsion,
            num_runs=50, n_parallel=8, omp_per_run=16
        )
    """
    # Get parameters
    if num_runs is None:
        num_runs = getattr(args, 'num_runs', 50)
    if n_parallel is None:
        n_parallel = getattr(args, 'n_parallel', 8)
    if omp_per_run is None:
        # Auto-calculate: divide total OMP threads by number of parallel workers
        total_omp = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1))
        omp_per_run = max(1, total_omp // n_parallel)
    
    # Setup output folder (use unique timestamp with PID to avoid Slurm array job collisions)
    from .trimci_driver import generate_unique_timestamp
    unique_ts = generate_unique_timestamp()
    if folder is None:
        parallel_folder = str(Path("trimci_parallel_results") / f"{mol_name}_{unique_ts}")
    else:
        parallel_folder = str(Path(folder) / f"parallel_{mol_name}_{unique_ts}")
    Path(parallel_folder).mkdir(parents=True, exist_ok=True)
    
    # Convert args to dict for pickling
    if hasattr(args, '__dict__'):
        args_dict = vars(args).copy()
    else:
        args_dict = dict(args)
    # Force single run per worker
    args_dict['num_runs'] = 1
    
    # Initialize summary logger (unified multi_summary.log for both parallel and sequential multi-run)
    summary_log_path = os.path.join(parallel_folder, "multi_summary.log")
    logger = TrimCISummaryLogger(summary_log_path, mode="parallel")
    logger.write_header(mol_name, n_orb, n_alpha, n_beta, args_dict, num_runs, n_parallel, omp_per_run)
    
    verbosity = args_dict.get('verbosity', 1 if args_dict.get('verbose', False) else 0)
    
    if verbosity >= 1:
        print(f"🚀 Starting parallel TrimCI: {num_runs} runs, {n_parallel} workers, {omp_per_run} OMP threads each")
    
    # Prepare task list
    tasks = [
        (omp_per_run, i, h1, eri, n_alpha, n_beta, n_orb,
         mol_name, args_dict, nuclear_repulsion, parallel_folder)
        for i in range(num_runs)
    ]
    
    # Execute in parallel using spawn
    ctx = get_context('spawn')
    if verbosity >= 1:
        print(f"📊 Launching {n_parallel} parallel workers...")
    
    start_all = time.perf_counter()
    successful = []
    failed = []
    
    with ctx.Pool(processes=n_parallel) as pool:
        # Use imap_unordered for real-time logging as each run completes
        for r in pool.imap_unordered(_worker_single_run, tasks):
            if r['success']:
                successful.append(r)
                n_iters = r.get('details', {}).get('total_iterations')
                logger.log_run_complete(r['run_idx'], r['final_energy'], r['elapsed'], n_iters)
                if verbosity >= 1:
                    print(f"  ✓ Run {r['run_idx']:03d}: E={r['final_energy']:.8f} ({r['elapsed']:.1f}s)")
            else:
                failed.append(r)
                logger.log_run_failed(r['run_idx'], r.get('error', 'Unknown error'))
                if verbosity >= 1:
                    print(f"  ✗ Run {r['run_idx']:03d}: FAILED - {r.get('error', 'Unknown')}")
    
    total_time = time.perf_counter() - start_all
    
    if failed:
        if verbosity >= 1:
            print(f"⚠️ {len(failed)} runs failed:")
            for r in failed:
                print(f"   Run {r['run_idx']}: {r.get('error', 'Unknown error')}")
    
    if not successful:
        logger.close()
        raise RuntimeError("All parallel runs failed!")
    
    # Find best result
    best = min(successful, key=lambda x: x['final_energy'])
    
    # Write final summary
    logger.write_parallel_summary(successful, failed, total_time, best)
    logger.close()
    
    # Generate Markdown Report (trimci_report.md)
    try:
        from .report_generator import generate_report
        # Convert args_dict back to object-like for compatibility
        class ArgsWrapper:
            pass
        args_obj = ArgsWrapper()
        for k, v in args_dict.items():
            setattr(args_obj, k, v)
        
        generate_report(successful, best, mol_name, args_obj, folder=parallel_folder)
        if verbosity >= 1:
            print(f"📄 Report generated: {parallel_folder}/trimci_report.md")
    except Exception as e:
        if verbosity >= 1:
            print(f"⚠️ Failed to generate Markdown report: {e}")
    
    if verbosity >= 1:
        print(f"✅ Completed in {total_time:.2f}s. Best energy: {best['final_energy']:.8f} (run {best['run_idx']})")
        print(f"📁 Results saved to: {parallel_folder}")
    
    # Return in same format as run_trimci_main_calculation
    return (
        best['final_energy'],
        best['dets'],
        best['coeffs'],
        best['details'],
        best['run_args']
    )
