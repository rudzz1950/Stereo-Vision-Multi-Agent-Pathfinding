"""
Vecros Assignment - Main Entry Point
=====================================

This script provides a unified interface to run both problems:
- Problem 1: Stereo Vision Feature Detection, Matching & Camera Pose Estimation
- Problem 2: 3D Multi-Agent Pathfinding with Collision Avoidance

Usage:
    python main.py --problem 1 --dataset kitti --sequence 00
    python main.py --problem 2 --start1 0,0,0 --end1 50,50,50 --start2 100,100,100 --end2 50,50,50

Author: Vecros Assignment Submission
"""

import argparse
import numpy as np
from pathlib import Path
import sys


def run_problem1_kitti(data_path: str, sequence: str = "00", output_dir: str = "results/problem1"):
    """
    Run Problem 1 with KITTI odometry dataset.
    
    Args:
        data_path: Path to KITTI odometry dataset
        sequence: Sequence number (e.g., "00")
        output_dir: Output directory for results
    """
    from problem1_stereo_vision import run_problem1
    
    try:
        import pykitti
    except ImportError:
        print("ERROR: pykitti not installed. Run: pip install pykitti")
        return None
    
    print(f"\nLoading KITTI Odometry Sequence {sequence}...")
    
    # Load KITTI data
    data = pykitti.odometry(data_path, sequence)
    
    # Get camera intrinsics from calibration
    # P0 is the projection matrix for left grayscale camera
    P = data.calib.P_rect_00
    K = P[:3, :3]
    
    print(f"Camera intrinsic matrix K:\n{K}")
    print(f"Number of frames: {len(data.cam0_files)}")
    
    # Use first two frames for demonstration
    img1_path = data.cam0_files[0]
    img2_path = data.cam0_files[1]
    
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    
    # Get ground truth poses
    poses = data.poses
    if len(poses) >= 2:
        T1 = poses[0]  # 4x4 transformation matrix
        T2 = poses[1]
        
        # Relative transformation from frame 1 to frame 2
        T_rel = np.linalg.inv(T1) @ T2
        R_gt = T_rel[:3, :3]
        t_gt = T_rel[:3, 3:]
        
        print(f"\nGround truth relative pose:")
        print(f"Rotation:\n{R_gt}")
        print(f"Translation: {t_gt.T}")
    else:
        print("\nWARNING: Ground truth poses not available")
        R_gt = None
        t_gt = None
    
    # Run the pipeline
    results = run_problem1(
        img1_path=img1_path,
        img2_path=img2_path,
        K=K,
        R_gt=R_gt,
        t_gt=t_gt,
        output_dir=output_dir
    )
    
    return results


def run_problem1_middlebury(data_path: str, output_dir: str = "results/problem1"):
    """
    Run Problem 1 with Middlebury stereo dataset.
    
    Args:
        data_path: Path to Middlebury scene folder
        output_dir: Output directory for results
    """
    from problem1_stereo_vision import run_problem1
    
    data_path = Path(data_path)
    
    # Find stereo images
    img1_path = data_path / "im0.png"
    img2_path = data_path / "im1.png"
    
    if not img1_path.exists():
        # Try alternative names
        for ext in ['png', 'jpg', 'pgm']:
            img1_path = list(data_path.glob(f"*left*.{ext}")) or list(data_path.glob(f"*0.{ext}"))
            img2_path = list(data_path.glob(f"*right*.{ext}")) or list(data_path.glob(f"*1.{ext}"))
            if img1_path and img2_path:
                img1_path = img1_path[0]
                img2_path = img2_path[0]
                break
    
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    
    # Load calibration if available
    calib_path = data_path / "calib.txt"
    if calib_path.exists():
        # Parse Middlebury calibration format
        with open(calib_path, 'r') as f:
            lines = f.readlines()
        # This is simplified - actual parsing depends on format
        # Default focal length for Middlebury (approximate)
        K = np.array([
            [3000, 0, 1500],
            [0, 3000, 1000],
            [0, 0, 1]
        ], dtype=float)
    else:
        # Default camera matrix (approximate for Middlebury)
        K = np.array([
            [3000, 0, 1500],
            [0, 3000, 1000],
            [0, 0, 1]
        ], dtype=float)
    
    print(f"\nUsing camera matrix K:\n{K}")
    print("(Note: For Middlebury, ground truth pose is not directly available)")
    
    # Run pipeline without ground truth
    results = run_problem1(
        img1_path=str(img1_path),
        img2_path=str(img2_path),
        K=K,
        R_gt=None,
        t_gt=None,
        output_dir=output_dir
    )
    
    return results


def run_problem1_custom(img1_path: str, img2_path: str, fx: float, fy: float, 
                        cx: float, cy: float, output_dir: str = "results/problem1"):
    """
    Run Problem 1 with custom images and camera parameters.
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        fx, fy: Focal lengths
        cx, cy: Principal point
        output_dir: Output directory for results
    """
    from problem1_stereo_vision import run_problem1
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    results = run_problem1(
        img1_path=img1_path,
        img2_path=img2_path,
        K=K,
        R_gt=None,
        t_gt=None,
        output_dir=output_dir
    )
    
    return results


def run_problem2_unified(path_requests: list, grid_size: int = 101, velocity: float = 1.0,
                           weight_ratio: float = 0.1, seed: int = 42,
                           solver: str = 'fast', output_dir: str = "results/problem2"):
    """
    Run Problem 2 using the consolidated solver.
    
    Args:
        path_requests: List of (start, goal) tuples
        grid_size: Size of grid
        velocity: Velocity
        weight_ratio: Weight ratio
        seed: Random seed
        solver: 'fast' or 'cbs'
        output_dir: Output directory
    """
    try:
        from problem2 import run_problem2
        
        results = run_problem2(
            path_requests=path_requests,
            grid_size=grid_size,
            velocity=velocity,
            weight_ratio=weight_ratio,
            seed=seed,
            solver=solver,
            output_dir=output_dir
        )
        return results
    except ImportError as e:
        print(f"ERROR: Could not import problem2.py: {e}")
        sys.exit(1)


def parse_point(s: str) -> tuple:
    """Parse a comma-separated point string like '0,0,0' to tuple."""
    parts = s.strip().split(',')
    if len(parts) != 3:
        raise ValueError(f"Expected 3 coordinates, got {len(parts)}")
    return tuple(int(p.strip()) for p in parts)


def parse_path(s: str) -> tuple:
    """Parse a path string like '0,0,0:50,50,50' to (start, goal) tuple."""
    parts = s.split(':')
    if len(parts) != 2:
        raise ValueError(f"Expected format 'start:end', got '{s}'")
    return (parse_point(parts[0]), parse_point(parts[1]))


def main():
    parser = argparse.ArgumentParser(
        description="Vecros Assignment - Stereo Vision & 3D Pathfinding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Problem 1 with KITTI dataset
  python main.py --problem 1 --dataset kitti --data-path ./data/kitti --sequence 00

  # Run Problem 1 with Middlebury dataset
  python main.py --problem 1 --dataset middlebury --data-path ./data/Adirondack

  # Run Problem 1 with custom images
  python main.py --problem 1 --dataset custom --img1 left.png --img2 right.png --fx 700 --fy 700 --cx 320 --cy 240

  # Run Problem 2 with 2 agents (legacy format)
  python main.py --problem 2 --start1 0,0,0 --end1 50,50,50 --start2 100,100,100 --end2 50,50,50

  # Run Problem 2 with 3+ agents using --paths (NEW!)
  python main.py --problem 2 --paths "0,0,0:20,20,20" "40,40,40:20,20,20" "0,40,0:20,20,20"

  # Run Problem 2 with CBS optimal solver
  python main.py --problem 2 --solver cbs --paths "0,0,0:15,15,15" "30,30,30:15,15,20"
        """
    )
    
    # General arguments
    parser.add_argument('--problem', type=int, choices=[1, 2], required=True,
                        help='Problem number (1 or 2)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    
    # Problem 1 arguments
    parser.add_argument('--dataset', type=str, choices=['kitti', 'middlebury', 'custom'],
                        help='Dataset type for Problem 1')
    parser.add_argument('--data-path', type=str,
                        help='Path to dataset folder')
    parser.add_argument('--sequence', type=str, default='00',
                        help='KITTI sequence number')
    parser.add_argument('--img1', type=str,
                        help='Path to first image (custom dataset)')
    parser.add_argument('--img2', type=str,
                        help='Path to second image (custom dataset)')
    parser.add_argument('--fx', type=float, default=700,
                        help='Focal length x (custom dataset)')
    parser.add_argument('--fy', type=float, default=700,
                        help='Focal length y (custom dataset)')
    parser.add_argument('--cx', type=float, default=320,
                        help='Principal point x (custom dataset)')
    parser.add_argument('--cy', type=float, default=240,
                        help='Principal point y (custom dataset)')
    
    # Problem 2 arguments
    parser.add_argument('--paths', type=str, nargs='+',
                        help='Paths in format "start:end" e.g. "0,0,0:50,50,50" (supports 3+ agents)')
    parser.add_argument('--start1', type=str, default='0,0,0',
                        help='Start point for agent 1 (x,y,z)')
    parser.add_argument('--end1', type=str, default='25,25,25',
                        help='End point for agent 1 (x,y,z)')
    parser.add_argument('--start2', type=str, default='50,50,50',
                        help='Start point for agent 2 (x,y,z)')
    parser.add_argument('--end2', type=str, default='25,25,25',
                        help='End point for agent 2 (x,y,z)')
    parser.add_argument('--start3', type=str, default=None,
                        help='Start point for agent 3 (x,y,z)')
    parser.add_argument('--end3', type=str, default=None,
                        help='End point for agent 3 (x,y,z)')
    parser.add_argument('--grid-size', type=int, default=101,
                        help='Grid size (0 to grid-size-1)')
    parser.add_argument('--velocity', type=float, default=1.0,
                        help='Travel velocity (m/s)')
    parser.add_argument('--weight-ratio', type=float, default=0.1,
                        help='Ratio of weighted grid points')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--solver', type=str, choices=['optimized', 'cbs', 'compare'], default='optimized',
                        help='Solver: "optimized" (fast), "cbs" (optimal), or "compare" (run both)')
    
    args = parser.parse_args()
    
    if args.problem == 1:
        # Run Problem 1
        if not args.dataset:
            print("ERROR: --dataset is required for Problem 1")
            print("Options: kitti, middlebury, custom")
            sys.exit(1)
        
        output_dir = f"{args.output_dir}/problem1"
        
        if args.dataset == 'kitti':
            if not args.data_path:
                print("ERROR: --data-path is required for KITTI dataset")
                sys.exit(1)
            run_problem1_kitti(args.data_path, args.sequence, output_dir)
            
        elif args.dataset == 'middlebury':
            if not args.data_path:
                print("ERROR: --data-path is required for Middlebury dataset")
                sys.exit(1)
            run_problem1_middlebury(args.data_path, output_dir)
            
        elif args.dataset == 'custom':
            if not args.img1 or not args.img2:
                print("ERROR: --img1 and --img2 are required for custom dataset")
                sys.exit(1)
            run_problem1_custom(args.img1, args.img2, args.fx, args.fy, 
                               args.cx, args.cy, output_dir)
    
    elif args.problem == 2:
        output_dir = f"{args.output_dir}/problem2"
        
        # Check if using new --paths format
        if args.paths:
            try:
                path_requests = [parse_path(p) for p in args.paths]
                print(f"Planning {len(path_requests)} agent paths...")
            except ValueError as e:
                print(f"ERROR: Invalid path format: {e}")
                print("Use format: 'start_x,start_y,start_z:end_x,end_y,end_z'")
                sys.exit(1)
        else:
            # Check if any legacy start/end args were provided
            if args.start1 != '0,0,0' or args.end1 != '25,25,25':
                # User provided specific points via legacy args
                try:
                    path_requests = [
                        (parse_point(args.start1), parse_point(args.end1)),
                        (parse_point(args.start2), parse_point(args.end2))
                    ]
                    if args.start3 and args.end3:
                        path_requests.append((parse_point(args.start3), parse_point(args.end3)))
                except ValueError as e:
                    print(f"ERROR: Invalid point format: {e}")
                    sys.exit(1)
            else:
                # No paths provided -> Use 4-Agent Default
                path_requests = [
                    ((0, 0, 0), (15, 15, 15)),
                    ((30, 30, 30), (15, 15, 20)),
                    ((0, 30, 0), (20, 15, 15)),
                    ((30, 0, 30), (15, 20, 15)),
                ]
        
        # Map 'optimized' to 'fast' for problem2.py
        solver_map = {'optimized': 'fast', 'cbs': 'cbs', 'compare': 'compare'}
        solver_type = solver_map.get(args.solver, 'fast')
        
        run_problem2_unified(
            path_requests=path_requests,
            grid_size=args.grid_size,
            velocity=args.velocity,
            weight_ratio=args.weight_ratio,
            seed=args.seed,
            solver=solver_type,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
