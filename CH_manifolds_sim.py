"""
Cahn-Hilliard simulation experiment with manifold transformations.
"""

import torch, numpy as np, os, datetime, argparse
from src_CH import cahn_hilliard_simulation
from src_toy_manifolds import (
    split,
    hole,
    pinch,
    collapse,
    swiss_roll,
    dog_ear,
    ribbon,
    cylinder,
)

# Default configuration
DEFAULT_CONFIG = {
    "device": None,
    "output_dir": "ch_manifold_results",
    "grid_size": 128,
    "nsteps": 1000,
    "noise_amplitude": 0.01,
    "seed": 42,
    "n_samples": 4096,
    "c_init_range": (0.3, 0.7),
    "alpha_range": (0.7, 1.3),
    "beta_range": (0.7, 1.3),
    "W_base": 1.0,
    "kappa_base": 1.5,
    "M_base": 1.0,
    "dt_base": 0.5,
}

# Transformations with metadata
TRANSFORMATIONS = [
    {"name": "Identity", "func": lambda p: p, "is_homeomorphic": True, "is_3d": False},
    {"name": "Split", "func": split, "is_homeomorphic": False, "is_3d": False},
    {"name": "Hole", "func": hole, "is_homeomorphic": False, "is_3d": False},
    {"name": "Pinch", "func": pinch, "is_homeomorphic": False, "is_3d": False},
    {"name": "Collapse", "func": collapse, "is_homeomorphic": False, "is_3d": False},
    {"name": "Swiss_Roll", "func": swiss_roll, "is_homeomorphic": True, "is_3d": True},
    {"name": "Dog_Ear", "func": dog_ear, "is_homeomorphic": False, "is_3d": True},
    {"name": "Ribbon", "func": ribbon, "is_homeomorphic": False, "is_3d": True},
    {"name": "Cylinder", "func": cylinder, "is_homeomorphic": False, "is_3d": True},
]


def get_device(device_str=None):
    """Get appropriate torch device (with support for CUDA, XPU, MPS)"""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if (
        hasattr(torch, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.backends, "mps")
    ):
        return torch.device("mps")
    return torch.device("cpu")


def normalize_to_unit_cube(points, c_init_range, alpha_range, beta_range):
    """Convert points from parameter ranges to [-1, 1] cube"""
    param_ranges = torch.tensor(
        [
            [c_init_range[0], c_init_range[1]],
            [alpha_range[0], alpha_range[1]],
            [beta_range[0], beta_range[1]],
        ],
        device=points.device,
    )
    ranges = param_ranges[:, 1] - param_ranges[:, 0]
    midpoints = (param_ranges[:, 1] + param_ranges[:, 0]) / 2
    return 2 * (points - midpoints) / ranges


def denormalize_from_unit_cube(norm_points, c_init_range, alpha_range, beta_range):
    """Convert points from [-1, 1] cube back to parameter ranges"""
    param_ranges = torch.tensor(
        [
            [c_init_range[0], c_init_range[1]],
            [alpha_range[0], alpha_range[1]],
            [beta_range[0], beta_range[1]],
        ],
        device=norm_points.device,
    )
    ranges = param_ranges[:, 1] - param_ranges[:, 0]
    midpoints = (param_ranges[:, 1] + param_ranges[:, 0]) / 2
    return (norm_points * ranges / 2) + midpoints


def generate_uniform_samples(n_samples, c_init_range, alpha_range, seed, device):
    """Generate uniform parameter samples"""
    torch.manual_seed(seed)
    c_init = torch.FloatTensor(n_samples).uniform_(*c_init_range).to(device)
    alpha = torch.FloatTensor(n_samples).uniform_(*alpha_range).to(device)
    beta = torch.ones(n_samples, device=device)  # Fix beta at 1.0
    return torch.stack([c_init, alpha, beta], dim=1)


def apply_transformations_to_samples(samples, c_init_range, alpha_range, beta_range):
    """Apply all transformations to parameter samples"""
    normalized = normalize_to_unit_cube(
        samples,
        c_init_range=c_init_range,
        alpha_range=alpha_range,
        beta_range=beta_range,
    )
    return {
        t["name"]: denormalize_from_unit_cube(
            t["func"](normalized), c_init_range, alpha_range, beta_range
        )
        for t in TRANSFORMATIONS
    }


def run_simulations(transformed_params, config):
    """Run Cahn-Hilliard simulations for transformed parameters"""
    device, grid_size = config["device"], config["grid_size"]

    # Generate common noise pattern
    torch.manual_seed(config["seed"])
    noise_pattern = torch.randn(
        grid_size, grid_size, dtype=torch.float32, device=device
    )

    results = {}
    for name, params in transformed_params.items():
        print(f"\nRunning simulations for {name} transformation:")
        n_samples = params.shape[0]
        c_init, alpha, beta = params[:, 0], params[:, 1], params[:, 2]

        # Set parameters
        W = torch.full((n_samples,), config["W_base"], device=device)
        dt = torch.full((n_samples,), config["dt_base"], device=device)
        L = torch.full((n_samples,), float(grid_size), device=device)
        kappa = config["kappa_base"] * (alpha**2) * (beta**2)
        M = config["M_base"] * (alpha**2) / (beta**2)

        # Create initial conditions with the same noise pattern
        c0 = torch.zeros(
            (n_samples, grid_size, grid_size), dtype=torch.float32, device=device
        )
        for i in range(n_samples):
            c0[i] = torch.full((grid_size, grid_size), c_init[i], device=device)
        c0 += config["noise_amplitude"] * noise_pattern

        # Run simulation
        results[name] = cahn_hilliard_simulation(
            c0=c0,
            W=W,
            kappa=kappa,
            L=L,
            Nsteps=config["nsteps"],
            dt=dt,
            M=M,
            progress_bar=True,
        )

    return results


def save_data(original_params, transformed_params, simulation_results, config):
    """Save all data for further analysis"""
    os.makedirs(config["output_dir"], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Convert tensors to numpy
    original_params_np = original_params.cpu().numpy()
    transformed_params_np = {k: v.cpu().numpy() for k, v in transformed_params.items()}
    transformed_params_norm_np = {
        k: normalize_to_unit_cube(
            v, config["c_init_range"], config["alpha_range"], config["beta_range"]
        )
        .cpu()
        .numpy()
        for k, v in transformed_params.items()
    }
    simulation_results_np = {k: v.cpu().numpy() for k, v in simulation_results.items()}

    # Prepare save dictionary
    save_dict = {
        "timestamp": timestamp,
        "original_params": original_params_np,
        "transformed_params": transformed_params_np,
        "transformed_params_norm": transformed_params_norm_np,
        "simulation_results": simulation_results_np,
        "config": config,
        "transformations": [
            {
                "name": t["name"],
                "is_homeomorphic": t["is_homeomorphic"],
                "is_3d": t["is_3d"],
            }
            for t in TRANSFORMATIONS
        ],
    }

    # Save data
    filename = f"{config['output_dir']}/{timestamp}"
    np.save(filename, save_dict)
    print(f"Data saved to: {filename}.npy")
    return filename


def parse_range(range_str):
    """Parse a string range in the format 'min,max' to a tuple of floats"""
    try:
        min_val, max_val = map(float, range_str.split(","))
        return (min_val, max_val)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Range must be in format 'min,max', got: {range_str}"
        )


def parse_args():
    """Parse command line arguments with detailed help"""
    parser = argparse.ArgumentParser(
        description="Run Cahn-Hilliard simulation with parameter transformations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example usage:\n"
            "  python src_ch_experiment.py --n_samples 500 --grid_size 64 --output_dir results\n"
            "  python src_ch_experiment.py --c_init_range 0.4,0.6 --alpha_range 0.8,1.2 --device cuda\n"
        ),
    )

    # Simulation parameters
    sim_group = parser.add_argument_group("Simulation Parameters")
    sim_group.add_argument(
        "--grid_size",
        type=int,
        default=DEFAULT_CONFIG["grid_size"],
        help="Size of the simulation grid (NxN)",
    )
    sim_group.add_argument(
        "--nsteps",
        type=int,
        default=DEFAULT_CONFIG["nsteps"],
        help="Number of simulation steps to run",
    )
    sim_group.add_argument(
        "--noise_amplitude",
        type=float,
        default=DEFAULT_CONFIG["noise_amplitude"],
        help="Amplitude of noise added to initial conditions",
    )

    # Physics parameters
    phys_group = parser.add_argument_group("Physics Parameters")
    phys_group.add_argument(
        "--W_base",
        type=float,
        default=DEFAULT_CONFIG["W_base"],
        help="Bulk free energy coefficient",
    )
    phys_group.add_argument(
        "--kappa_base",
        type=float,
        default=DEFAULT_CONFIG["kappa_base"],
        help="Base gradient energy coefficient",
    )
    phys_group.add_argument(
        "--M_base",
        type=float,
        default=DEFAULT_CONFIG["M_base"],
        help="Base mobility parameter",
    )
    phys_group.add_argument(
        "--dt_base",
        type=float,
        default=DEFAULT_CONFIG["dt_base"],
        help="Base time step size",
    )

    # Parameter ranges
    range_group = parser.add_argument_group("Parameter Ranges")
    range_group.add_argument(
        "--c_init_range",
        type=parse_range,
        default=DEFAULT_CONFIG["c_init_range"],
        help="Range for initial concentration parameter (format: min,max)",
    )
    range_group.add_argument(
        "--alpha_range",
        type=parse_range,
        default=DEFAULT_CONFIG["alpha_range"],
        help="Range for alpha parameter (format: min,max)",
    )
    range_group.add_argument(
        "--beta_range",
        type=parse_range,
        default=DEFAULT_CONFIG["beta_range"],
        help="Range for beta parameter (format: min,max)",
    )

    # Experiment settings
    exp_group = parser.add_argument_group("Experiment Settings")
    exp_group.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_CONFIG["n_samples"],
        help="Number of parameter samples to generate",
    )
    exp_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed for reproducibility",
    )
    exp_group.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help="Directory to save results",
    )
    exp_group.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG["device"],
        help="Compute device (cuda, xpu, mps, cpu, or auto)",
    )
    exp_group.add_argument(
        "--selected_transforms",
        type=str,
        default=None,
        help=(
            "Comma-separated list of transformations to run. "
            "Options: " + ", ".join([t["name"] for t in TRANSFORMATIONS])
        ),
    )

    # Utility arguments
    util_group = parser.add_argument_group("Utility Options")
    util_group.add_argument(
        "--list_transforms",
        action="store_true",
        help="List available transformations and exit",
    )
    util_group.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and exit without running simulations",
    )

    return parser.parse_args()


def main():
    """Main function to run the experiment"""
    # Setup configuration
    args = parse_args()

    # Handle special utility arguments
    if args.list_transforms:
        print("Available transformations:")
        for i, t in enumerate(TRANSFORMATIONS, 1):
            homeomorphic = (
                "Homeomorphic" if t["is_homeomorphic"] else "Non-homeomorphic"
            )
            dimension = "3D" if t["is_3d"] else "2D"
            print(f"{i}. {t['name']} ({dimension}, {homeomorphic})")
        return

    # Set up configuration
    config = DEFAULT_CONFIG.copy()
    for key, value in vars(args).items():
        if value is not None:  # Only update if argument was provided
            config[key] = value

    # Set device
    config["device"] = get_device(config["device"])
    print(f"Using device: {config['device']}")

    # Ensure output directory exists
    os.makedirs(config["output_dir"], exist_ok=True)

    # Process selected transformations if specified
    selected_transforms = None
    if args.selected_transforms:
        transform_names = [t.strip() for t in args.selected_transforms.split(",")]
        available_names = [t["name"] for t in TRANSFORMATIONS]
        invalid_names = [n for n in transform_names if n not in available_names]

        if invalid_names:
            print(f"Warning: Unknown transformations: {', '.join(invalid_names)}")
            print(f"Available transformations: {', '.join(available_names)}")
            transform_names = [n for n in transform_names if n in available_names]

        if not transform_names:
            print(
                "No valid transformations specified. Using all available transformations."
            )
        else:
            selected_transforms = [n for n in transform_names]
            print(f"Using selected transformations: {', '.join(selected_transforms)}")

    # Print configuration summary
    print("\nConfiguration:")
    print(f"  Grid size: {config['grid_size']}x{config['grid_size']}")
    print(f"  Simulation steps: {config['nsteps']}")
    print(f"  Number of samples: {config['n_samples']}")
    print(f"  Parameter ranges:")
    print(f"    - Initial concentration: {config['c_init_range']}")
    print(f"    - Alpha: {config['alpha_range']}")
    print(f"    - Beta: {config['beta_range']}")
    print(f"  Physics parameters:")
    print(f"    - W: {config['W_base']}")
    print(f"    - Kappa base: {config['kappa_base']}")
    print(f"    - M base: {config['M_base']}")
    print(f"    - dt: {config['dt_base']}")
    print(f"  Output directory: {config['output_dir']}")

    # Exit if dry run
    if args.dry_run:
        print("\nDry run completed. Exiting without running simulations.")
        return

    print(f"\nGenerating {config['n_samples']} samples and running simulations...")

    # Generate samples
    original_params = generate_uniform_samples(
        config["n_samples"],
        config["c_init_range"],
        config["alpha_range"],
        config["seed"],
        config["device"],
    )

    # Apply transformations
    transformed_params = apply_transformations_to_samples(
        original_params,
        config["c_init_range"],
        config["alpha_range"],
        config["beta_range"],
    )

    # Filter transformations if needed
    if selected_transforms:
        transformed_params = {
            k: v for k, v in transformed_params.items() if k in selected_transforms
        }

    # Run simulations
    simulation_results = run_simulations(transformed_params, config)

    # Save results
    output_file = save_data(
        original_params, transformed_params, simulation_results, config
    )

    print(f"Done! Data saved to {output_file}.npy")
    print("Run the visualization script to analyze results.")


if __name__ == "__main__":
    main()
