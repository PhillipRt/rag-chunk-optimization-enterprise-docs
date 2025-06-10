#!/usr/bin/env python
"""
Test runner for the RAG evaluation framework.
"""
import os
import sys
import subprocess
import argparse


def run_tests(specific_test=None, verbose=False, coverage=False):
    """
    Run pytest with the specified options.

    Args:
        specific_test: Optional specific test file or directory to run
        verbose: Whether to run tests in verbose mode
        coverage: Whether to generate a coverage report
    """
    # Determine the command to run
    uv_path = os.path.expanduser("~/.local/bin/uv.exe") if os.name == 'nt' else "uv"

    # Check if uv exists at the expected path for Windows
    if os.name == 'nt' and not os.path.exists(uv_path):
        # Try to find uv in PATH
        try:
            uv_path_result = subprocess.run(["where", "uv"], capture_output=True, text=True, check=True)
            uv_paths = uv_path_result.stdout.strip().split('\n')
            if uv_paths:
                uv_path = uv_paths[0]
        except subprocess.CalledProcessError:
            print("Warning: Could not find uv in PATH. Falling back to pip.")
            # Install with pip instead
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], check=True)
            return

    # Install pytest and pytest-cov using uv
    print(f"Installing test dependencies using {uv_path}...")
    try:
        subprocess.run([uv_path, "pip", "install", "pytest", "pytest-cov"], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Failed to install dependencies with uv. Falling back to pip.")
        # Fall back to pip if uv fails
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"], check=True)

    # Build the pytest command
    pytest_command = ["python", "-m", "pytest"]

    # Add options
    if verbose:
        pytest_command.append("-v")

    if coverage:
        pytest_command.extend(["--cov=.", "--cov-report=term", "--cov-report=html"])

    # Add the specific test if provided
    if specific_test:
        if specific_test.startswith("master-thesis-rag/"):
            # Remove the prefix if it exists
            specific_test = specific_test[len("master-thesis-rag/"):]
        pytest_command.append(specific_test)
    else:
        pytest_command.append("tests")

    # Run pytest
    print(f"Running tests with command: {' '.join(pytest_command)}")
    result = subprocess.run(pytest_command)

    # Return the exit code
    return result.returncode


def main():
    """Parse arguments and run the tests."""
    parser = argparse.ArgumentParser(description="Run tests for the RAG evaluation framework")
    parser.add_argument("--test", help="Specific test file or directory to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Run tests in verbose mode")
    parser.add_argument("--coverage", action="store_true", help="Generate a coverage report")

    args = parser.parse_args()

    exit_code = run_tests(args.test, args.verbose, args.coverage)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()