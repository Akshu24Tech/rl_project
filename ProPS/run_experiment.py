"""
Master script to guide through the entire ProPS experiment
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✓ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        if e.stderr:
            print(e.stderr)
        return False

def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} detected")
        print("  Python 3.8 or higher required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def main():
    print_header("ProPS Gemini 3.0 Comparison Experiment")
    
    print("This script will guide you through:")
    print("  1. Setup verification")
    print("  2. Quick test (2-5 minutes)")
    print("  3. Full comparison (15-30 minutes)")
    print("  4. Results analysis")
    
    # Check Python version
    print("\n" + "-"*70)
    print("Step 0: Checking Python version")
    print("-"*70)
    if not check_python_version():
        return
    
    # Step 1: Setup
    print("\n" + "-"*70)
    print("Step 1: Setup Verification")
    print("-"*70)
    
    choice = input("\nRun setup test? (y/n, default=y): ").strip().lower()
    if choice != 'n':
        if not run_command("python test_setup.py", "Testing setup"):
            print("\n⚠️  Setup test failed. Please install dependencies:")
            print("   pip install -r requirements_props.txt")
            retry = input("\nTry again after installing? (y/n): ").strip().lower()
            if retry == 'y':
                if not run_command("python test_setup.py", "Re-testing setup"):
                    print("\n❌ Setup still failing. Please check error messages above.")
                    return
            else:
                return
    
    # Step 2: Quick test
    print("\n" + "-"*70)
    print("Step 2: Quick Test (Optional)")
    print("-"*70)
    print("\nThe quick test runs 10 episodes with one model (~2-5 minutes)")
    print("This verifies everything works before the full experiment.")
    
    choice = input("\nRun quick test? (y/n, default=y): ").strip().lower()
    if choice != 'n':
        print("\n📝 You'll need your Gemini API key for this step")
        print("   Get it from: https://aistudio.google.com/")
        input("\nPress Enter when ready...")
        
        if not run_command("python quick_test.py", "Running quick test"):
            print("\n⚠️  Quick test failed. Common issues:")
            print("   - Invalid API key")
            print("   - Network connection")
            print("   - Rate limits")
            
            cont = input("\nContinue to full experiment anyway? (y/n): ").strip().lower()
            if cont != 'y':
                return
    
    # Step 3: Full comparison
    print("\n" + "-"*70)
    print("Step 3: Full Comparison Experiment")
    print("-"*70)
    print("\nThis will test 3 Gemini models with 50 episodes each")
    print("Estimated time: 15-30 minutes")
    print("You can customize settings in props_gemini_comparison.py")
    
    choice = input("\nRun full comparison? (y/n, default=y): ").strip().lower()
    if choice != 'n':
        print("\n🚀 Starting full experiment...")
        print("   This may take a while. You can monitor progress in the output.")
        
        if not run_command("python props_gemini_comparison.py", "Running full comparison"):
            print("\n⚠️  Experiment failed. Check error messages above.")
            return
        
        print("\n✅ Experiment completed!")
    
    # Step 4: Analysis
    print("\n" + "-"*70)
    print("Step 4: Results Analysis")
    print("-"*70)
    
    # Check if results exist
    results_dir = "PrPoS/results"
    if os.path.exists(results_dir) and os.listdir(results_dir):
        print("\nResults found! Running analysis...")
        
        if run_command("python analyze_results.py", "Analyzing results"):
            print("\n✅ Analysis complete!")
            print("\nGenerated files:")
            print("  - Comparison plots (PNG)")
            print("  - Statistical analysis")
            print("  - LaTeX table")
        else:
            print("\n⚠️  Analysis failed, but your results are saved in:")
            print(f"   {results_dir}")
    else:
        print("\nNo results found to analyze.")
        print("Results will be saved in: PrPoS/results/")
    
    # Final summary
    print_header("Experiment Complete!")
    
    print("What you can do next:")
    print("  1. Check plots in PrPoS/results/")
    print("  2. Review JSON data files")
    print("  3. Run more experiments for better statistics")
    print("  4. Try different configurations")
    print("  5. Test ProPS+ (with environment descriptions)")
    
    print("\nKey files:")
    print("  - props_gemini_comparison.py : Main experiment")
    print("  - quick_test.py              : Fast test")
    print("  - analyze_results.py         : Analysis tool")
    print("  - GETTING_STARTED.md         : Detailed guide")
    
    print("\n📊 To re-analyze results anytime:")
    print("   python analyze_results.py")
    
    print("\n🎉 Thank you for testing ProPS with Gemini 3.0!")
    print("   If you find interesting results, consider sharing them!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("You can resume anytime by running: python run_experiment.py")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("Please check the error message and try again")
