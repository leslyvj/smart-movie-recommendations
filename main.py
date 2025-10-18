"""
Main Orchestrator for Movie Recommendation System
Runs all modules in sequence
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print application banner"""
    print("\n")
    print("="*80)
    print("           MOVIE RECOMMENDATION SYSTEM")
    print("     User-Based Collaborative Filtering with Pearson Correlation")
    print("="*80)
    print("\n")


def run_module(module_name, description):
    """
    Run a module and handle errors
    
    Args:
        module_name: Name of the module to import
        description: Description of the module
    """
    print("\n")
    print("="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print("\n")
    
    start_time = time.time()
    
    try:
        # Import and run module
        module = __import__(module_name)
        module.main()
        
        elapsed = time.time() - start_time
        print(f"✅ {description} completed in {elapsed:.2f} seconds\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error in {description}:")
        print(f"   {str(e)}\n")
        return False


def main():
    """Main execution function"""
    print_banner()
    
    print("This will run all modules in sequence:")
    print("  1. Data Preparation")
    print("  2. Similarity Calculation")
    print("  3. Rating Prediction")
    print("  4. Generate Recommendations")
    print("\n")
    
    # Ask for confirmation
    response = input("Do you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Execution cancelled.")
        return
    
    start_time = time.time()
    
    # Run all modules in sequence
    modules = [
        ("01_data_preparation", "Part 1-3: Data Preparation"),
        ("02_similarity_calculation", "Part 4: Similarity Calculation"),
        ("03_rating_prediction", "Part 5: Rating Prediction"),
        ("04_generate_recommendations", "Part 6: Generate Recommendations")
    ]
    
    success_count = 0
    
    for module_name, description in modules:
        if run_module(module_name, description):
            success_count += 1
        else:
            print(f"\n❌ Stopping execution due to error in {description}\n")
            break
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n")
    print("="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"\nModules completed: {success_count}/{len(modules)}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    if success_count == len(modules):
        print("\n✅ ALL MODULES COMPLETED SUCCESSFULLY!")
        print("\n📁 Output files are available in: ./output/")
        print("\n📸 Screenshots for PDF report:")
        print("  1. User-User Similarity Matrix (from Part 4)")
        print("  2. Sample Predicted Ratings (from Part 5)")
        print("  3. Final Top Recommendations (from Part 6)")
    else:
        print("\n⚠ Some modules failed. Please check the errors above.")
    
    print("\n" + "="*80)
    print("\n")


if __name__ == "__main__":
    main()