from src.data.make_dataset import load_data, clean_data, save_data
from src.models.train_model import train

def run_pipeline():
    print("🚀 Pipeline Starting...\n")
    
    print("=" * 40)
    print("📦 STEP 1: Data Cleaning")
    print("=" * 40)
    df = load_data()
    df = clean_data(df)
    save_data(df)
    
    print("\n" + "=" * 40)
    print("🤖 STEP 2: Training")
    print("=" * 40)
    train()
    
    print("\n✅ Pipeline Complete!")

if __name__ == "__main__":
    run_pipeline()