import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_cassava_data(data_dir, test_size=0.2, random_state=42):
    """
    Splits the Cassava train.csv into train and validation sets.
    """
    csv_path = os.path.join(data_dir, 'train.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please download the dataset.")
        return

    df = pd.read_csv(csv_path)
    
    # Stratified split to maintain class balance
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['label'], 
        random_state=random_state
    )

    train_df.to_csv(os.path.join(data_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val_split.csv'), index=False)
    
    print(f"Data split completed.")
    print(f"Train set: {len(train_df)} images")
    print(f"Val set: {len(val_df)} images")

if __name__ == "__main__":
    DATA_DIR = "data/cassava"
    # Ensure directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    split_cassava_data(DATA_DIR)
