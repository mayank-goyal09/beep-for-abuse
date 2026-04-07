from datasets import load_dataset
import pandas as pd
import os

def fetch_toxic_data():
    print("📥 Downloading Dataset: 'dffesalbon/dota-2-toxic-chat-data'...")
    try:
        dataset1 = load_dataset("dffesalbon/dota-2-toxic-chat-data")
        df_dota = dataset1['train'].to_pandas()
        
        print("📥 Downloading Clean Dataset: 'sst2' (Greetings/Daily words)...")
        dataset2 = load_dataset("sst2")
        df_sst2 = dataset2['train'].to_pandas()
        
        # Map SST2's 'sentence' column to 'message'
        df_sst2.rename(columns={'sentence': 'message'}, inplace=True)
        # Labelling ALL of sst2 as 0 (clean) as per the instruction
        df_sst2['target'] = 0
        
        # Ensure only the two critical columns are kept before merging
        df_sst2 = df_sst2[['message', 'target']]
        df_dota = df_dota[['message', 'target']]
        
        # Combine both datasets
        df = pd.concat([df_dota, df_sst2], ignore_index=True)
        
        # Shuffle the mixed dataset so it learns cleanly
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save the dataset to the assets/samples folder
        output_dir = os.path.join("assets", "samples")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "toxic_chat_data.csv")
        df.to_csv(output_file, index=False)
        
        print(f"✅ Successfully combined and saved {len(df)} rows to {output_file}!")
        return df
    except Exception as e:
        print(f"❌ Error fetching dataset: {e}")

if __name__ == "__main__":
    fetch_toxic_data()
