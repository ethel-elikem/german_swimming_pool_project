import pandas as pd
import random  # Import the random module

random.seed(1234)  # Now this works correctly

input_df = pd.read_parquet(
    "/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/collected_reviews_25112025.parquet"
)

# Use random_state for consistent sampling in pandas
output_df = input_df.sample(20, random_state=1234).reset_index(drop=True)

output_df.to_csv('/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/test_df.csv', index=False)