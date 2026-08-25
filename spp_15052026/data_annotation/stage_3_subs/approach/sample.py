import pandas as pd

df = pd.read_parquet("/home/ethel/Desktop/swimming_pool_project/spp_15052026/data/racialised_reclassed_reviews.parquet")
sample = df.sample(n=50, random_state=42)
sample.to_parquet("sample_50.parquet")
print(sample.shape)
print(sample.head())

#merge processed successful  classification
successfull_classification = pd.read_csv("processed_success.csv")
merge = pd.merge(df, successfull_classification, on="review_id", how="left")
merge.to_parquet("/home/ethel/Desktop/swimming_pool_project/spp_15052026/data/reclassed_racialised_approach.parquet")

