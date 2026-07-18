import pandas as pd
from bert_score import score

# Load evaluation results
df = pd.read_csv("evaluation/evaluation_results.csv")

ground_truth = df["ground_truth"].astype(str).tolist()
generated = df["generated_answer"].astype(str).tolist()

print("Calculating BERTScore...\n")

P, R, F1 = score(
    generated,
    ground_truth,
    lang="en",
    verbose=True
)

df["BERT_F1"] = F1.tolist()

threshold = 0.85

correct = (df["BERT_F1"] >= threshold).sum()

accuracy = (correct / len(df)) * 100

print(f"\nAccuracy: {accuracy:.2f}%")

df.to_csv(
    "evaluation/accuracy_results.csv",
    index=False
)

print("\nSaved accuracy_results.csv")