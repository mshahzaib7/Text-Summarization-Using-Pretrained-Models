
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import evaluate
import nltk
from gensim.summarization import summarize as textrank_summarize

# Download NLTK data (for sentence tokenization)
nltk.download('punkt')

# ==========================================
print("Loading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Use small subset for quick execution
train_data = dataset["train"].select(range(100))
test_data = dataset["test"].select(range(10))


# ==========================================
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


# ==========================================
def abstractive_summary(text, max_len=130, min_len=30):
    text = text.strip().replace("\n", " ")
    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
    return summary


# ==========================================
def extractive_summary(text, ratio=0.2):
    try:
        return textrank_summarize(text, ratio=ratio)
    except ValueError:
        return text  # if text too short


# ==========================================
print("\nGenerating sample summaries...")
sample_text = test_data[0]["article"]
reference_summary = test_data[0]["highlights"]

abs_summary = abstractive_summary(sample_text)
ext_summary = extractive_summary(sample_text)

print("\nOriginal Text:\n", sample_text[:500], "...")
print("\nReference Summary:\n", reference_summary)
print("\nAbstractive Summary (BART):\n", abs_summary)
print("\nExtractive Summary (TextRank):\n", ext_summary)


# ==========================================
rouge = evaluate.load("rouge")

# Prepare small batch for evaluation
texts = [item["article"] for item in test_data[:5]]
refs = [item["highlights"] for item in test_data[:5]]

abs_preds = [abstractive_summary(t) for t in texts]
results = rouge.compute(predictions=abs_preds, references=refs)

print("\nROUGE Evaluation Results (Abstractive Model):")
for k, v in results.items():
    print(f"{k}: {v:.4f}")



# ==========================================
output_df = pd.DataFrame({
    "Article": texts,
    "Reference Summary": refs,
    "Abstractive Summary": abs_preds,
    "Extractive Summary": [extractive_summary(t) for t in texts]
})
output_df.to_csv("summaries_output.csv", index=False)
print("\nSummaries saved to 'summaries_output.csv'.")
