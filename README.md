# Tweet Sentiment Classification (RNN / LSTM, PyTorch)

Sentiment classification with recurrent neural networks in PyTorch, built during my AI/ML
internship at the Open Source Intelligence Lab, IIT Guwahati.

## Notebooks
- `RNN_for_SentimentAnalysis.ipynb`: baseline RNN on IMDB movie reviews (binary, TF-IDF
  features), 85.7% test accuracy
- `Tweet_Sentiment_LSTM.ipynb`: bidirectional LSTM on tweets (TweetEval, 3-class) with
  full evaluation and error analysis

## Tweet Sentiment LSTM

### Dataset
TweetEval (sentiment), loaded via Hugging Face `datasets`. Test set: 12,284 tweets.

### Model
- Text cleaning, vocabulary (min frequency 2), padding to 40 tokens
- Embedding (128) → Bidirectional LSTM (128) → Dropout (0.3) → Linear
- Adam (lr 1e-3), 8 epochs, best checkpoint chosen by validation accuracy

### Results (test set)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.581 | 0.474 | 0.522 |
| Neutral | 0.601 | 0.628 | 0.614 |
| Positive | 0.515 | 0.616 | 0.561 |

**Accuracy: 57.6% | Macro F1: 0.566**

### Error analysis
5,211 of 12,284 tweets were misclassified. About 90% of the errors involved the neutral
class; the biggest confusion was negative predicted as neutral (1,733 tweets). Sampled
errors showed sarcasm, implicit negativity in headline-style tweets, mixed sentiment and
opinion-less tweets (stock tickers, name lists) as common patterns.

### Possible improvements
Pretrained embeddings (GloVe), keeping emojis, class weights, transformer fine-tuning.

## Run it
Open a notebook in Google Colab (T4 GPU) and run all cells.
