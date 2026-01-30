# Analysis Report: Text Embedding Comparison for Search-Based Retrieval

**Author:** Owen Zhang  
**Course:** EE P 596 A  
**Assignment:** Mini Project 1 - Text Embedding Comparison

---

## Introduction

This report analyzes the behavior of four different text embedding models in a search-based retrieval application. The models compared are:

1. **GloVe (50d)**: Pre-trained word embeddings from Twitter data (2 billion tweets)
2. **Sentence Transformers (384d)**: Fine-tuned `all-MiniLM-L6-v2` model for sentence similarity
3. **OpenAI Small (1536d)**: `text-embedding-3-small` API model
4. **OpenAI Large (3072d)**: `text-embedding-3-large` API model

Each model computes cosine similarity between an input sentence and category words, then ranks categories by semantic relevance.

---

## Part A: Word Order Sensitivity Analysis

### Test Case: "Chocolate Milk" vs "Milk Chocolate"

This test examines whether embedding models treat word order as semantically significant.

**Categories:** `Chocolate Milk`

| Input | GloVe (50d) | Sentence Transformer | OpenAI Small | OpenAI Large |
|-------|-------------|---------------------|--------------|--------------|
| "chocolate milk" | Chocolate (2.6498) | Chocolate (2.3162) | Milk (1.8816) | Milk (1.8839) |
| "milk chocolate" | Chocolate (2.6498) | Chocolate (2.3837) | Milk (1.8781) | Milk (1.7927) |

### Observations

1. **GloVe (Bag-of-Words):** Shows identical scores regardless of word order. This is expected because GloVe uses averaged word embeddings, treating sentences as a "bag of words" without positional information.

2. **Sentence Transformers:** Shows slight score variation (2.3162 vs 2.3837) but maintains the same top category (Chocolate). The transformer architecture captures some positional context but not enough to change classification.

3. **OpenAI Models:** Both consistently classify toward "Milk" regardless of word order, with minor score variations. These models may semantically group both phrases similarly since "chocolate milk" and "milk chocolate" are both food-related compound terms.

### Conclusion

Word order has minimal impact on category classification for these specific inputs. GloVe's bag-of-words approach shows no sensitivity, while transformer-based models show subtle numerical differences without changing the final classification.

---

## Part B: Sentiment Analysis Limitations

### Test Case: Negative Sentiment Detection

**Categories:** `Positive Negative`

| Input | GloVe (50d) | Sentence Transformer | OpenAI Small | OpenAI Large |
|-------|-------------|---------------------|--------------|--------------|
| "The movie was upsetting" | **Positive** (1.8110) ❌ | Negative (1.1408) ✓ | Negative (1.2862) ✓ | Negative (1.2693) ✓ |
| "This is terrible" | **Positive** (1.9140) ❌ | Negative (1.3541) ✓ | Negative (1.3417) ✓ | Negative (1.3204) ✓ |

### Critical Findings

**GloVe Fails at Sentiment Classification:**

GloVe incorrectly classifies clearly negative sentences as "Positive" in both test cases. This is a fundamental limitation of word-level embeddings:

1. **No Contextual Understanding:** GloVe embeddings are trained on word co-occurrence statistics, not sentiment. Words like "upsetting" and "terrible" may appear in various contexts (positive and negative) in Twitter data.

2. **Averaging Dilutes Meaning:** When averaging word vectors, negation and emotional modifiers lose their impact. The word "movie" or "is" may have neutral embeddings that dilute the negative sentiment.

3. **Static Embeddings:** Each word has a single fixed embedding regardless of context, missing nuances like sarcasm or contextual sentiment.

**Modern Models Succeed:**

Sentence Transformers and OpenAI embeddings correctly identify negative sentiment because:

- They process entire sentences contextually
- They're fine-tuned on semantic similarity tasks that capture emotional meaning
- They have much higher dimensionality (384d-3072d vs 50d) to represent subtle meanings

### Recommendation

For sentiment-aware applications, **avoid using GloVe or similar word-level embeddings**. Use sentence-level or contextual embeddings like Sentence Transformers or OpenAI embeddings.

---

## Part C: Real-World Application Testing

### Application 1: Food Ordering System

**Use Case:** A voice assistant needs to categorize customer orders to the correct restaurant type.

**Categories:** `Pizza Burger Sushi Tacos`

| Input | GloVe | Sentence Transformer | OpenAI Small | OpenAI Large |
|-------|-------|---------------------|--------------|--------------|
| "I want cheese and pepperoni on my order" | Pizza (2.0556) ✓ | Pizza (1.6406) ✓ | Pizza (1.5361) ✓ | Pizza (1.5094) ✓ |

**Result:** All models correctly associate pizza-specific toppings with the Pizza category. This demonstrates that embeddings can effectively capture domain-specific vocabulary associations.

### Application 2: Default Multi-Category Demo

**Use Case:** General semantic search across diverse categories.

**Categories:** `Flowers Colors Cars Weather Food`

**Input:** "Roses are red, trucks are blue, and Seattle is grey right now"

| Model | Top Category | Score |
|-------|--------------|-------|
| glove_50d | Food | 2.0814 |
| sentence_transformer_384 | Colors | 1.7090 |
| openai_small_1536 | Colors | 1.3676 |
| openai_large_3072 | Flowers | 1.3515 |

**Analysis:** This complex sentence mentions multiple categories. Models disagree on the primary topic:
- **GloVe** picks "Food" (possibly due to color word associations in food contexts)
- **Sentence Transformers & OpenAI Small** pick "Colors" (red, blue, grey are explicitly mentioned)
- **OpenAI Large** picks "Flowers" (roses are the subject of the sentence)

This shows that different models emphasize different semantic aspects of complex inputs.

### Application 3: Word Order in Compound Terms

**Categories:** `Chocolate Milk`

Testing whether models distinguish between:
- "chocolate milk" (a beverage - milk flavored with chocolate)
- "milk chocolate" (a type of candy)

As analyzed in Part A, these models don't strongly distinguish between word orders in compound terms, which could be problematic for e-commerce search where "chicken fried" vs "fried chicken" matters significantly.

---

## Model Comparison Summary

| Characteristic | GloVe (50d) | Sentence Transformer (384d) | OpenAI Small (1536d) | OpenAI Large (3072d) |
|----------------|-------------|----------------------------|---------------------|---------------------|
| **Architecture** | Word-level, averaged | Transformer, sentence-level | Transformer, API | Transformer, API |
| **Dimensionality** | Low (50) | Medium (384) | High (1536) | Very High (3072) |
| **Word Order Sensitivity** | None | Low | Low | Low |
| **Sentiment Understanding** | Poor | Good | Good | Good |
| **Speed** | Fastest (local) | Fast (local) | Slow (API call) | Slower (API call) |
| **Cost** | Free | Free | Paid per token | Paid per token |

---

## Key Insights and Recommendations

### 1. Choose the Right Model for the Task

- **GloVe:** Best for simple keyword matching, topic classification, or when computational resources are limited. Avoid for sentiment analysis.
- **Sentence Transformers:** Excellent balance of quality and speed for semantic similarity tasks. Runs locally without API costs.
- **OpenAI Embeddings:** Best for production applications requiring high accuracy. Consider cost-benefit tradeoff.

### 2. GloVe Limitations Are Significant

The sentiment analysis failure demonstrates that GloVe's bag-of-words approach has fundamental limitations for understanding meaning beyond word co-occurrence. Modern applications should prefer contextual embeddings.

### 3. Dimensionality Matters (With Diminishing Returns)

Higher dimensionality generally allows for more nuanced representations, but the difference between OpenAI Small (1536d) and Large (3072d) is often minimal in practice. The cost/performance tradeoff favors the smaller model in most use cases.

### 4. Word Order Remains Challenging

All tested models show limited sensitivity to word order in compound terms. Applications requiring word-order awareness may need additional preprocessing or specialized models.

---

## Conclusion

This project demonstrates the practical differences between embedding approaches in a search-based retrieval system. While GloVe remains useful for simple applications, modern transformer-based embeddings (Sentence Transformers, OpenAI) provide significantly better semantic understanding, particularly for sentiment-aware applications. The choice of embedding model should be driven by the specific requirements of the application, balancing accuracy, speed, and cost considerations.

---

**Appendix: Technical Implementation Notes**

- Cosine similarity is computed using `numpy.dot(x, y) / (numpy.linalg.norm(x) * numpy.linalg.norm(y))`
- Results are exponentiated (`math.exp(cosine_similarity)`) to provide positive, easily comparable scores
- Category embeddings are cached in Streamlit session state to avoid redundant API calls
- GloVe embeddings are downloaded from Google Drive and cached locally
