# Text Embedding Analysis Report
**Owen Zhang | EE P 596 A | Mini Project 1**

## Part A: Word Order Sensitivity

**Test:** Categories `Chocolate Milk` with inputs "chocolate milk" vs "milk chocolate"

| Input | GloVe 50d | Sentence Transformer | OpenAI Small | OpenAI Large |
|-------|-----------|---------------------|--------------|--------------|
| chocolate milk | Chocolate (2.65) | Chocolate (2.32) | Milk (1.88) | Milk (1.88) |
| milk chocolate | Chocolate (2.65) | Chocolate (2.38) | Milk (1.88) | Milk (1.79) |

**Which models got it right?** None fully distinguish the semantic difference between "chocolate milk" (beverage) and "milk chocolate" (candy). All maintain the same top category regardless of word order.

**Why did some fail?**
- **GloVe:** Uses bag-of-words averaging—identical scores because word order is completely ignored
- **Transformers/OpenAI:** Though they encode position, the semantic difference between these compound terms is subtle; models treat them as similar food concepts

**What this reveals:** Word order sensitivity is limited across all models for compound terms. GloVe has zero sensitivity; transformer models show minor score variations but no classification change.

---

## Part B: Model Comparison

**Sentiment Test:** Categories `Positive Negative`

| Input | GloVe 50d | Sentence Transformer | OpenAI Small | OpenAI Large |
|-------|-----------|---------------------|--------------|--------------|
| "The movie was upsetting" | Positive ❌ | Negative ✓ | Negative ✓ | Negative ✓ |
| "This is terrible" | Positive ❌ | Negative ✓ | Negative ✓ | Negative ✓ |

**Comparison Table:**

| Metric | GloVe 50d | Sentence Trans. 384d | OpenAI Small 1536d | OpenAI Large 3072d |
|--------|-----------|---------------------|-------------------|-------------------|
| Sentiment Accuracy | 0% | 100% | 100% | 100% |
| Word Order Sensitivity | None | Low | Low | Low |
| Dimensionality | 50 | 384 | 1536 | 3072 |
| Speed | Fastest | Fast | Slow (API) | Slower (API) |
| Cost | Free | Free | ~$0.02/1M tokens | ~$0.13/1M tokens |

**Key Findings:**
1. **Accuracy:** GloVe fails sentiment tasks; transformer models succeed
2. **Dimensionality vs Performance:** Higher dimensions help (GloVe 50d fails, others succeed), but OpenAI Small vs Large shows diminishing returns
3. **Speed:** Local models (GloVe, Sentence Transformers) are faster than API calls
4. **Word Order:** All models have limited sensitivity to word order in compound terms

---

## Part C: Real-World Applications

### Category Group 1: E-commerce Products
**Categories:** `Kitchen Furniture Baby Outdoor`

| Input | Top Result | Explanation |
|-------|-----------|-------------|
| "baby furniture safety covers" | Furniture | "furniture" appears first, dominates classification |
| "furniture for baby room" | Baby/Furniture | "baby" at end shifts toward Baby category |

**Why it changes:** Word position affects averaging weight in GloVe; transformers give slightly more attention to later context words.

### Category Group 2: Food Services
**Categories:** `Pizza Burger Sushi Tacos`

| Input | Top Result | Explanation |
|-------|-----------|-------------|
| "cheese pepperoni order" | Pizza ✓ | All models correctly associate toppings with pizza |
| "order pepperoni cheese" | Pizza ✓ | Word order doesn't affect this—vocabulary is strongly pizza-specific |

**Why no change:** The vocabulary (pepperoni, cheese) has strong pizza associations regardless of order.

### Category Group 3: Multi-Topic Sentences
**Categories:** `Flowers Colors Cars Weather Food`

| Input | GloVe | Sent. Trans. | OpenAI Small | OpenAI Large |
|-------|-------|-------------|--------------|--------------|
| "Roses are red, trucks are blue" | Food | Colors | Colors | Flowers |

**Why models disagree:**
- GloVe averages all words, getting confused by mixed topics
- Sentence Transformers focuses on explicit color words (red, blue)
- OpenAI Large prioritizes the sentence subject (roses → Flowers)

---

## Summary

| Model | Best For | Avoid For |
|-------|----------|-----------|
| GloVe | Fast keyword matching, low resources | Sentiment, context-dependent tasks |
| Sentence Transformers | Balanced quality/speed, local deployment | When API latency is acceptable |
| OpenAI Small | Production apps needing good accuracy | Cost-sensitive applications |
| OpenAI Large | Maximum accuracy requirements | Most use cases (diminishing returns) |

**Deployed App:** https://eep596a-miniproj-iidmokgr2nhdgemwc4rmbl.streamlit.app/
