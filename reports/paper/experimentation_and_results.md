The proposed method is evaluated on the STL-10 dataset using a subset of 500 images. A total of 200 query samples are used for evaluation. Performance is measured using Precision@K for K = 3, 5, and 10.

Results are reported as mean ± standard deviation.

The experimental results demonstrate that:

* Raw embeddings provide strong baseline performance
* Edge-based features alone perform poorly due to lack of semantic information
* Hybrid fusion consistently improves retrieval performance over individual methods

The best performance is observed at moderate fusion weights (α = 0.3–0.5), indicating an optimal balance between structural and semantic features.

Additionally, performance decreases significantly at high α values (e.g., 0.9), suggesting that excessive reliance on structural features negatively impacts retrieval accuracy.

These findings validate the effectiveness of the proposed hybrid feature fusion approach.
