The proposed system consists of four main stages: preprocessing, feature extraction, feature fusion, and retrieval.

First, input images are converted into edge representations using Canny edge detection to simulate sketch inputs. Next, feature extraction is performed using a pretrained ResNet18 model, where the final classification layer is removed to obtain 512-dimensional embeddings.

Two types of features are extracted:

* Raw image embeddings capturing semantic information
* Edge image embeddings capturing structural information

The final representation is obtained through weighted fusion:

F = αE + (1 - α)R

where E represents edge embeddings, R represents raw embeddings, and α is the fusion weight.

Similarity between query and database images is computed using cosine similarity, and top-K results are retrieved based on similarity scores.
