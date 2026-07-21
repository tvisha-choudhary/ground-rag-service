"""Dense retrieval via TF-IDF plus truncated SVD (latent semantic analysis).

The honest offline choice (docs/adr/0001): this environment and the CI
runners download no pretrained models, so the dense leg is a classical
embedding learned from the corpus itself. The Retriever interface is
where a sentence-transformer or API embedding model plugs in unchanged;
the harness measures whatever sits behind the interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class LSAIndex:
    def __init__(self, dims: int = 256, seed: int = 0) -> None:
        self.dims = dims
        self.seed = seed
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._matrix: np.ndarray | None = None

    def fit(self, texts: list[str]) -> "LSAIndex":
        if not texts:
            raise ValueError("cannot index an empty corpus")
        self._vectorizer = TfidfVectorizer(token_pattern=r"[a-z0-9]+(?:-[a-z0-9]+)*",
                                           lowercase=True, sublinear_tf=True)
        tfidf = self._vectorizer.fit_transform(texts)
        dims = min(self.dims, tfidf.shape[1] - 1, tfidf.shape[0] - 1)
        self._svd = TruncatedSVD(n_components=max(dims, 2), random_state=self.seed)
        self._matrix = normalize(self._svd.fit_transform(tfidf))
        return self

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        if self._vectorizer is None:
            raise RuntimeError("index not fitted")
        vec = normalize(self._svd.transform(self._vectorizer.transform([query])))
        sims = (self._matrix @ vec.T).ravel()
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]
