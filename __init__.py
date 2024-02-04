import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import OPTICS as skOptics
from sklearn.cluster import KMeans as skKMeans
from sklearn.decomposition import PCA as skPca

warnings.filterwarnings("ignore", category=FutureWarning)


class Deck2Vector:
    def __init__(self, decklists: list, **kwargs):
        self._decklists = decklists
        self._card_names = list(set(name for deck in decklists for _, name in deck))
        self._card_index = self._build_card_index()
        self._vectors = np.array([self._deck_to_vector(deck) for deck in decklists])

    def _build_card_index(self):
        index = {}
        for idx, card in enumerate(self._card_names):
            index[card] = idx
        return index

    def _deck_to_vector(self, deck):
        vector = np.zeros(len(self._card_names), dtype=int)
        for qty, card in deck:
            vector[self._card_index[card]] = qty
        return vector


class ACP(Deck2Vector):
    def __init__(self, decklists: list, **kwargs):
        super().__init__(decklists, **kwargs)

        pca = skPca(n_components=kwargs.get("n_components", 2))
        self._vectors = pca.fit_transform(self._vectors)

    def _plot_pca(self, output: str = ""):
        plt.figure(figsize=(10, 6))
        plt.scatter(
            self._vectors[:, 0],
            self._vectors[:, 1],
            c=self.labels,
            cmap="viridis",
            s=10,
        )
        plt.title("PCA Plot")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        if output:
            plt.savefig(output)
        else:
            plt.show()
        plt.close()


class OpticsACP(ACP):
    def __init__(self, decklists: list, **kwargs):
        super().__init__(decklists, **kwargs)

        self.optics = skOptics(min_samples=5, eps=0.5)
        self.optics.fit(self._vectors)
        self.labels = self.optics.labels_

    def plot(self, output: str = ""):
        reachability = self.optics.reachability_

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(reachability)), np.sort(reachability))
        plt.title("Reachability Plot")
        plt.xlabel("Data Point Index")
        plt.ylabel("Reachability Distance")
        if output:
            plt.savefig(f"reachability_{output}")
        else:
            plt.show()
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(
            np.arange(len(reachability)),
            reachability,
            c=self.labels,
            cmap="viridis",
            s=10,
        )
        plt.title("Clustered Reachability Plot")
        plt.xlabel("Data Point Index")
        plt.ylabel("Reachability Distance")
        if output:
            plt.savefig(output)
        else:
            plt.show()
        plt.close()

    def decks_by_label(self, un_label):
        deck_label_tuple = self._add_label_to_decks()
        return [
            (deck, label) for (deck, label) in deck_label_tuple if label == un_label
        ]

    def _add_label_to_decks(self):
        labels = list(self.labels)
        return list(zip(self._decklists, labels))


class KMeansACP(ACP):
    def __init__(self, decklists: list, **kwargs):
        super().__init__(decklists, **kwargs)

        self._random_state = kwargs.get("random_state", 123)
        self.nb_clusters = kwargs.get(
            "n_clusters",
            self._recommand_k(
                kwargs.get("min_clusters", 2), kwargs.get("max_clusters", 20)
            ),
        )

        self.kmeans = skKMeans(
            n_clusters=self.nb_clusters, random_state=self._random_state
        )
        self.kmeans.fit(self._vectors)
        self.labels = self.kmeans.labels_

    def _recommand_k(self, start: int = 2, end: int = 20) -> int:
        sum_squared_errors = []

        for k in range(start, min(end, len(self._decklists)) + 1):
            kmeans = skKMeans(n_clusters=k, random_state=self._random_state)
            kmeans.fit(self._vectors)
            sum_squared_errors.append(kmeans.inertia_)

        sse_derivation2 = [abs(x) for x in np.gradient(np.gradient(sum_squared_errors))]

        return int(np.argmin(sse_derivation2[:-1])) + 2

    def plot(self, output: str = ""):
        plt.figure(figsize=(10, 6))

        label_colors = {
            label: f"C{i}" for i, label in enumerate(np.unique(self.labels))
        }
        for label, color in label_colors.items():
            mask = self.labels == label
            plt.scatter(
                self._vectors[mask, 0],
                self._vectors[mask, 1],
                c=color,
                s=10,
                label=f"Cluster {label+1}",
            )

        plt.title("K-Means Clustering with PCA")
        ax = plt.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        plt.legend()

        if output:
            plt.savefig(output)
        else:
            plt.show()
        plt.close()

    def decks_by_label(self, un_label):
        deck_label_tuple = self._add_label_to_decks()
        return [
            (deck, label) for (deck, label) in deck_label_tuple if label == un_label
        ]

    def _add_label_to_decks(self):
        labels = list(self.labels)
        return list(zip(self._decklists, labels))
