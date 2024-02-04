import glob
import os
from datetime import datetime

from mtgdc_aggregator import Aggregator
from mtgdc_decklists import ImportDecks

from . import KMeansACP, OpticsACP

if __name__ == "__main__":
    for file in glob.glob(os.path.join(".", "*_cluster_*")):
        os.remove(file)

    print(".", "Chargement des decks")
    liste_decks = ImportDecks.from_directory("./mtgdc_decklists/decklists")
    liste_decks.load_decks(commander=["Kess, Dissident Mage"])
    print("\t", f"{len(liste_decks.decklists)} decks chargés")

    print(".", "Analyse en Composantes Principales et Clustering")
    kmeans_with_pca = KMeansACP(liste_decks.decklists, max_clusters=5)
    kmeans_with_pca.plot(output="result_plot.pdf")
    print("\t", f"KMeans: {kmeans_with_pca.nb_clusters} clusters identifiés")

    print(".", "Decks par label et aggrégation")
    for k in sorted(list(set(kmeans_with_pca.labels))):
        decks_label = [deck for deck, _ in kmeans_with_pca.decks_by_label(k)]
        aggregate = Aggregator(decks_label)
        aggregate.aggregate(action=f"\t Cluster {k+1}")
        aggregate.export(f"kmeans_cluster_{k+1}.txt", title=f"Cluster {k+1}")

    """
    optics_with_acp = OpticsACP(liste_decks.decklists)
    optics_with_acp.plot(output="optics_result_plot.pdf")
    print("\t", f"OPTICS: {len(list(set(optics_with_acp.labels)))} clusters identifiés")

    print(".", "Decks par label et aggrégation")
    print("\t", "Via KMeans")
    for k in sorted(list(set(kmeans_with_pca.labels))):
        decks_label = [deck for deck, _ in kmeans_with_pca.decks_by_label(k)]
        aggregate = Aggregator(decks_label)
        aggregate.aggregate(action=f"\t Cluster {k+1}")
        aggregate.export(f"kmeans_cluster_{k+1}.txt", title=f"Cluster {k+1}")

    print("\t", "Via OPTICS")
    for k in sorted(list(set(optics_with_acp.labels))):
        decks_label = [deck for deck, _ in optics_with_acp.decks_by_label(k)]
        aggregate = Aggregator(decks_label)
        aggregate.aggregate(action=f"\t Cluster {k+1}")
        aggregate.export(f"optics_cluster_{k+1}.txt", title=f"Cluster {k+1}")
    """
