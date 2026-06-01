import pandas as pd
import networkx as nx


def load_graph_from_csv(
    edges_file: str = "ace_graph_edges.csv",
    nodes_file: str = "ace_graph_nodes.csv",
) -> nx.DiGraph:
    g = nx.DiGraph()

    nodes_df = pd.read_csv(nodes_file)
    g.add_nodes_from(nodes_df["concept"].astype(str))

    edges_df = pd.read_csv(edges_file)
    g.add_edges_from(edges_df[["source", "target"]].itertuples(index=False, name=None))

    return g


def compute_learning_effort(g: nx.DiGraph) -> dict[str, float]:
    n = len(g)

    if n <= 1:
        return {node: 0.0 for node in g.nodes()}

    return {
        node: len(nx.ancestors(g, node)) / (n - 1)
        for node in g.nodes()
    }


def compute_graph_metrics(g: nx.DiGraph) -> pd.DataFrame:
    pagerank = nx.pagerank(g, alpha=0.85)
    betweenness = nx.betweenness_centrality(g, normalized=True)
    out_closeness = nx.closeness_centrality(
        g.reverse(copy=False),
        wf_improved=True,
    )
    learning_effort = compute_learning_effort(g)

    rows = []
    for node in g.nodes():
        rows.append({
            "concept": node,
            "pagerank": pagerank[node],
            "betweenness_centrality": betweenness[node],
            "out_closeness_centrality": out_closeness[node],
            "learning_effort": learning_effort[node],
        })

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["pagerank", "learning_effort", "betweenness_centrality"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return df


def save_metrics(
    edges_file: str = "ace_graph_edges.csv",
    nodes_file: str = "ace_graph_nodes.csv",
    output_metrics_file: str = "ace_graph_metrics.csv",
):
    g = load_graph_from_csv(
        edges_file=edges_file,
        nodes_file=nodes_file,
    )

    metrics_df = compute_graph_metrics(g)
    metrics_df.to_csv(output_metrics_file, index=False, encoding="utf-8-sig")

    print(f"Метрики сохранены в файл: {output_metrics_file}")
    print(f"Всего концептов: {len(g)}")
    print("\nПервые строки результата:\n")
    print(metrics_df.head(20).to_string(index=False))


if __name__ == "__main__":
    save_metrics(
        edges_file="ace_graph_edges.csv",
        nodes_file="ace_graph_nodes.csv",
        output_metrics_file="ace_graph_metrics.csv",
    )
