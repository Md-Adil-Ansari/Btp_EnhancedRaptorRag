import os
import sys
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tiktoken
from pyvis.network import Network
from scipy import spatial

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from enhancedRaptor.EmbeddingModels import SBertEmbeddingModel
from enhancedRaptor.tree_structures import Node, Tree
from enhancedRaptor.utils import reverse_mapping

EMB_KEY = "SBert"
TREES_DIR = ROOT / "trees"

st.set_page_config(page_title="RAPTOR Tree Visualizer", layout="wide")


@st.cache_resource(show_spinner="Loading SBert embedder...")
def get_embedder() -> SBertEmbeddingModel:
    return SBertEmbeddingModel()


@st.cache_resource
def get_tokenizer():
    return tiktoken.get_encoding("cl100k_base")


@st.cache_resource(show_spinner=False)
def load_tree(path: str) -> Tree:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_tree_path(p: Path):
    name = p.stem.replace("_tree", "")
    doc_id, _, variant = name.rpartition("_")
    return doc_id, variant


def list_trees():
    return sorted(TREES_DIR.glob("*_tree.pkl"))


def child_to_parent_map(tree: Tree):
    m = {}
    for idx, node in tree.all_nodes.items():
        for c in node.children:
            m[c] = idx
    return m


def collect_ancestors(tree: Tree, selected_indices):
    selected = set(selected_indices)
    ancestors = set()
    c2p = child_to_parent_map(tree)
    for s in selected:
        cur = s
        while cur in c2p:
            p = c2p[cur]
            if p not in selected:
                ancestors.add(p)
            cur = p
    return ancestors


def collect_descendants(tree: Tree, idx: int):
    out = set()
    stack = [idx]
    while stack:
        cur = stack.pop()
        for c in tree.all_nodes[cur].children:
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def retrieve_with_distances(tree: Tree, query: str, top_k: int, max_tokens: int,
                            embedder, tokenizer):
    """Faithful re-implementation of TreeRetriever.retrieve_information_collapse_tree
    that also returns per-node distances for the visualization."""
    q_emb = embedder.create_embedding(query)
    node_to_layer = reverse_mapping(tree.layer_to_nodes)
    indices_sorted = sorted(tree.all_nodes.keys())
    nodes = [tree.all_nodes[i] for i in indices_sorted]
    embeddings = [n.embeddings[EMB_KEY] for n in nodes]
    distances = np.array([spatial.distance.cosine(q_emb, e) for e in embeddings])
    order = np.argsort(distances)

    selected = []
    total_tokens = 0
    # Match original logic: walk top_k, break on first overflow
    for rank, idx in enumerate(order[:top_k]):
        node = nodes[idx]
        n_tokens = len(tokenizer.encode(node.text))
        if total_tokens + n_tokens > max_tokens:
            break
        selected.append({
            "node": node,
            "rank": rank,
            "global_rank": int(rank),
            "distance": float(distances[idx]),
            "layer": node_to_layer[node.index],
            "tokens": n_tokens,
        })
        total_tokens += n_tokens

    all_info = []
    for global_rank, pos in enumerate(order):
        n = nodes[pos]
        all_info.append({
            "index": n.index,
            "layer": node_to_layer[n.index],
            "distance": float(distances[pos]),
            "rank": global_rank,
            "text": n.text,
            "tokens": len(tokenizer.encode(n.text)),
        })

    return selected, all_info, total_tokens


def build_pyvis_html(tree: Tree, selected_indices: set, ancestor_indices: set,
                     height_px: int = 700) -> str:
    node_to_layer = reverse_mapping(tree.layer_to_nodes)
    max_layer = max(node_to_layer.values())

    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#0e1117",
        font_color="white",
        directed=True,
    )
    net.set_options("""
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "DU",
          "sortMethod": "directed",
          "levelSeparation": 130,
          "nodeSpacing": 110,
          "treeSpacing": 180
        }
      },
      "physics": {
        "enabled": true,
        "hierarchicalRepulsion": {"nodeDistance": 140, "springLength": 100},
        "stabilization": {"iterations": 250}
      },
      "interaction": {"hover": true, "tooltipDelay": 80, "navigationButtons": true},
      "edges": {"smooth": {"type": "cubicBezier", "forceDirection": "vertical"}}
    }
    """)

    # Build a JS-safe dict of full node texts for the click popup
    import html as html_mod
    node_full_texts = {}

    for idx, node in tree.all_nodes.items():
        layer = node_to_layer[idx]
        is_selected = idx in selected_indices
        is_ancestor = idx in ancestor_indices

        if is_selected:
            color = "#ff4b4b"
            size = 30
        elif is_ancestor:
            color = "#ffa500"
            size = 22
        elif layer == 0:
            color = "#4a90e2"
            size = 14
        else:
            color = "#9b59b6"
            size = 18

        snippet = node.text[:280].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
        if len(node.text) > 280:
            snippet += "..."
        title = f"<b>Layer {layer} | idx {idx}</b><br/>{snippet}<br/><i>(click for full text)</i>"
        label = f"L{layer}#{idx}"

        net.add_node(idx, label=label, title=title, color=color, size=size,
                     level=max_layer - layer)

        # Store the full text for the JS popup
        full_text = html_mod.escape(node.text).replace("\n", "<br/>")
        node_full_texts[idx] = {"layer": layer, "text": full_text}

    for idx, node in tree.all_nodes.items():
        for child_idx in node.children:
            net.add_edge(idx, child_idx, color="#555", arrows="")

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp:
        tmp_path = tmp.name
    net.write_html(tmp_path, open_browser=False, notebook=False)
    html = Path(tmp_path).read_text(encoding="utf-8")
    Path(tmp_path).unlink(missing_ok=True)

    # Build the JS data object for node texts
    import json as json_mod
    node_data_js = json_mod.dumps(node_full_texts)

    # Inject modal CSS + JS into the HTML before </body>
    modal_injection = f"""
    <style>
      #nodeModal {{
        display: none;
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background: rgba(0,0,0,0.75);
        z-index: 9999;
        justify-content: center;
        align-items: center;
      }}
      #nodeModal.active {{
        display: flex;
      }}
      #nodeModalContent {{
        background: #1a1a2e;
        color: #e0e0e0;
        border: 1px solid #4a90e2;
        border-radius: 12px;
        padding: 24px 28px;
        max-width: 700px;
        width: 90%;
        max-height: 75vh;
        overflow-y: auto;
        box-shadow: 0 8px 32px rgba(74, 144, 226, 0.3);
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        line-height: 1.6;
      }}
      #nodeModalHeader {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid #333;
      }}
      #nodeModalHeader h3 {{
        margin: 0;
        color: #4a90e2;
        font-size: 18px;
      }}
      #nodeModalClose {{
        cursor: pointer;
        background: #ff4b4b;
        color: white;
        border: none;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        font-size: 18px;
        line-height: 30px;
        text-align: center;
      }}
      #nodeModalClose:hover {{
        background: #e03030;
      }}
      #nodeModalBody {{
        white-space: pre-wrap;
        word-wrap: break-word;
      }}
    </style>

    <div id="nodeModal" onclick="if(event.target===this)closeNodeModal()">
      <div id="nodeModalContent">
        <div id="nodeModalHeader">
          <h3 id="nodeModalTitle"></h3>
          <button id="nodeModalClose" onclick="closeNodeModal()">&times;</button>
        </div>
        <div id="nodeModalBody"></div>
      </div>
    </div>

    <script>
      var nodeFullTexts = {node_data_js};

      function closeNodeModal() {{
        document.getElementById('nodeModal').classList.remove('active');
      }}

      // Wait for the vis network to be ready, then attach click handler
      var checkNetwork = setInterval(function() {{
        if (typeof network !== 'undefined' && network !== null) {{
          clearInterval(checkNetwork);
          network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
              var nodeId = params.nodes[0];
              var data = nodeFullTexts[nodeId];
              if (data) {{
                document.getElementById('nodeModalTitle').textContent =
                  'Layer ' + data.layer + ' | Node #' + nodeId;
                document.getElementById('nodeModalBody').innerHTML = data.text;
                document.getElementById('nodeModal').classList.add('active');
              }}
            }}
          }});
        }}
      }}, 200);
    </script>
    """

    html = html.replace("</body>", modal_injection + "\n</body>")
    return html


def render_variant(doc_id: str, variant: str, query: str, top_k: int,
                   max_tokens: int, run: bool, embedder, tokenizer, gen_answer: bool,
                   cite_sources: bool = False):
    tree_path = TREES_DIR / f"{doc_id}_{variant}_tree.pkl"
    if not tree_path.exists():
        st.warning(f"No tree at {tree_path.name}")
        return None

    tree = load_tree(str(tree_path))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total nodes", len(tree.all_nodes))
    c2.metric("Leaves", len(tree.leaf_nodes))
    c3.metric("Layers", tree.num_layers + 1)

    selected, all_info, used_tokens = [], [], 0
    if run and query.strip():
        with st.spinner(f"Retrieving from {variant}..."):
            selected, all_info, used_tokens = retrieve_with_distances(
                tree, query, top_k, max_tokens, embedder, tokenizer
            )

    selected_idx_set = {s["node"].index for s in selected}
    ancestor_set = collect_ancestors(tree, selected_idx_set) if selected_idx_set else set()

    st.markdown("#### Tree (red = selected, orange = ancestor of selected)")
    html = build_pyvis_html(tree, selected_idx_set, ancestor_set, height_px=620)
    components.html(html, height=640, scrolling=False)

    if not selected:
        if run and query.strip():
            st.warning("No nodes retrieved.")
        else:
            st.info("Enter a query in the sidebar and click Retrieve.")
        return tree

    st.markdown(f"#### Selected nodes — {len(selected)} chosen, {used_tokens}/{max_tokens} tokens used")

    df = pd.DataFrame([
        {
            "rank": s["rank"] + 1,
            "node_idx": s["node"].index,
            "layer": s["layer"],
            "distance": round(s["distance"], 4),
            "tokens": s["tokens"],
            "preview": s["node"].text[:120].replace("\n", " ") + ("..." if len(s["node"].text) > 120 else ""),
        }
        for s in selected
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Full text of each selected node", expanded=False):
        for s in selected:
            n = s["node"]
            icon = "📄" if s["layer"] == 0 else "📦"
            st.markdown(
                f"**#{s['rank']+1} {icon} Layer {s['layer']} · idx {n.index} · "
                f"distance {s['distance']:.4f} · tokens {s['tokens']}**"
            )
            st.write(n.text)
            if s["layer"] > 0:
                desc = collect_descendants(tree, n.index)
                leaf_desc = [d for d in desc if d in tree.leaf_nodes]
                st.caption(f"Summarizes {len(leaf_desc)} leaf chunks: {sorted(leaf_desc)}")
            st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### Selected by layer")
        layer_counts = Counter(s["layer"] for s in selected)
        layer_df = pd.DataFrame(
            [{"layer": L, "count": layer_counts.get(L, 0)} for L in sorted(layer_counts)]
        )
        st.bar_chart(layer_df, x="layer", y="count")

    with col_b:
        st.markdown("##### Top-30 nodes by distance (selected highlighted)")
        top30 = all_info[:30]
        chart_df = pd.DataFrame([
            {
                "rank": r["rank"] + 1,
                "distance": r["distance"],
                "selected": "selected" if r["index"] in selected_idx_set else "other",
            }
            for r in top30
        ])
        st.scatter_chart(chart_df, x="rank", y="distance", color="selected")

    st.markdown("#### Context passed to QA model")
    if cite_sources and selected:
        ctx = "\n\n".join(
            f"[L{s['layer']}#{s['node'].index}] {s['node'].text}" for s in selected
        )
    else:
        ctx = "\n\n".join(s["node"].text for s in selected)
    st.text_area("context", ctx, height=180, key=f"ctx_{variant}_{doc_id}")

    if gen_answer:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.warning("GEMINI_API_KEY not set — cannot generate live answer.")
        else:
            with st.spinner("Calling LLM model..."):
                try:
                    from enhancedRaptor.QAModels import GeminiQAModel
                    qa = GeminiQAModel(model="gemma-3-27b-it")
                    ans = qa.answer_question(ctx, query, cite_sources=cite_sources)
                    st.markdown(f"#### Generated answer ({variant.upper()})")
                    st.success(ans)
                except Exception as e:
                    st.error(f"QA failed: {e}")

    return tree


def page():
    st.title("RAPTOR Tree Visualizer")
    st.caption("Inspect which nodes drove a given answer — leaves vs cluster summaries, ranks, distances, lineage.")

    if not TREES_DIR.exists():
        st.error(f"No trees directory at {TREES_DIR}. Run the evaluator first.")
        return

    tree_files = list_trees()
    if not tree_files:
        st.error("No saved trees found. Run evaluate_raptor_models.py to build some.")
        return

    parsed = [parse_tree_path(p) for p in tree_files]
    docs = sorted({d for d, _ in parsed})

    with st.sidebar:
        st.header("Settings")
        doc_id = st.selectbox("Document", docs)
        available_variants = sorted({v for d, v in parsed if d == doc_id})
        compare = st.toggle(
            "Compare variants side-by-side",
            value=("normal" in available_variants and "ice" in available_variants),
            disabled=len(available_variants) < 2,
        )
        if not compare:
            chosen_variant = st.selectbox("Variant", available_variants)
            variants = [chosen_variant]
        else:
            variants = [v for v in ["normal", "ice"] if v in available_variants]

        st.divider()
        query = st.text_area(
            "Query", "", height=120,
            placeholder="Ask something about this document...",
        )
        top_k = st.slider("top_k", 1, 20, 10,
                          help="Top-K nearest nodes considered for the context.")
        max_tokens = st.slider("max_tokens (context budget)", 200, 5000, 3500, step=100,
                               help="Max total tokens of context fed to the QA LLM.")
        gen_answer = st.checkbox(
            "Generate live answer (LLM model)",
            value=False,
            help=(
                "When ON, after retrieval the app calls the QA LLM on the "
                "retrieved context to produce a fresh answer. "
                "Requires GEMINI_API_KEY in your env. "
                "Costs an API call per variant per click."
            ),
        )
        cite_sources = st.checkbox(
            "Cite sources [L#N]",
            value=False,
            help=(
                "When ON, each retrieved chunk is labeled with [L{layer}#{index}] "
                "and the LLM is instructed to cite these tags in its answer. "
                "Match the citations to nodes in the tree graph above."
            ),
        )
        run = st.button("Retrieve", type="primary", use_container_width=True)

    embedder = get_embedder()
    tokenizer = get_tokenizer()

    if len(variants) == 1:
        st.subheader(f"{variants[0].upper()} RAPTOR — {doc_id}")
        render_variant(doc_id, variants[0], query, top_k, max_tokens, run,
                       embedder, tokenizer, gen_answer, cite_sources)
    else:
        cols = st.columns(len(variants))
        for col, variant in zip(cols, variants):
            with col:
                st.subheader(f"{variant.upper()} RAPTOR")
                render_variant(doc_id, variant, query, top_k, max_tokens, run,
                               embedder, tokenizer, gen_answer, cite_sources)


page()
