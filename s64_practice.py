import json
from pathlib import Path
import duckdb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# ==============================================================================
# PILLAR 1: DESIGNING DATA STORES (BLOB, NoSQL, RELATIONAL & VECTOR)
# ==============================================================================
console.print(Panel.fit("[bold cyan]1. Designing Multi-Modal Data Stores (Blob + NoSQL + Relational + Vector)[/bold cyan]"))

# 1A. Blob Storage (Raw Unstructured Private Market Fund Memorandums)
blob_dir = Path("data/blob_storage")
blob_dir.mkdir(parents=True, exist_ok=True)

raw_documents = {
    "FUNDS_01.txt": "Apollo European Private Credit Fund IV focuses on senior secured direct lending across Western Europe. Features quarterly liquidity gates capped at 5% NAV and targets 11.5% net IRR with strict ESG covenant tracking.",
    "FUNDS_02.txt": "Blackstone APAC Infrastructure Partners invests in renewable energy, offshore wind, and data centers across Asia-Pacific. Long-term 10-year lockup structure with semi-annual distributions and inflation-linked yield protection.",
    "FUNDS_03.txt": "EQT Global Secondary Opportunities acquires mature private equity LP stakes at a discount. Offers accelerated J-curve mitigation, quarterly redemptions after year 1, and diversified vintage exposure across North America and EMEA."
}

for filename, text in raw_documents.items():
    (blob_dir / filename).write_text(text)

# 1B. NoSQL Document Store (JSONL Metadata & Flexible Attributes)
nosql_path = Path("data/nosql_fund_attributes.jsonl")
nosql_docs = [
    {"fund_id": "FUND-01", "asset_class": "Private Credit", "tags": ["Direct Lending", "ESG", "Quarterly Gate"]},
    {"fund_id": "FUND-02", "asset_class": "Infrastructure", "tags": ["Renewables", "Data Centers", "10Y Lockup"]},
    {"fund_id": "FUND-03", "asset_class": "Secondaries", "tags": ["LP Stakes", "J-Curve Mitigation", "Liquid"]}
]
with open(nosql_path, "w") as f:
    for doc in nosql_docs:
        f.write(json.dumps(doc) + "\n")

# 1C. Relational SQL Store (DuckDB / PostgreSQL pattern)
con = duckdb.connect("data/s64_platform.duckdb")

con.execute("""
    CREATE OR REPLACE TABLE dim_funds_relational (
        fund_id VARCHAR PRIMARY KEY,
        fund_name VARCHAR,
        jurisdiction VARCHAR,
        min_ticket_usd INTEGER,
        target_irr_pct DOUBLE
    );
    INSERT INTO dim_funds_relational VALUES
        ('FUND-01', 'Apollo European Credit IV', 'EMEA', 2500000, 11.5),
        ('FUND-02', 'Blackstone APAC Infra', 'APAC', 10000000, 14.0),
        ('FUND-03', 'EQT Global Secondaries', 'EMEA', 1000000, 13.2);
""")
console.print("[green]✔ Created Blob files, NoSQL JSONL documents, and Relational SQL table.[/green]\n")

# ==============================================================================
# PILLAR 2: CONVERTING UNSTRUCTURED DATA INTO VECTOR EMBEDDINGS
# ==============================================================================
console.print(Panel.fit("[bold cyan]2. Converting Unstructured Text into Vector Embeddings[/bold cyan]"))

fund_ids = ["FUND-01", "FUND-02", "FUND-03"]
corpus = [
    (blob_dir / "FUNDS_01.txt").read_text(),
    (blob_dir / "FUNDS_02.txt").read_text(),
    (blob_dir / "FUNDS_03.txt").read_text()
]

# Convert unstructured text into normalized float32 dense vectors
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(corpus).toarray().astype(np.float32)
embedding_dim = tfidf_matrix.shape[1]

console.print(f"• Unstructured chunks processed: [bold]{len(corpus)}[/bold]")
console.print(f"• Generated Vector Embedding Shape: [bold]({tfidf_matrix.shape[0]} chunks, {embedding_dim} dimensions)[/bold]")
console.print(f"• Sample Vector (FUND-01 first 6 floats): [yellow]{np.round(tfidf_matrix[0][:6], 4).tolist()}[/yellow]\n")

# ==============================================================================
# PILLAR 3: BUILDING VECTOR DATABASES & INDEXING ALGORITHMS
# ==============================================================================
console.print(Panel.fit("[bold cyan]3. Building Vector Database Table & Similarity Indexing[/bold cyan]"))

con.execute(f"""
    CREATE OR REPLACE TABLE vector_fund_chunks (
        fund_id VARCHAR,
        chunk_text VARCHAR,
        embedding FLOAT[{embedding_dim}]
    );
""")

for fid, text, vec in zip(fund_ids, corpus, tfidf_matrix):
    con.execute("INSERT INTO vector_fund_chunks VALUES (?, ?, ?)", [fid, text, vec.tolist()])

console.print("[green]✔ Stored dense vectors in `vector_fund_chunks` (FLOAT[N] vector column).[/green]")
console.print("• [bold]Specialized Indexing Algorithms:[/bold]")
console.print("  - [cyan]Exact Flat Search (KNN):[/cyan] O(N) brute-force scan; 100% recall, slow at scale.")
console.print("  - [cyan]IVFFlat (Inverted File Index):[/cyan] Clusters vectors into centroids via k-means; searches nearest buckets.")
console.print("  - [cyan]HNSW (Hierarchical Navigable Small World):[/cyan] Multi-layer proximity graph; O(log N) sub-10ms search (Standard in pgvector & Azure AI Search).\n")

# ==============================================================================
# PILLAR 4: IMPLEMENTING LLMs WITH LARGE DATASETS & COMPLEX DATA MODELS (HYBRID RAG)
# ==============================================================================
console.print(Panel.fit("[bold cyan]4. Implementing LLMs over Complex Data Models (Hybrid SQL + Vector + NoSQL RAG)[/bold cyan]"))

user_query = "Which funds offer quarterly liquidity or redemptions for EMEA wealth clients?"
client_max_ticket = 5000000
client_jurisdiction = "EMEA"

# Step 4A: Embed the incoming user query into the exact same vector space
query_vec = vectorizer.transform([user_query]).toarray().astype(np.float32)[0].tolist()

# Step 4B: Execute a Hybrid Query joining Relational SQL + NoSQL JSONL + Vector Cosine Similarity
hybrid_results = con.execute(f"""
    SELECT 
        r.fund_id,
        r.fund_name,
        r.jurisdiction,
        r.min_ticket_usd,
        r.target_irr_pct,
        n.asset_class,
        list_cosine_similarity(v.embedding, ?::FLOAT[{embedding_dim}]) AS cosine_score,
        v.chunk_text
    FROM vector_fund_chunks v
    JOIN dim_funds_relational r ON v.fund_id = r.fund_id
    JOIN read_json_auto('data/nosql_fund_attributes.jsonl') n ON v.fund_id = n.fund_id
    WHERE r.jurisdiction = ?
      AND r.min_ticket_usd <= ?
    ORDER BY cosine_score DESC
""", [query_vec, client_jurisdiction, client_max_ticket]).fetchall()

table = Table(title=f"Hybrid Search Results (Filtered: Jurisdiction={client_jurisdiction}, Ticket<=${client_max_ticket:,})")
table.add_column("Fund ID", style="cyan")
table.add_column("Fund Name", style="white")
table.add_column("Asset Class (NoSQL)", style="magenta")
table.add_column("Min Ticket (SQL)", justify="right")
table.add_column("Cosine Similarity (Vector)", justify="right", style="bold green")

for row in hybrid_results:
    table.add_row(row[0], row[1], row[5], f"${row[3]:,}", f"{row[6]:.4f}")

console.print(table)

# Step 4C: Construct the Grounded Prompt for the LLM
top_hit = hybrid_results[0]
llm_prompt = f"""[SYSTEM: S64 Private Markets AI Copilot - Strict RBAC & Grounded Context]
User Query: "{user_query}"
Client Constraints Applied (SQL Pre-Filter): Jurisdiction={client_jurisdiction}, Max Ticket=${client_max_ticket:,}

Retrieved Structured & Unstructured Context:
1. {top_hit[1]} ({top_hit[0]}) | Asset Class: {top_hit[5]} | Target IRR: {top_hit[4]}% | Similarity: {top_hit[6]:.4f}
   Prospectus Extract: "{top_hit[7]}"
2. {hybrid_results[1][1]} ({hybrid_results[1][0]}) | Asset Class: {hybrid_results[1][5]} | Target IRR: {hybrid_results[1][4]}% | Similarity: {hybrid_results[1][6]:.4f}
   Prospectus Extract: "{hybrid_results[1][7]}"
"""
console.print("\n[bold yellow]Generated Grounded LLM Context Payload (Ready for Azure OpenAI / Vertex AI):[/bold yellow]")
console.print(Panel(llm_prompt, border_style="yellow"))
