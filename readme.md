## FinClarity: Finance News Retrieval and Storytelling
FinClarity is an end‑to‑end application for retrieving, organising and explaining Indian finance news. It ingests raw news headlines, cleans and vectorises them, groups stories by company, and then uses a combination of retrieval algorithms and a large language model (LLM) to produce friendly summaries and jargon explanations via a Streamlit web interface. The project also includes tools for evaluating different vectorisation schemes and retrieval strategies.
Project Components
This repository contains several Python modules and supporting scripts:
Module	Purpose
vectoriser.py	Performs data cleaning and tokenisation on a CSV of headlines, builds three types of headline embeddings (TF‑IDF, Word2Vec CBOW and Word2Vec Skip‑gram), compares their retrieval quality, and stores the best embedding vectors in a SQLite database for later use.
ensemble_cluster.py	Implements the EnsembleClusterRetriever, a hybrid retrieval engine that clusters TF‑IDF vectors, routes queries to the most relevant clusters and ranks documents by cosine similarity. It also provides evaluation metrics such as precision@k, MAP and nDCG.
rag_streamlit_app.py	The Streamlit front‑end. It loads the cleaned dataset, applies weak labelling to associate headlines with companies, matches user queries to the correct company (using NER, alias matching and fuzzy logic), ranks the company’s headlines, builds prompts and calls a Groq LLM to generate human‑readable stories and jargon definitions. It exposes a simple user interface where people can ask about any company in natural language.
vector_eval.py	A standalone evaluation script that streamlines the comparison of TF‑IDF, CBOW and Skip‑gram embeddings across retrieval and classification tasks. It can compute macro P@5, MAP@10 and nDCG@10 for each model, produce bar charts and PCA plots, and optionally train a simple classifier to predict companies.
pptx tools	The answer.js and slides_template.js files demonstrate how to use pptxgenjs and PrismJS to build rich PowerPoint presentations programmatically. They are used in the context of this assignment to generate slide decks and are not part of the FinClarity pipeline.
Data Preprocessing and Vectorisation
The workflow begins by loading a CSV file (default: sensex_finance_news_daily.csv) containing a headline column. The scripts perform the following steps:
1.Cleaning & Tokenisation: Headlines are converted to lowercase, non‑alphabetic characters are removed, and the text is tokenised. Common English stop words and very short tokens are filtered out.
2.	TF‑IDF Embeddings: The TfidfVectorizer from scikit‑learn transforms the cleaned headlines into sparse vectors. These vectors capture the importance of words relative to the corpus.
3.	Word2Vec Embeddings: Two gensim Word2Vec models are trained on the tokenised headlines: one using the Continuous Bag‑of‑Words (CBOW) architecture and the other using the Skip‑gram architecture. Headline embeddings are obtained by averaging the word vectors.
4.	Evaluation & Storage: Retrieval quality is compared across TF‑IDF, CBOW and Skip‑gram using semantic search on example queries. The script reports which model produces the most coherent results and stores the skip‑gram vectors in a SQLite database (news_vectors.db) along with their corresponding headlines. A helper function demonstrates how to query this database for similar headlines.
Retrieval Engine

The core retrieval algorithm is implemented in ensemble_cluster.py. It defines an EnsembleClusterRetriever class that:
1.	Vectorises and Clusters: Fits a TF‑IDF vectoriser on the documents and partitions the vector space using K‑means (or MiniBatchKMeans for large corpora). Centroids are normalised for cosine similarity.
2.	Routes Queries: For a given query, the retriever computes its TF‑IDF vector, identifies the most similar clusters via centroid similarity, and assigns softmax weights to clusters according to their relevance.
3.	Ranks Documents: It then computes cosine similarities between the query and all documents within the selected clusters, multiplies them by the cluster weights and returns the top‑ranking documents along with their metadata.
4.	Evaluation: Built‑in methods compute precision@k, average precision and nDCG across grouped labels, enabling objective comparison of retrieval performance.
Streamlit Application

The rag_streamlit_app.py file launches a web app that allows users to ask questions such as “What happened to Infosys this week?” and receive narrative answers. Key features include:
•	Data Loading & Weak Labelling: The app reads the CSV, drops missing or duplicate headlines, and applies a simple keyword‑based function to assign each headline to a company. Documents labelled Unknown are discarded. Optionally, it fits an EnsembleClusterRetriever for improved retrieval.
•	Organisation Recognition: When a user submits a query, the app uses a BERT‑based NER pipeline (from Hugging Face Transformers) to extract organisation names. It also scans for known aliases (e.g., “RIL” → “Reliance Industries”) and applies edit‑distance and fuzzy matching to map the extracted names to canonical company names.
•	Ranking Headlines: Once a company is identified, the app performs a second stage of ranking within that company using a simple TF‑IDF similarity measure to select the most relevant headlines.
•	Prompt Building & LLM Integration: The top headlines are formatted into a prompt instructing a large language model (via the Groq chat API) to generate a succinct story explaining what happened, why it matters and clarifying any technical jargon. A separate prompt builder is used to explain individual financial terms.
•	Streamlit UI: The interface allows users to select or upload a dataset, adjust retrieval parameters, view the matched headlines and their scores, and read the generated stories. Setting the environment variable GROQ_API_KEY is required to call the Groq API. To run the app locally, execute:

pip install -r requirements.txt  # ensure Python dependencies are installed
streamlit run rag_streamlit_app.py
Evaluation Script
vector_eval.py provides a command‑line tool for systematically comparing different vectorisation schemes. It supports a range of configurable parameters (feature dimensionality, Word2Vec window size and epochs, etc.), computes macro retrieval metrics across companies, generates bar charts and PCA scatter plots and optionally trains a logistic‑regression classifier to predict the company of a headline. Example usage:
python vector_eval.py \
  --csv sensex_finance_news_daily.csv \
  --save_dir out_eval \
  --max_features 5000 --w2v_dim 100 --w2v_epochs 8 --topk 10 \
  --do_clf
Outputs are saved under the specified save_dir and include metrics tables (CSV), bar charts for P@5, MAP@10 and nDCG@10, PCA visualisations of the embedding spaces and F1 scores for the classifier.

Dependencies:
The Python components depend on the following packages (installable via pip):
•	pandas, numpy, scikit‑learn, gensim, nltk — for data handling, machine learning and embedding training.
•	rapidfuzz — for fuzzy string matching in company identification.
•	transformers, torch — to run the BERT‑based NER pipeline. Warnings from tokenisers are suppressed in the app.
•	streamlit — to create the web interface.
•	requests — to call the Groq chat API.

The environment also includes front‑end tooling (Tailwind CSS, PrismJS, PptxGenJS) in the node_modules folder. These are used for slide generation and are not essential for FinClarity’s core functionality.
Running the Application
1.	Prepare the Dataset: Place your CSV file with a headline column (e.g., sensex_finance_news_daily.csv) in the project directory.
2.	Install Python Dependencies: Run pip install -r requirements.txt or manually install the packages listed above. Ensure you have Python 3.8+.
3.	Set API Key: Obtain a Groq API key and set it as an environment variable:
 	export GROQ_API_KEY="your_api_key_here"
4.	Start the Web App: Execute streamlit run rag_streamlit_app.py. The app will load the data, display options to adjust retrieval parameters and allow you to query any
company. For jargon explanations, click the “Explain Jargon” button and enter a term.

Contributing
Contributions are welcome! Feel free to open issues or pull requests to enhance the retrieval engine, improve evaluation metrics, add new data sources or refine the user interface. Please ensure that any additional dependencies are listed in the project documentation.
License
This project is provided for educational purposes. See the repository for licensing details.
________________________________________
