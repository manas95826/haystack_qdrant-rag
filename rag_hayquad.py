# conda create - name haystack-env
# conda activate haystack-env


# pip install --upgrade pip
# pip install 'farm-haystack'

# pip install qdrant-haystack

# docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
# Crawler Instance

from haystack.nodes import Crawler
url = "https://opendoorsdata.org/annual-release/u-s-study-abroad/l"
crawler = Crawler(output_dir="data/crawled_gcp", crawler_depth=1)
crawled_docs = crawler.crawl(urls=[url])

# Metadata

import os
json_files = []
for crawled_doc in crawled_docs:
   # Extract name from file path
   name = os.path.splitext(os.path.basename(str(crawled_doc)))[0]
   # Load file and add 'name' key-value pair to 'meta'
   with open(crawled_doc) as f:
       file_data = json.load(f)
       file_data['meta']['name'] = name
       json_files.append(file_data)


from haystack.nodes import PreProcessor
preprocessor = PreProcessor(
   clean_empty_lines=True,
   clean_whitespace=True,
   clean_header_footer=False,
   split_by="word",
   split_length=100,
   split_respect_sentence_boundary=True
)

docs = preprocessor.process(json_files)


# If Cloud Instance


from qdrant_haystack.document_stores import QdrantDocumentStore
document_store = QdrantDocumentStore(host="your-qdrant-instance-prefix.cloud.qdrant.io",
                                    api_key="your_api_key",
                                    https=True,
                                    index="opendoorsdata",
                                    embedding_dim=768,
                                    recreate_index=True,
                                    timeout=120,
                                    grpc_port=6334,
                                    prefer_grpc=True )
document_store.write_documents(docs)


# If Local Instance

from qdrant_haystack.document_stores import QdrantDocumentStore
document_store = QdrantDocumentStore(host="localhost",
                                    index="opendoorsdata",
                                    embedding_dim=768,
                                    recreate_index=True,
                                    timeout=120
                                   )
document_store.write_documents(docs)



from haystack.nodes import EmbeddingRetriever
retriever = EmbeddingRetriever(
   document_store=document_store,
   embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
   model_format="sentence_transformers",
)
document_store.update_embeddings(retriever)

# Querying Your Data

from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
reader = FARMReader(model_name_or_path="meta-llama/Meta-Llama-3-8B", use_gpu=True)
pipe = ExtractiveQAPipeline(reader, retriever)


prediction = pipe.run(
   query="How to apply for a college in Canada",
   params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
)
print_answers(prediction, details="all")


