from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from difflib import SequenceMatcher as SM
from llama_index.llms.replicate import Replicate
import os
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer
import json
import nltk

nltk.download("punkt")

def metrics(a,d,list1,list5):
  
  def calculate_bleu(reference, hypothesis):
      reference = [nltk.word_tokenize(ref) for ref in reference]
      hypothesis = nltk.word_tokenize(hypothesis)

      # Calculate BLEU score
      bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)
      return bleu_score

  def calculate_f1_score(reference, hypothesis):
      reference_tokens = set(nltk.word_tokenize(reference))
      hypothesis_tokens = set(nltk.word_tokenize(hypothesis))

      # Calculate precision, recall, and F1 score
      if len(hypothesis_tokens) == 0:
        return 0
      precision = len(reference_tokens.intersection(hypothesis_tokens)) / len(hypothesis_tokens)
      recall = len(reference_tokens.intersection(hypothesis_tokens)) / len(reference_tokens)

      f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
      return f1_score

  def calculate_hallucination_rate(reference, hypothesis):
      reference_tokens = set(nltk.word_tokenize(reference))
      hallucination_tokens = set(nltk.word_tokenize(hypothesis)) - reference_tokens
      if len(set(nltk.word_tokenize(hypothesis))) == 0:
        return 0

      hallucination_rate = len(hallucination_tokens) / len(set(nltk.word_tokenize(hypothesis)))
      return hallucination_rate

  def evaluate_metrics(reference, hypothesis):
      if hypothesis.lower() == "pass":
          return "pass"  # Skip evaluation for "pass"

      # BLEU Score
      bleu_score = calculate_bleu([reference], hypothesis)

      # F1 Score
      f1_score = calculate_f1_score(reference, hypothesis)

      # Hallucination Rate
      hallucination_rate = calculate_hallucination_rate(reference, hypothesis)

      return {
          'BLEU': bleu_score,
          'F1 Score': f1_score,
          'Hallucination Rate': hallucination_rate
      }

  def calculate_pass_percentage(results, list1):
      total = len(list1)
      pass_count = total -  len(results)
      pass_percentage = (pass_count / total) * 100 if total > 0 else 0
      return pass_percentage

  def calculate_average_metrics(results):
      if not results:
          return None

      total_metrics = len(results)
      average_metrics = {
          'BLEU': sum(result['BLEU'] for result in results) / total_metrics,
          'F1 Score': sum(result['F1 Score'] for result in results) / total_metrics,
          'Hallucination Rate': sum(result['Hallucination Rate'] for result in results) / total_metrics
      }
      return average_metrics

  evaluation_results = []
  for generated_text, reference_text in zip(list1, list5):
      result = evaluate_metrics(reference_text, generated_text)
      if result != "pass":
          evaluation_results.append(result)

  pass_percentage = calculate_pass_percentage(evaluation_results, list1)
  average_metrics = calculate_average_metrics(evaluation_results)

  with open(f"Adversarial_Results_{d}_{a}.txt",'a') as f:
    f.write(f"\n Prompting: {a} \n Pass Percentage: {pass_percentage:.2f}% \n Average Metrics: {average_metrics}")

  print(f"Pass Percentage: {pass_percentage:.2f}%")
  print("Average Metrics:")
  print(average_metrics)
  return

def train():
    documents = SimpleDirectoryReader("ENTER_DIRECTORY_PATH").load_data() # give the path of the directory which contains all the files of the dataset
    os.environ["REPLICATE_API_TOKEN"] = "ENTER_REPLICATE_API_KEY"

    # Using replicate's llama-2-7b
    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    llm = Replicate(
        model=llama2_7b_chat,
        temperature=1.0,
        additional_kwargs={"top_p": 0.95, "max_new_tokens": 300},
    )

    set_global_tokenizer(
        AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf").encode
    )

    # Defining the embeding model
    embed_model = resolve_embed_model("local:BAAI/bge-small-en")
    service_context = ServiceContext.from_defaults(
        chunk_size=512,
        llm=llm,
        embed_model=embed_model
    )
    
    # Defining the cross-encoder for re-ranking
    rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=10 
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)

    return [index, rerank]

def compare(list1,list2):
  if list1[-1] == list2[-1]:
    return False
  else:
    return True

# Function to find most similar paper title
def find_most_similar_title(str1, titles):
    max_ratio = 0
    most_similar_title = None
    for title in titles:
        ratio = SM(None, str1, title).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            most_similar_title = title
    print(max_ratio)
    return most_similar_title

# Function to find most similar paper abstract
def find_most_similar_abstract(str1, abstracts):
    max_ratio = 0
    most_similar_abstract = None
    for abstract in abstracts:
        ratio = SM(None, str1, abstract).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            most_similar_abstract = abstract
    print(max_ratio)
    return most_similar_abstract

def prompting(file_name,d):
    list1 = []
    list2 = []
    list3 = []
    list5 = []
    t = train()
    index = t[0]
    rerank = t[1]
    query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank]) # Note we are first selecting 10 chunks.
    with open(file_name, 'r') as f:
        data = json.load(f)
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]

                # Find most similar paper title
                other_titles = [paper_info["Paper Title"] for paper_info in papers.values() if paper_info["Paper Title"] != paper_title]
                most_similar_title = find_most_similar_title(paper_title, other_titles)

                print(f"Paper Title: {paper_title}")
                print(f"Most Similar Title: {most_similar_title}")

                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    list5.append(paper.split(":", 1)[1].strip())
                    r = query_engine.query(f"I have taken a sentence from the research paper titled \"{most_similar_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\"").response
                    list2.append(r)
                    if r.lower()=='pass' or compare(list2,list5):
                        r1 = query_engine.query(f"I have taken a sentence from the research paper titled \"{most_similar_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}").response
                        list3.append(r1)
                    if r.lower()=='pass' or compare(list3,list5):
                        r2 = query_engine.query(f"I have taken a sentence from the research paper titled \"{most_similar_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}. Let me give you some more context by providing the author names of the research paper it is citing to. {author}").response
                        list1.append(r2)
                    else:
                        list1.append(r1)
                else:
                    list1.append(r)
                print(list1,"\n", list5)
        metrics("SID_Paper_Title","Advanced_RAG_LLAMA2",list1,list5)
        list1.clear()
        list2.clear()
        list3.clear()
        list5.clear()

        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    paper_abstract = sentence_info["Citation"]["Citation Paper Abstract"]

                    # Find most similar paper abstract
                    other_abstracts = [sentence["Citation"]["Citation Paper Abstract"] for sentence in sentences if sentence["Citation"]["Citation Paper Abstract"] != paper_abstract]
                    most_similar_abstract = find_most_similar_abstract(paper_abstract, other_abstracts)

                    print(f"Paper Title: {paper_title}")
                    print(f"Original Abstract: {paper_abstract}")
                    print(f"Most Similar Abstract: {most_similar_abstract}")

                    list5.append(paper.split(":", 1)[1].strip())
                    r = query_engine.query(f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\"").response
                    list2.append(r)
                    if r.lower()=='pass' or compare(list2,list5):
                        r1 = query_engine.query(f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {most_similar_abstract}").response
                        list3.append(r1)
                    if r.lower()=='pass' or compare(list3,list5):
                        r2 = query_engine.query(f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {most_similar_abstract}. Let me give you some more context by providing the author names of the research paper it is citing to. {author}").response
                        list1.append(r2)
                    else:
                        list1.append(r1)
                else:
                    list1.append(r)
                print(list1,"\n", list5)
        metrics("SID_Abstract","Advanced_RAG_LLAMA2",list1,list5)
    return

def main():
  file_name = "ENTER_FILE_PATH" # Give the file path over here
  domain = 'ENTER_DOMAIN_NAME' # specify the domain over here
  prompting(file_name,domain)

if __name__ == "__main__":
  main()
