from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core import set_global_tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from bert_score import score as bert_score
import json
import nltk
import torch
import re

nltk.download("punkt")
nltk.download('punkt_tab')

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

def extract_boxed_text(text):
    match = re.search(r'\*\*Answer:\*\*\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    return match.group(1).strip() if match else None

def compare(list1,list2):
  
  if list1[-1] == list2[-1]:
    return False
  else:
    return True

def metrics(a, domain, list1, list5):
    def calculate_bleu(reference, hypothesis):
        reference = [nltk.word_tokenize(ref) for ref in reference]
        hypothesis = nltk.word_tokenize(hypothesis)
        bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, hypothesis)
        return bleu_score

    def calculate_f1_score(reference, hypothesis):
        reference_tokens = set(nltk.word_tokenize(reference))
        hypothesis_tokens = set(nltk.word_tokenize(hypothesis))
        if len(hypothesis_tokens) == 0:
            return 0
        precision = len(reference_tokens.intersection(hypothesis_tokens)) / len(hypothesis_tokens)
        recall = len(reference_tokens.intersection(hypothesis_tokens)) / len(reference_tokens)
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
    
    def calculate_pass_percentage(results, list1):
      
      total = len(list1)
      pass_count = total -  len(results)
      pass_percentage = (pass_count / total) * 100 if total > 0 else 0
      return pass_percentage

    def calculate_hallucination_rate(reference, hypothesis):
        reference_tokens = set(nltk.word_tokenize(reference))
        hallucination_tokens = set(nltk.word_tokenize(hypothesis)) - reference_tokens
        if len(set(nltk.word_tokenize(hypothesis))) == 0:
            return 0
        hallucination_rate = len(hallucination_tokens) / len(set(nltk.word_tokenize(hypothesis)))
        return hallucination_rate

    def evaluate_metrics(reference, hypothesis):
        if hypothesis == None:
            return "pass"
        if hypothesis.lower() == "pass":
            return "pass"
        _, _, bert_score_value = bert_score([hypothesis], [reference], lang="en")
        bleu_score = calculate_bleu([reference], hypothesis)
        f1_score = calculate_f1_score(reference, hypothesis)
        hallucination_rate = calculate_hallucination_rate(reference, hypothesis)
        return {
            'BERTScore': bert_score_value.mean().item(),
            'BLEU': bleu_score,
            'F1 Score': f1_score,
            'Hallucination Rate': hallucination_rate
        }
    
    def calculate_average_metrics(results):
      
      if not results:
          return None

      total_metrics = len(results)
      average_metrics = {
          'BERTScore': sum(result['BERTScore'] for result in results) / total_metrics,
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

    with open(f"Naive_RAG_DeepSeek_Results_{domain}.txt",'a') as f:
      f.write(f"\n Prompting: {a} \n Pass Percentage: {pass_percentage:.2f}% \n Average Metrics: {average_metrics}")

    print("Average Metrics:", evaluation_results)
    return

def func_none(r):
    if r == None:
        return "pass"

def generate_response(prompt):
    messages = [{ 'role': 'user', 'content': prompt }]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=300, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

def train():
    documents = SimpleDirectoryReader("/home/ysaxena1/manas_ada/users/ysaxena1/Legal_Work_Yash/Data").load_data()
    set_global_tokenizer(AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B").encode)
    Settings.embed_model = resolve_embed_model("local:BAAI/bge-small-en")
    index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
    return index

def prompting(file_name, d):
    list1, list5, list2, list3 = [], [], [], []
    index = train()
    Settings.llm = None
    query_engine = index.as_query_engine()
    with open(file_name, 'r') as f:
        data = json.load(f)
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    list5.append(paper.split(":", 1)[1].strip())
                    retrieved_docs = query_engine.query(user_content)
                    # print(type(retrieved_docs))
                    # print(retrieved_docs)
                    context = retrieved_docs.response[:500] if hasattr(retrieved_docs, 'response') else ""
                    prompt = f"I have taken a sentence from the research paper titled \"{paper_title}\". Using the retrieved context, determine the title of the possible research paper that this sentence is citing to. If unsure, write 'pass'.\n\nContext: {context}\n\nSentence: \"{user_content}\""
                    r = generate_response(prompt)
                    list1.append(r)
                    print(list1, "\n", list5)
        metrics("Zero-Shot Indirect",d,list1,list5)
        list1.clear()
        list5.clear()
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    list5.append(author.split(":", 1)[1].strip())
                    retrieved_docs = query_engine.query(user_content)
                    context = retrieved_docs.response[:500] if hasattr(retrieved_docs, 'response') else ""
                    prompt = f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'."
                    r = generate_response(prompt)
                    list1.append(r)
                    print(list1,"\n", list5)
        metrics("Zero-Shot Direct",d,list1,list5)
        list1.clear()
        list5.clear()
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    list5.append(author.split(":", 1)[1].strip())
                    retrieved_docs = query_engine.query(user_content)
                    context = retrieved_docs.response[:500] if hasattr(retrieved_docs, 'response') else ""
                    prompt = f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'."
                    r = generate_response(prompt)
                    r = func_none(r)
                    list2.append(r)
                    if r.lower()=='pass' or compare(list2,list5):
                      retrieved_docs = query_engine.query(user_content)
                      context = retrieved_docs.response[:500] if hasattr(retrieved_docs, 'response') else ""
                      r1 = generate_response(f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'. Let me give you some more context by providing the abstract of the research paper. {abstract}").response
                      list1.append(r1)
                    else:
                      list1.append(r)
                    print(list1,"\n", list5)
        metrics("Direct with Metadata",d,list1,list5)
        list1.clear()
        list2.clear()
        list5.clear()
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    list5.append(paper.split(":", 1)[1].strip())
                    retrieved_docs = query_engine.query(user_content)
                    context = retrieved_docs.response[:500] if hasattr(retrieved_docs, 'response') else ""
                    r = generate_response(f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\"").response
                    r = func_none(r)
                    list2.append(r)
                    if r.lower()=='pass' or compare(list2,list5):
                      retrieved_docs = query_engine.query(user_content)
                      context = retrieved_docs.response[:500] if hasattr(retrieved_docs, 'response') else ""
                      r1 = generate_response(f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}").response
                      r1 = func_none(r1)
                      list3.append(r1)
                      if r1.lower()=='pass' or compare(list3,list5):
                        retrieved_docs = query_engine.query(user_content)
                        context = retrieved_docs.response[:500] if hasattr(retrieved_docs, 'response') else ""
                        r2 = generate_response(f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}. Let me give you some more context by providing the author names of the research paper it is citing to. {author}").response
                        list1.append(r2)
                      else:
                        list1.append(r1)
                    else:
                      list1.append(r)
                    print(list1,"\n", list5)
        metrics("SID",d,list1,list5)
        list1.clear()
        list2.clear()
        list3.clear()
        list5.clear()
    return

def main():
  file_name = "/home/ysaxena1/manas_ada/users/ysaxena1/Legal_Work_Yash/Data/Database.json" # Give the file name/file path over here
  domain = 'Database' # specify the domain over here
  prompting(file_name,domain)

if __name__ == "__main__":
  main()