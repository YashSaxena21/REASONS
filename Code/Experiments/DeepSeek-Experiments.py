import nltk
import os
import json
import re
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

def func_none(r):
    if r == None:
        return "pass"

def compare(list1,list2):
  
  if list1[-1] == list2[-1]:
    return False
  else:
    return True

def metrics(a, list1, list5, domain):
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

    with open(f"Normal_DeepSeek_Results_{domain}.txt",'a') as f:
      f.write(f"\n Prompting: {a} \n Pass Percentage: {pass_percentage:.2f}% \n Average Metrics: {average_metrics}")

    print("Average Metrics:", evaluation_results)
    return

def prompting(file_name, domain):
    list1 = []
    list5 = []
    list2 = []
    list3 = []

    with open(file_name, 'r') as f:
        data = json.load(f)
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    list5.append(paper.split(":", 1)[1].strip())
                    messages=[
                        { 'role': 'user', 'content': f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. Give the final answer after Answer:.If you are not able to come up with the paper title write 'pass'. Don't write anything else. Sentence: \"{user_content}\"" }
                    ]
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(inputs, max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    extracted_text = extract_boxed_text(generated_text)
                    list1.append(extracted_text)
                    print(list1, "\n", list5)
        metrics("Zero-Shot Indirect", list1, list5, domain)
        list1.clear()
        list5.clear()
    
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    list5.append(author.split(":", 1)[1].strip())
                    messages=[
                        { 'role': 'user', 'content': f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Please reason step by step, and put your final answer within \boxed().Do not mention the paper in the title, also if you don't know write 'pass'" }
                    ]
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(inputs, max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                    extracted_text = extract_boxed_text(tokenizer.decode(outputs[0][len(inputs[0]):]))
                    list1.append(extracted_text)
                    print(list1, "\n", list5)
        metrics("Zero-Shot Direct",list1,list5, domain)
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
                    messages=[
                        { 'role': 'user', 'content': f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Please reason step by step, and put your final answer within \boxed().Do not mention the paper in the title, also if you don't know write 'pass'" }
                    ]
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(inputs, max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                    extracted_text = extract_boxed_text(tokenizer.decode(outputs[0][len(inputs[0]):]))
                    extracted_text = func_none(extracted_text)
                    list2.append(extracted_text)
                    if extracted_text.lower()=='pass' or compare(list2,list5):
                        messages1=[
                            { 'role': 'user', 'content': f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Please reason step by step, and put your final answer within \boxed().Do not mention the paper in the title, also if you don't know write 'pass'. Let me give you some more context by providing the abstract of the research paper. {abstract}" }
                        ]
                        inputs1 = tokenizer.apply_chat_template(messages1, add_generation_prompt=True, return_tensors="pt").to(model.device)
                        outputs1 = model.generate(inputs1, max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                        extracted_text1 = extract_boxed_text(tokenizer.decode(outputs1[0][len(inputs1[0]):]))
                        list1.append(extracted_text1)
                    else:
                      list1.append(extracted_text)
                    print(list1,"\n", list5)
        metrics("Direct with Metadata",list1,list5, domain)
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
                    messages=[
                        { 'role': 'user', 'content': f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. Please reason step by step, and put your final answer within \boxed().If you are not able to come up with the paper title write 'pass'. Don't write anything else. Sentence: \"{user_content}\"" }
                    ]
                    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    outputs = model.generate(inputs, max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                    extracted_text = extract_boxed_text(tokenizer.decode(outputs[0][len(inputs[0]):]))
                    extracted_text = func_none(extracted_text)
                    list2.append(extracted_text)
                    if extracted_text.lower()=='pass' or compare(list2,list5):
                      messages1=[
                        { 'role': 'user', 'content': f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. Please reason step by step, and put your final answer within \boxed().If you are not able to come up with the paper title write 'pass'. Don't write anything else. Sentence: \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}" }
                      ]
                      inputs1 = tokenizer.apply_chat_template(messages1, add_generation_prompt=True, return_tensors="pt").to(model.device)
                      outputs1 = model.generate(inputs1, max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                      extracted_text1 = extract_boxed_text(tokenizer.decode(outputs1[0][len(inputs1[0]):]))
                      extracted_text1 = func_none(extracted_text1)
                      list3.append(extracted_text1)
                      if extracted_text1.lower()=='pass' or compare(list3,list5):
                        messages2=[
                        { 'role': 'user', 'content': f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. Please reason step by step, and put your final answer within \boxed().If you are not able to come up with the paper title write 'pass'. Don't write anything else. Sentence: \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}. Let me give you some more context by providing the author names of the research paper it is citing to. {author}" }
                        ]
                        inputs2 = tokenizer.apply_chat_template(messages2, add_generation_prompt=True, return_tensors="pt").to(model.device)
                        outputs2 = model.generate(inputs2, max_new_tokens=2000, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
                        extracted_text2 = extract_boxed_text(tokenizer.decode(outputs2[0][len(inputs2[0]):]))
                        list1.append(extracted_text2)
                      else:
                        list1.append(extracted_text1)
                    else:
                      list1.append(extracted_text)
                    print(list1,"\n", list5)
        metrics("SID",list1,list5, domain)
        list1.clear()
        list2.clear()
        list3.clear()
        list5.clear()
    return

def main():
    file_name = "/home/ysaxena1/manas_ada/users/ysaxena1/Legal_Work_Yash/Data/Database.json"
    domain = "Database"
    prompting(file_name, domain)

if __name__ == "__main__":
    main()
