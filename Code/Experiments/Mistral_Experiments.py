import replicate
import nltk
import os
import json
from bert_score import score as bert_score

nltk.download("punkt")

def metrics(a,list1,list5, domain):
  
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

      # BERTScore
      _, _, bert_score_value = bert_score([hypothesis], [reference], lang="en")

      # BLEU Score
      bleu_score = calculate_bleu([reference], hypothesis)

      # F1 Score
      f1_score = calculate_f1_score(reference, hypothesis)

      # Hallucination Rate
      hallucination_rate = calculate_hallucination_rate(reference, hypothesis)

      return {
          'BERTScore': bert_score_value.mean().item(),
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

  with open(f"Normal_LLama_Results_{domain}.txt",'a') as f:
    f.write(f"\n Prompting: {a} \n Pass Percentage: {pass_percentage:.2f}% \n Average Metrics: {average_metrics}")

  print(f"Pass Percentage: {pass_percentage:.2f}%")
  print("Average Metrics:")
  print(average_metrics)
  return

def compare(list1,list2):
  
  if list1[-1] == list2[-1]:
    return False
  else:
    return True

def prompting(file_name, domain):
  
  list1 = []
  list2 = []
  list3 = []
  list5 = []

  os.environ["REPLICATE_API_TOKEN"] = "ENTER_REPLICATE_API_KEY"

  with open(file_name, 'r') as f:
        data = json.load(f)
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
                    output = replicate.run(
                        "mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
                        input={
                            "prompt": f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to, also provide the explanation as to why you think the sentence is citing to the paper title provided by you, while providing the explanation don't mention the sentence just take key concepts from the sentence try to map it with the abstract of the paper that you have given. Also, format the output as <Paper Title>\n<Explanation>. If you are not able to come up with the paper title write 'pass'. Don't write anything else. Sentence: \"{user_content}\""
                        }
                    )
                    answer = ''
                    for text in output:
                      answer += text
                    list1.append(answer)
                    print(list1,"\n", list5)
        metrics("Zero-Shot Indirect",list1,list5, domain)
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
                    list5.append(paper.split(":", 1)[1].strip())
                    output = replicate.run(
                        "mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
                        input={
                            "prompt": f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'."
                        }
                    )
                    answer = ''
                    for text in output:
                      answer += text
                    list1.append(answer)
                    print(list1,"\n", list5)
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
                    list5.append(paper.split(":", 1)[1].strip())
                    output = replicate.run(
                        "mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
                        input={
                            "prompt": f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'."
                        }
                    )
                    answer = ''
                    for text in output:
                      answer += text
                    list2.append(answer)
                    if answer.lower()=='pass' or compare(list2,list5):
                      output = replicate.run(
                          "mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
                          input={
                              "prompt": f"Who were the authors of the research paper \"{paper}\". List only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'. Let me give you some more context by providing the abstract of the research paper. {abstract}"
                          }
                      )
                      answer1 = ''
                      for text in output:
                        answer1 += text
                      list1.append(answer1)
                    else:
                      list1.append(answer)
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
                    output = replicate.run(
                        "mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
                        input={
                            "prompt": f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\""
                        }
                    )
                    answer = ''
                    for text in output:
                      answer += text
                    list2.append(answer)
                    if answer.lower()=='pass' or compare(list2,list5):
                      output = replicate.run(
                          "mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
                          input={
                              "prompt": f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}"
                          }
                      )
                      answer1 = ''
                      for text in output:
                        answer1 += text
                      list3.append(answer1)
                      if answer.lower()=='pass' or compare(list3,list5):
                        output = replicate.run(
                            "mistralai/mistral-7b-instruct-v0.2:79052a3adbba8116ebc6697dcba67ad0d58feff23e7aeb2f103fc9aa545f9269",
                            input={
                                "prompt": f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else. \"{user_content}\". Let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}. Let me give you some more context by providing the author names of the research paper it is citing to. {author}"
                            }
                        )
                        answer2 = ''
                        for text in output:
                          answer2 += text
                        list1.append(answer2)
                      else:
                        list1.append(answer1)
                    else:
                      list1.append(answer)
                    print(list1,"\n", list5)
        metrics("SID",list1,list5, domain)
        list1.clear()
        list2.clear()
        list3.clear()
        list5.clear()
  return

def main():
  file_name = "ENTER_FILE_PATH" # Give the path of the file
  domain = "ENTER_DOMAIN_NAME" # Write the domain of the file
  prompting(file_name, domain)

if __name__ == "__main__":
  main()