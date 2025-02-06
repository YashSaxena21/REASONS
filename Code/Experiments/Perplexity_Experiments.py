import requests
import json
import time
import nltk
from bert_score import score as bert_score

nltk.download("punkt")

# Function for calculating average metric score
def metrics(a,list1,list5):
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

  with open("Perplexity_Results.txt",'a') as f:
    f.write(f"\n Prompting: {a} \n Pass Percentage: {pass_percentage:.2f}% \n Average Metrics: {average_metrics}")

  print(f"Pass Percentage: {pass_percentage:.2f}%")
  print("Average Metrics:")
  print(average_metrics)
  return

# This function is being used in Few-Shot Direct Prompting and SID Prompting 
def compare(list1,list2):
  if list1[-1] == list2[-1]:
    return False
  else:
    return True

# Function for conducting all the prompting techniques
def prompting(file_name):
    list1 = []
    list2 = []
    list3 = []
    list5 = []
    url = "https://api.perplexity.ai/chat/completions"

    with open(file_name, 'r') as f:
        data = json.load(f)

        # Zero-Shot Indirect Prompting
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    list5.append(paper.split(":", 1)[1].strip())
                    payload = {
                        "model": "pplx-7b",
                        "messages": [
                            {
                            "role": "system",
                            "content": (
                              f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                user_content
                            ),
                        },
                        ],
                        "max_tokens": 0,
                        "temperature": 1,
                        "top_p": 1,
                        "top_k": 0,
                        "stream": False,
                        "presence_penalty": 0,
                        "frequency_penalty": 1
                    }
                    headers = {
                        "accept": "application/json",
                        "content-type": "application/json",
                        "authorization": "ENTER_PERPLEXITY_TOKEN"
                    }
                    response = requests.post(url, json=payload, headers=headers)
                    time.sleep(1)
                    response_data = response.json()
                    response_content = response_data["choices"][0]["message"]["content"]
                    list1.append(response_content)
                    print(list1,"\n", list5)
        metrics("Zero-Shot Indirect",list1,list5)
        list1.clear()
        list5.clear()

        # Zero-Shot Direct Prompting
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    list5.append(author.split(":", 1)[1].strip())
                    payload = {
                        "model": "pplx-7b",
                        "messages": [
                            {
                            "role": "system",
                            "content": (
                              f"Who were the authors of the research paper \"{user_content}\""
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "list only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'."
                            ),
                        },
                        ],
                        "max_tokens": 0,
                        "temperature": 1,
                        "top_p": 1,
                        "top_k": 0,
                        "stream": False,
                        "presence_penalty": 0,
                        "frequency_penalty": 1
                    }
                    headers = {
                        "accept": "application/json",
                        "content-type": "application/json",
                        "authorization": "ENTER_PERPLEXITY_TOKEN"
                    }
                    response = requests.post(url, json=payload, headers=headers)
                    time.sleep(1)
                    response_data = response.json()
                    response_content = response_data["choices"][0]["message"]["content"]
                    list1.append(response_content)
                    print(list1,"\n", list5)
        metrics("Zero-Shot Direct",list1,list5)
        list1.clear()
        list5.clear()

        # Few-Shot Direct Prompting
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Citation"]["Citation Paper Title"]
                    paper = sentence_info["Citation"]["Citation Paper Authors"]
                    abstract = sentence_info["Citation"]["Citation Paper Authors"]
                    list5.append(paper.split(":", 1)[1].strip())
                    payload = {
                        "model": "pplx-7b",
                        "messages": [
                            {
                            "role": "system",
                            "content": (
                              f"Who were the authors of the research paper \"{user_content}\""
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "list only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'."
                            ),
                        },
                        ],
                        "max_tokens": 0,
                        "temperature": 1,
                        "top_p": 1,
                        "top_k": 0,
                        "stream": False,
                        "presence_penalty": 0,
                        "frequency_penalty": 1
                    }
                    headers = {
                        "accept": "application/json",
                        "content-type": "application/json",
                        "authorization": "ENTER_PERPLEXITY_TOKEN"
                    }
                    response = requests.post(url, json=payload, headers=headers)
                    time.sleep(1)
                    response_data = response.json()
                    assistant_response = response_data["choices"][0]["message"]["content"]
                    list2.append(assistant_response)
                    if assistant_response == 'pass' or compare(list2,list5):
                        payload = {
                          "model": "pplx-7b",
                          "messages": [
                              {
                              "role": "system",
                              "content": (
                                f"Who were the authors of the research paper \"{user_content}\""
                              ),
                          },
                          {
                              "role": "user",
                              "content": (
                                  "list only author names, formatted as <first name><last name>, separated by comma. Do not mention the paper in the title, also if you don't know write 'pass'."
                              ),
                          },
                          {
                              "role": "user",
                              "content": (
                                  f"let me give you some more context by providing the abstract of the research paper. {abstract}"
                              ),
                          },
                          ],
                          "max_tokens": 0,
                          "temperature": 1,
                          "top_p": 1,
                          "top_k": 0,
                          "stream": False,
                          "presence_penalty": 0,
                          "frequency_penalty": 1
                        }
                        headers = {
                          "accept": "application/json",
                          "content-type": "application/json",
                          "authorization": "ENTER_PERPLEXITY_TOKEN"
                        }
                        r1 = requests.post(url, json=payload, headers=headers)
                        time.sleep(1)
                        r1_data = r1.json()
                        assistant_r1 = r1_data["choices"][0]["message"]["content"]
                        list1.append(assistant_r1)
                    else:
                      list1.append(assistant_response)
                    print(list1,"\n", list5)
        metrics("Direct with Metadata",list1,list5)
        list1.clear()
        list2.clear()
        list5.clear()

        # SID Prompting
        for domain, papers in data.items():
            for paper_url, info in papers.items():
                paper_title = info["Paper Title"]
                sentences = info["Sentences"]
                for sentence_info in sentences:
                    user_content = sentence_info["Sentence"]
                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    list5.append(paper.split(":", 1)[1].strip())
                    payload = {
                        "model": "pplx-7b",
                        "messages": [
                            {
                            "role": "system",
                            "content": (
                              f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                user_content
                            ),
                        },
                        ],
                        "max_tokens": 0,
                        "temperature": 1,
                        "top_p": 1,
                        "top_k": 0,
                        "stream": False,
                        "presence_penalty": 0,
                        "frequency_penalty": 1
                    }
                    headers = {
                        "accept": "application/json",
                        "content-type": "application/json",
                        "authorization": "ENTER_PERPLEXITY_TOKEN"
                    }
                    response = requests.post(url, json=payload, headers=headers)
                    time.sleep(1)
                    response_data = response.json()
                    assistant_response = response_data["choices"][0]["message"]["content"]
                    list2.append(assistant_response)
                    if assistant_response == 'pass' or compare(list2,list5):
                        payload = {
                          "model": "pplx-7b",
                          "messages": [
                              {
                              "role": "system",
                              "content": (
                                f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else."
                              ),
                          },
                          {
                              "role": "user",
                              "content": (
                                  user_content
                              ),
                          },
                          {
                              "role": "user",
                              "content": (
                                  f"let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}"
                              ),
                          },
                          ],
                          "max_tokens": 0,
                          "temperature": 1,
                          "top_p": 1,
                          "top_k": 0,
                          "stream": False,
                          "presence_penalty": 0,
                          "frequency_penalty": 1
                        }
                        headers = {
                          "accept": "application/json",
                          "content-type": "application/json",
                          "authorization": "ENTER_PERPLEXITY_TOKEN"
                        }
                        r1 = requests.post(url, json=payload, headers=headers)
                        time.sleep(1)
                        r1_data = r1.json()
                        assistant_r1 = r1_data["choices"][0]["message"]["content"]
                        list3.append(assistant_r1)
                        if assistant_r1 == 'pass' or compare(list3,list5):
                            payload = {
                            "model": "pplx-7b",
                            "messages": [
                                {
                                "role": "system",
                                "content": (
                                  f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    user_content
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}"
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"let me give you some more context by providing the author names of the research paper it is citing to. {author}"
                                ),
                            },
                            ],
                            "max_tokens": 0,
                            "temperature": 1,
                            "top_p": 1,
                            "top_k": 0,
                            "stream": False,
                            "presence_penalty": 0,
                            "frequency_penalty": 1
                            }
                            headers = {
                            "accept": "application/json",
                            "content-type": "application/json",
                            "authorization": "ENTER_PERPLEXITY_TOKEN"
                            }
                            r2 = requests.post(url, json=payload, headers=headers)
                            time.sleep(1)
                            r2_data = r2.json()
                            assistant_r2 = r2_data["choices"][0]["message"]["content"]
                            list1.append(assistant_r2)
                        else:
                         list1.append(assistant_r1)
                    else:
                      list1.append(assistant_response)
                    print(list1,"\n", list5)
        metrics("SID",list1,list5)
        list1.clear()
        list2.clear()
        list3.clear()
        list5.clear()
        return

def main():
  file_name = "ENTER_FILE_PATH"
  prompting(file_name)

if __name__ == "__main__":
  main()