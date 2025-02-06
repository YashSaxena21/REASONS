import nltk
from bert_score import score as bert_score
import openai
import json
from difflib import SequenceMatcher as SM

nltk.download("punkt")

# Function for calculating average metric score
def metrics(a, b, list1, list5):
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

  with open("OpenAI_Results.txt",'a') as f:
    f.write(f"\n Model Name: {a}, Prompting: {b} \n Pass Percentage: {pass_percentage:.2f}% \n Average Metrics: {average_metrics}")

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

# Function for conducting all the prompting techniques
def prompting(file_name, a):
    list1 = []
    list2 = []
    list3 = []
    list5 = []
    client = openai
    openai.api_key = "ENTER_API_KEY"

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
                    abstract = sentence_info["Citation"]["Citation Paper Abstract"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    list5.append(paper.split(":", 1)[1].strip())
                    response = client.chat.completions.create(
                        model= a,
                        messages=[
                            {
                                "role": "system",
                                "content": f"I have taken a sentence from the research paper titled \"{most_similar_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else."
                            },
                            {
                                "role": "user",
                                "content": user_content
                            }
                        ],
                        temperature=1,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    assistant_response = response.choices[0].message.content
                    list2.append(assistant_response)
                    if assistant_response == 'pass' or compare(list2,list5):
                        r1 = client.chat.completions.create(
                                model= a,
                                messages=[
                                {
                                    "role": "system",
                                    "content": f"I have taken a sentence from the research paper titled \"{most_similar_title}\", give me a list of the title of the possible top 5 research papers that this sentence is citing to. If you are not able to come up with the paper titles write 'pass'.Don't write anything else."
                                },
                                {
                                    "role": "user",
                                    "content": user_content
                                },
                                {
                                "role": "user",
                                "content": f"let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}"
                                }
                            ],
                            temperature=1,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
                        assistant_r1 = r1.choices[0].message.content
                        list3.append(assistant_r1)
                    if assistant_r1 == 'pass' or compare(list3,list5):
                        r2 = client.chat.completions.create(
                            model= a,
                            messages=[
                            {
                                "role": "system",
                                "content": f"I have taken a sentence from the research paper titled \"{most_similar_title}\", give me a list of the title of the possible top 5 research papers that this sentence is citing to. If you are not able to come up with the paper titles write 'pass'.Don't write anything else."
                            },
                            {
                                "role": "user",
                                "content": user_content
                            },
                            {
                                "role": "user",
                                "content": f"let me give you some more context by providing the abstract of the research paper it is citing to. {abstract}"
                            },
                            {
                            "role": "user",
                            "content": f"let me give you some more context by providing the author names of the research paper it is citing to. {author}"
                            }
                            ],
                            temperature=1,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
                        assistant_r2 = r2.choices[0].message.content
                        list1.append(assistant_r2)
                    else:
                        list1.append(assistant_r1)
                else:
                    list1.append(response.choices[0].message.content)
                    print(list1,"\n", list5)
        metrics(a, "SID_Paper_Title",list1,list5)
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
                    paper_abstract = sentence_info["Citation"]["Citation Paper Abstract"]

                    # Find most similar paper abstract
                    other_abstracts = [sentence["Citation"]["Citation Paper Abstract"] for sentence in sentences if sentence["Citation"]["Citation Paper Abstract"] != paper_abstract]
                    most_similar_abstract = find_most_similar_abstract(paper_abstract, other_abstracts)

                    print(f"Paper Title: {paper_title}")
                    print(f"Original Abstract: {paper_abstract}")
                    print(f"Most Similar Abstract: {most_similar_abstract}")

                    paper = sentence_info["Citation"]["Citation Paper Title"]
                    author = sentence_info["Citation"]["Citation Paper Authors"]
                    list5.append(paper.split(":", 1)[1].strip())
                    response = client.chat.completions.create(
                        model= a,
                        messages=[
                            {
                                "role": "system",
                                "content": f"I have taken a sentence from the research paper titled \"{paper_title}\", give me the title of the possible research paper that this sentence is citing to. If you are not able to come up with the paper title write 'pass'. Don't write anything else."
                            },
                            {
                                "role": "user",
                                "content": user_content
                            }
                        ],
                        temperature=1,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    assistant_response = response.choices[0].message.content
                    list2.append(assistant_response)
                    if assistant_response == 'pass' or compare(list2,list5):
                        r1 = client.chat.completions.create(
                                model= a,
                                messages=[
                                {
                                    "role": "system",
                                    "content": f"I have taken a sentence from the research paper titled \"{paper_title}\", give me a list of the title of the possible top 5 research papers that this sentence is citing to. If you are not able to come up with the paper titles write 'pass'.Don't write anything else."
                                },
                                {
                                    "role": "user",
                                    "content": user_content
                                },
                                {
                                "role": "user",
                                "content": f"let me give you some more context by providing the abstract of the research paper it is citing to. {most_similar_abstract}"
                                }
                            ],
                            temperature=1,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
                        assistant_r1 = r1.choices[0].message.content
                        list3.append(assistant_r1)
                    if assistant_r1 == 'pass' or compare(list3,list5):
                        r2 = client.chat.completions.create(
                            model= a,
                            messages=[
                            {
                                "role": "system",
                                "content": f"I have taken a sentence from the research paper titled \"{paper_title}\", give me a list of the title of the possible top 5 research papers that this sentence is citing to. If you are not able to come up with the paper titles write 'pass'.Don't write anything else."
                            },
                            {
                                "role": "user",
                                "content": user_content
                            },
                            {
                                "role": "user",
                                "content": f"let me give you some more context by providing the abstract of the research paper it is citing to. {most_similar_abstract}"
                            },
                            {
                            "role": "user",
                            "content": f"let me give you some more context by providing the author names of the research paper it is citing to. {author}"
                            }
                            ],
                            temperature=1,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                            )
                        assistant_r2 = r2.choices[0].message.content
                        list1.append(assistant_r2)
                    else:
                        list1.append(assistant_r1)
                else:
                    list1.append(response.choices[0].message.content)
                    print(list1,"\n", list5)
        metrics(a, "SID_Paper_Abstract",list1,list5)
        list1.clear()
        list2.clear()
        list3.clear()
        list5.clear()
        return

def main():
  openai_models = ["gpt-3.5-turbo","gpt-4","gpt-4-1106-preview"]
  file_name = "ENTER_FILE_NAME"
  for i in range(len(openai_models)):
    model = openai_models[i]
    prompting(file_name, model)

if __name__ == "__main__":
    main()