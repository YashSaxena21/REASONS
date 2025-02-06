from pickle import NONE
import json
import re
import requests
import PyPDF2
from bs4 import BeautifulSoup
import feedparser
import time

def extract_text_from_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as pdf_file:
            pdf_file.write(response.content)

        text = ""
        with open("temp.pdf", "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

#Retrieving all the citation details
def get_citation_details(url):
  try:
    page = requests.get(url)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, 'html.parser')

    # Extracting Citation Paper Title
    paper_title = soup.find('h1', class_='title mathjax').text.strip()

    # Extracting Citation Paper ID
    paper_id = soup.find('strong').text.strip()

    # Extracting Citation Abstract
    abstract = soup.find('blockquote', class_='abstract mathjax').text.strip()

    # Extracting Citation Paper Authors
    authors = soup.find_all('div', class_='authors')
    authors_list = []
    for author in authors:
      if(author=='Authors:'):
        continue
      authors_list.append(author.text)

    return {
      "Citation Paper ID": paper_id,
      "Citation Paper Title": paper_title,
      "Citation Paper Abstract": abstract,
      "Citation Paper Authors": ' '.join(map(str, authors_list))
    }
  except Exception as e:
    print(f"Error getting citation details: {str(e)}")
    return {}

#Searching for arxiv paper based on citation text using google search via proxy chaining
def get_first_arxiv_page_url(citation_text, max_retries=3, delay_seconds=1):
        try:
          time.sleep(1)
          payload = {
                    'source': 'google_search',
                    'query': citation_text
                }

          # Get response, used Oxylabs SERP API
          response = requests.request(
                    'POST',
                    'https://realtime.oxylabs.io/v1/queries',
                    auth=('USER_NAME', 'PASSWORD'),
                    json=payload,
                )
          r = response.json()
          data = r['results'][0]['content']
          soup = BeautifulSoup(data, 'html.parser')
          result_links = soup.find_all('a')

          for link in result_links:
            href = link.get('href')
            if href and href.startswith("https://arxiv.org/"): #Fetch the first page that starts with 'https://arxiv.org/'
              return href

        except Exception as e:
          print(f"Error: {str(e)}")

        return None

def check_authors_match(citation_text, citation_info_authors):

  # Get the first author from the citation text.
    citation_text_authors = re.findall(r'^(.*?),', citation_text)
    if(len(citation_text_authors)==0 or len(citation_text_authors)>1):
      return False

  # Get the first author from the citation info.
    citation_info_auths = re.findall(r'Authors:(.*?),', citation_info_authors)
    if(len(citation_info_auths)==0 or len(citation_info_auths)>1):
      return False

  # Check if the authors match.
    if citation_text_authors and citation_info_auths:  # Check if both lists are non-empty
        last_A = citation_text_authors[-1]
        last_B = citation_info_auths[-1]
        if isinstance(last_A, str) and isinstance(last_B, str):  # Check if the last elements are strings
            words_A = last_A.split()
            words_B = last_B.split()
            if words_A and words_B:  # Check if the lists resulting from the split are non-empty
                last_word_A = words_A[-1]
                last_word_B = words_B[-1]
                return last_word_A.lower() == last_word_B.lower()
    return False

#When all the contents of the pdf are read, then finding the Related works section in it, extracting sentences with citations and returning the json structure
def combine_and_create_json_structure(domain, pdf_url, min_sentence_length, max_sentence_length):
    pdf_text = extract_text_from_pdf(pdf_url)
    if not pdf_text:
        return {}

    section_titles = ["Related Works", "Related Work", "Literature Review", "Background", "Prior Work", "LITERATURE SEARCH", "RELATED WORKS", "RELATED WORK", "LITERATURE REVIEW", "Literature Search"]

    related_works = None
    for title in section_titles:
        pattern = re.compile(rf'{title}(.*?)(?:References|$)', re.DOTALL | re.IGNORECASE)
        Match = pattern.search(pdf_text)
        if Match:
            related_works = Match.group(1).strip()
            break

    if not related_works:
        return {}

    sentences = {}
    citation_pattern = re.compile(r'(\[\s*\d+\s*\])')
    citations_and_sentences = re.split(citation_pattern, related_works)

    citation_id = None
    prev_sentence = ""
    citation_text = ""

    for item in citations_and_sentences[::-1]:
        if item.startswith('['):
            if citation_id:
                sentences[citation_id] = {
                    'sentence': prev_sentence,
                    'citation_text': citation_text,
                }
            citation_id = int(re.search(r'\d+', item).group())
            prev_sentence = ""
            citation_text = ""
        else:
            if item.strip():
                prev_sentence = item.strip() + ' ' + prev_sentence
            else:
                prev_sentence = item.strip() if not prev_sentence else prev_sentence

    if citation_id:
        sentences[citation_id] = {
            'sentence': prev_sentence,
            'citation_text': citation_text,
        }

    references_pattern = re.compile(r'(\[\s*\d+\s*\])\s*(.*?)\s*(?=\[\s*\d+\s*\]|$)', re.DOTALL)
    references = references_pattern.findall(pdf_text)
    for ref_id, ref_text in references:
        ref_id = int(re.search(r'\d+', ref_id).group())
        if ref_id in sentences:
            sentences[ref_id]['citation_text'] = ref_text.strip()
    sentence_keys = list(sentences.keys())
    for sentence_key in sentence_keys:
        sentence_length = len(sentences[sentence_key]['sentence'].split())
        if sentence_length < min_sentence_length or sentence_length > max_sentence_length:
            del sentences[sentence_key]

    json_structure = {
        "Domain": domain,
        "Papers": [
            {
                "Paper ID": pdf_url,
                "Paper Title": "Placeholder Title",
                "Sentences": []
            }
        ]
    }
    for sentence_id, sentence_data in sentences.items():
        citation_text = sentence_data['citation_text']
        citation_url = get_first_arxiv_page_url(citation_text)
        if citation_url:
            citation_info = get_citation_details(citation_url)
            if 'Citation Paper Authors' in citation_info:
              # The Citation Paper Authors key exists.
              citation_info_authors = citation_info['Citation Paper Authors']
            else:
              continue
            if not check_authors_match(sentence_data['citation_text'], citation_info_authors):
              continue
            json_structure["Papers"][0]["Sentences"].append({
                "Sentence ID": sentence_id,
                "Sentence": sentence_data['sentence'],
                "Citation Text": citation_text,
                "Citation": citation_info
            })


    return json_structure

def main():
    domain = "DOMAIN_NAME"  # You can set the domain over here
    max_results = 50  # You can change the number of papers been scraped from here
    min_sentence_length = 10 #Minimum number of words that the sentence should have
    max_sentence_length = 40 #Maximum number of words that the sentence should have

    # Specify the output filename
    output_filename = 'OUTPUT_FILE_NAME'

    arxiv_data = extract_related_works_from_arxiv(domain, max_results, min_sentence_length, max_sentence_length, output_filename)

def extract_related_works_from_arxiv(domain, max_results, min_sentence_length, max_sentence_length, output_filename):
    # Mention the time interval from which the papers should be extracted
    start_date = "2017-01-01" 
    end_date = "2023-12-01"
    papers = get_papers_from_arxiv(start_date, end_date, max_results)

    data = {}

    for entry in papers.entries:
        pdf_url = entry.links[1].href
        if pdf_url:
            paper_id = entry.id
            paper_title = entry.title
            json_structure = combine_and_create_json_structure(domain, pdf_url, min_sentence_length, max_sentence_length)
            if json_structure:
                if domain not in data:
                    data[domain] = {}

                if paper_id not in data[domain]:
                    data[domain][paper_id] = {
                        'Paper Title': paper_title,
                        'Sentences': {},
                    }
                data[domain][paper_id]['Sentences'] = json_structure["Papers"][0]["Sentences"]

                with open(output_filename, 'w') as json_file:
                    json.dump(data, json_file, indent=4)

    return data

def get_papers_from_arxiv(start_date, end_date, max_results):
    url = f"http://export.arxiv.org/api/query?search_query=cat:cs.CV+AND+submittedDate:[{start_date}+TO+{end_date}]&sortBy=lastUpdatedDate&sortOrder=descending&start=0&max_results={max_results}"
    response = requests.get(url)
    feed = feedparser.parse(response.text)
    return feed

if __name__ == "__main__":
    main()