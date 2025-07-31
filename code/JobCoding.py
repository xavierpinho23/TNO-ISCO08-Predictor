# Import packages
import pandas as pd
import numpy as np
from openai import AzureOpenAI
import json
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from itertools import islice
import re
import os

# Define Azure OpenAI client
client = AzureOpenAI(
    api_key        = os.getenv('AZURE_OPENAI_API_KEY'),
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT'),
    api_version    = os.getenv('AZURE_OPENAI_VERSION')
)

# Import data from ISCO 2008 ontology
isco08_data = pd.read_excel("./data/ISCO08_all_groups.xlsx", "complete list")
isco08_jobs = pd.read_excel("./data/ISCO08 coding indexes (various languages).xlsx", "English index")

################ Generate embeddings for ISCO08 data ###################
# The embedding for an ISCO code is based on the description of that 
# ISCO category and a list of 10 example jobs for that ISCO category

# First, combine descriptive and job data from ISCO ontology
def convert_to_readable_job(job_name):
    split_chars = ['(', ':']

    # Create a regular expression pattern using the characters
    pattern = f"[{''.join(re.escape(char) for char in split_chars)}]"

    # Use re.split() to split the string based on the pattern
    j = re.split(pattern, job_name)

    j1 = j[0].strip().split(', ')
    j1 = reversed(j1)
    readable_job_name = ' '.join(j1)
    if len(j) > 1:
        readable_job_name += ' (' + '('.join([jj.strip() for jj in j[1:]])
        if readable_job_name[-1] != ')':
            readable_job_name += ')'

    return readable_job_name

isco_data_merged = {}
with tqdm(total = len(isco08_data)) as pbar:
    for index, row in isco08_data.iterrows():
        isco08_code = row['CODE']
        group_name = row['DESCRIPTION']
        isco_data_merged[isco08_code] = {
            'description': group_name,
            'jobs': []
        }        

        pbar.update(1)

with tqdm(total = len(isco08_jobs)) as pbar:
    for index, row in isco08_jobs.iterrows():
        isco08_code = row['Code']
        job_name    = row['Text']

        readable_job_name = convert_to_readable_job(job_name.lower())

        if isco08_code not in isco_data_merged:
            print(f'code not found: {isco08_code}')
            continue

        isco_data_merged[isco08_code]['jobs'].append(readable_job_name.strip())

        pbar.update(1)

# Use Azure OpenAI client to generate embedding for a phrase
def get_gpt_embedding(phrase):
    embedding = client.embeddings.create(input = [phrase], model = 'text-embedding-3-large').data[0].embedding
    return embedding

# Second, generate an embedding based on ISCO description + jobs
with tqdm(total = len(isco_data_merged)) as pbar:
    for isco_code, merged_data in isco_data_merged.items():
        natural_text = merged_data['description'] + '\n' + '\n'.join(merged_data['jobs'])
        isco_data_merged[isco_code]['embedding'] = get_gpt_embedding(natural_text)
        pbar.update(1)

# Write ISCO data with embeddings to a txt file
output_file = './data/isco08_embeddings_based_on_description_and_example_jobs.txt'
with open(output_file, 'w') as op_file:
    for code, embedding_data in isco_data_merged.items():
        op_file.write('{}\t{}\t{}\t{}\n'.format(str(code).ljust(4, '0'), embedding_data['description'], ';'.join(embedding_data['jobs']), ','.join([str(n) for n in embedding_data['embedding']])))
        
isco_embedding_file = './data/isco08_embeddings_based_on_description_and_example_jobs.txt'
isco_embeddings = {}

with open(isco_embedding_file, 'r') as ip_file:
    lines = ip_file.readlines()
    for line in lines:
        line_elements = line.strip().split('\t')
        isco_embeddings[line_elements[0]] = {
            'description': line_elements[1],
            'jobs': line_elements[2],
            'embedding': [float(e) for e in line_elements[3].split(',')]
        }

############### Read competition data file and convert to embeddings ######### 

competition_data = pd.read_csv('./data/competition_data.csv', sep = ';', encoding = 'utf-8')

competition_set_embeddings = []
with tqdm(total = len(competition_data)) as pbar:
    for index, row in competition_data.iterrows():
        job = row['title']
        job_embedding = get_gpt_embedding(job.strip())
    
        competition_set_embeddings.append(
            {
                'id': row['id'],
                'job_title': row['title'],
                'job_description': row['description'],
                'job_embedding': job_embedding
            }
        )
    
        pbar.update(1)

############## Write competition data with embeddings to txt file
output_file = './data/competition_data_with_job_embeddings.txt'
with open(output_file, 'w') as op_file:      
    for record in competition_set_embeddings:
        if isinstance(record['job_description'], float):
            op_file.write('{}\t{}\t{}\t{}\t{}\n'.format(record['id'], record['job_title'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' '), '', record['prediction_xgboost'], ','.join([str(j) for j in record['job_embedding']])))
        else:
            op_file.write('{}\t{}\t{}\t{}\t{}\n'.format(record['id'], record['job_title'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' '), record['job_description'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' '), record['prediction_xgboost'], ','.join([str(j) for j in record['job_embedding']])))

############## RETRIEVER MODEL ###############
# For each entry in the competition dataset: get top-20 ISCO candidates based on comparison between ISCO embeddings and the entry's embedding 

def determine_code_based_on_embedding(embeddings, job_embedding, top_n):
    matched_codes = {}
    
    for code, embedding in embeddings.items():
        similarity = cosine_similarity(np.array(embedding['embedding']).reshape(1, -1), np.array(job_embedding).reshape(1, -1))
        matched_codes[code] = similarity[0][0]

        matched_codes = dict(sorted(matched_codes.items(), key=lambda item: item[1], reverse=True))

        if len(matched_codes) > top_n:
            matched_codes = dict(islice(matched_codes.items(), top_n))

    return matched_codes

counter = 0
with tqdm(total = len(competition_set_embeddings)) as pbar:
    for competition_set_embedding in competition_set_embeddings:
        candidate_codes = determine_code_based_on_embedding(isco_embeddings, competition_set_embedding['job_embedding'], 20)
        candidate_data = []
        for cc, cand in candidate_codes.items():
            candidate_data.append({'code': cc, 'jobs': isco_embeddings[cc]['jobs'], 'description': isco_embeddings[cc]['description']})
        
        competition_set_embeddings[counter]['candidates'] = candidate_data

        pbar.update(1)
        counter += 1


############## GENERATOR MODEL ################
# For each entry in the competition dataset: ask GPT to select the best match from the 20 candidates provided by the Retriever model

def select_best_code_with_gpt_based_on_job_only(job, candidates):
    task_message = {
        "role": "user",
        "content": f"""You are an expert in assigning ISCO-08 ontology labels to job titles.
        Provide the ISCO-08 code corresponding to the following job title.

        JOB: {job}
        
        You should select the ISCO-08 code from the following list of candidates: {candidates}
        
        When selecting the correct candidate code, make sure to take into account the sample jobs listed for each of the candidates.
        Skip any explanation. Only return the selected ISCO-08 code. Do not write any plain text. Always select a code, even if you're not sure.
        """
    }
    
    response = client.chat.completions.create(
        model='gpt4o',
        messages=[task_message],
        temperature=0,
        max_tokens=100,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        seed=123
    )

    if response.choices[0].message.content is not None:
        return response.choices[0].message.content.strip()

    return -1


# Extract four-digit code from GPT response text
def process_response(response_text):
    if len(response_text) > 4:
        if response_text[-4:].isdigit():
            #print(response_text + ' >>> ' + response_text[-4:])
            return response_text[-4:]

    return response_text


rag_results = []
with tqdm(total = len(competition_set_embeddings)) as pbar:
    for competition_set_embedding in competition_set_embeddings:
        job_id = competition_set_embedding['id']
        job = competition_set_embedding['job_title']
        
        candidate_codes = [f"Code: {c['code']}\tDescription: {c['description']}\tExample jobs: {'; '.join(c['jobs'].split(';')[:10])}" for c in competition_set_embedding['candidates'][:10]]
    
        best_code = select_best_code_with_gpt_based_on_job_only(job, '\n'.join(candidate_codes))
    
        rag_results.append({
            'id': competition_set_embedding['id'],
            'job': job,
            'rag_code': process_response(best_code),
            'candidates': competition_set_embedding['candidates']
        })

        pbar.update(1)

# write results to pandas dataframe and to an Excel file
rag_df = pd.DataFrame(rag_results)
rag_df.to_excel('./data/rag_results_competition_complete.xlsx', sheet_name='predictions', index=False)


###### Correct invalid codes
labels = pd.read_csv("./data/wi_labels.csv", dtype="str")

def find_valid_isco(isco, labels):
    # make sure that isco is a string
    isco = str(isco)

    # valid ISCO codes
    valid_codes = labels["code"].tolist()

    # check if it is a valid ISCO code (i.e. exists in the labels file)
    if isco in valid_codes:
        valid_isco = isco
    else:
        # select codes that start with the same digit(s)
        for i in range(2, -1, -1):
            filtered_codes = [s for s in valid_codes if s[:i] == isco[:i]]
            if len(filtered_codes) > 0:
                # select the smallest code (as if they were numbers)
                valid_isco = min(filtered_codes, key=int)
                break
            
    return valid_isco

for index, record in rag_df.iterrows():
    rag_code = record['rag_code']
    valid_rag_code = find_valid_isco(rag_code, labels)

    if rag_code != valid_rag_code:
        rag_df.loc[rag_df['id'] == record['id'], 'rag_code'] = valid_rag_code


# write corrected results to an Excel file
rag_df.to_excel('./data/rag_results_competition_complete_valid_codes.xlsx', sheet_name='predictions', index=False)


################# ADJUST CODES FOR UNCERTAIN JOBS AND MANAGER JOBS ###########
# If the similarity score between a competition record's embedding and the most
# similar ISCO entry embedding is < 0.35, or if the competition record's job
# title contains the word 'manager', extract the occupation from the record's
# description and generate a new embedding for this record based on the extracted
# occupation 
        
def get_occupation(text):
    """
    Extracts the occupation/task of a given text.
    """

    if text.strip():
        combined_message = {
            "role": "user",
            "content": f"""Extract the main occupation and tasks from the text: {text} and return it in JSON format:

            The text can be in different languages. If the language is not english, return the occupation/task in english.
            Keep the answer short and concise.
            Skip any explanation.
            Use only english words in your answer.
            Keep your answer to a limit of maximum of 3 tasks and just 1 occupation.
            You only speak JSON. Do not write any plain text.

            Here is an example of input and output:

            "Input": "12 month contract - Italian, French, German or Spanish JOB DESCRIPTION · Handle day to day Credit and Collection activities on customer accounts · Liaising with Customer Services to ensure customer queries are resolved on a timely basis. · Reconciliation of customer accounts to ensure clean aged debt report · Order release QUALIFICATIONS · At least 18 months experience in a credit control environment · Educated to at least Certificate/Diploma level · Excellent communication skills both written and verbal · Good numeracy & analytical skills · Fluency in Italian, French, German or Spanish and English essential Desirable · Previous exposure to a culturally diverse working environment · Shared Service Centre experience"

            "Output": 
                "Tasks": "Credit and Collection Activities, customer account reconcilliation, liasing with customer service and order release",
                "Occupation": "Credit Controller or Collections Specialist"
                
            """
            }

        response = client.chat.completions.create(
            model='gpt4o',
            messages=[combined_message],
            response_format = {"type":"json_object"},
            temperature=0,
            max_tokens=200,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0,
        )
        response_content = response.choices[0].message.content.strip()

        response_json = json.loads(response_content)

        tasks = response_json.get('Tasks', None)
        occupation = response_json.get('Occupation', None)

        # Convert the tasks list to a single string if it's a list
        if isinstance(tasks, list):
            tasks = ', '.join(tasks)
        
        return [tasks, occupation]
    
    
new_competition_set_embeddings = []
changed_records = []

counter = 0
with tqdm(total = len(competition_set_embeddings)) as pbar:
    for record in competition_set_embeddings:
        candidate_codes = determine_code_based_on_embedding(isco_embeddings, record['job_embedding'], 10)
        max_score = round(list(candidate_codes.values())[0], 2)
        new_competition_set_embeddings.append(record)

        if 'manager' in record['job_title'].lower() or max_score < 0.35:
            try:
                [tasks, occupation] = get_occupation(record['job_description'])
                new_job_embedding = get_gpt_embedding(occupation.strip())
                new_competition_set_embeddings[counter]['job_embedding'] = new_job_embedding
                changed_records.append(record['id'])
            except Exception as e:
                print(f"Skipping record {record['job_title']} because GPT raised an error: {e}")
        pbar.update(1)
        counter += 1


def get_adapted_record(job_id):
    for record in new_competition_set_embeddings:
        if record['id'] == job_id:
            return record
    return {}


with tqdm(total = len(rag_df)) as pbar:
    for idx, record in rag_df.iterrows():
        if str(record['id']) in changed_records:
            adapted_record = get_adapted_record(str(record['id']))
            candidate_codes = determine_code_based_on_embedding(isco_embeddings, adapted_record['job_embedding'], 10)
            candidate_data = []
            for cc, cand in candidate_codes.items():
                candidate_data.append({'code': cc, 'jobs': isco_embeddings[cc]['jobs'], 'description': isco_embeddings[cc]['description']})
            candidate_codes = [f"Code: {cd['code']}\tDescription: {cd['description']}\tExample jobs: {'; '.join(cd['jobs'].split(';')[:10])}" for cd in candidate_data]
            best_code = select_best_code_with_gpt_based_on_job_only(adapted_record['job_title'], '\n'.join(candidate_codes))
    
            valid_rag_code = find_valid_isco(best_code[-4:], labels)
            rag_df.loc[rag_df['id'] == record['id'], 'rag_code'] = valid_rag_code
            rag_df.loc[rag_df['id'] == record['id'], 'candidates'] = str(candidate_data)

        pbar.update(1)

# write results to a csv file that can be submitted for the competition 
rag_df.to_excel('./data/rag_results_competition_complete_use_tasks_for_uncertain_jobs.xlsx', sheet_name='predictions', index=False)

# Specify the columns you want to write to the CSV
columns_to_write = ['id', 'rag_code']
rag_df.to_csv('./data/tno_ailab_competition.csv', columns = columns_to_write, index=False)
