#!/usr/bin/env python
# coding: utf-8

# importing libraries

# In[23]:


import pandas as pd
import numpy as np
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score


# Reduced Dataset is the portion of the initial Dataset that was complete

# In[3]:


reduced_df = pd.read_csv('/home/profniggastein/PycharmProjects/ReducedDataset/Project_CodeNet/Reduced_df.csv')
reduced_df


# Using a small sample of the first 100 entries as a test for the summarisation pipeline

# In[4]:


df = reduced_df.copy()
df_sample = df.head(100).reset_index(drop=True)


# In[5]:


df = df_sample
df


# Using the DeepSeek model to generate summaries for the code submissions in the dataset.              In this test the base deepseek model is used and gives satisfactoy results at least for the sample data ,in proper implementation the plan is to use the deepsek reasoning model for better results.

# In[8]:


client = OpenAI(api_key="sk-9a18103e17be4329a06b65b1520c5d9b", base_url="https://api.deepseek.com")

def GenerateSummary(text):

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": """You are are an assistant/tutor helping users understand various code snippets by giving a general summary and as much as poosible give them  in a general summary form for specifically c and c++ code like this;Code Summary Format

    1. Function/Method Name:
    process_data

    2. Purpose:
    Briefly describe what the code is intended to do.
    E.g., Processes and cleans input data by removing null values and standardizing formats.

    3. Inputs:

    dataframe (pandas.DataFrame): The input dataset containing raw values.

    columns (List[str]): Specific columns to be processed.


    4. Outputs:

    cleaned_dataframe (pandas.DataFrame): A cleaned version of the input dataset.


    5. Key Operations/Logic:

    Drops rows with null values in specified columns

    Converts column names to lowercase

    Normalizes date formats


    6. Dependencies:

    pandas

    datetime


    7. Edge Cases Handled:

    Returns an empty DataFrame if all rows contain nulls

    Skips processing if columns is empty


    8. Example Usage (Optional):

    clean_df = process_data(df, ["date", "price", "location"])"""},
            {"role": "user", "content": f"{text}"},
        ],
        stream=False
    )
    return response



# In[9]:


df = df_sample
df['GenSummary'] = df['code'].apply(lambda x: GenerateSummary(x).choices[0].message.content if pd.notnull(x) else None)
df


# In[16]:


df.to_csv('/home/profniggastein/PycharmProjects/ReducedDataset/Project_CodeNet/Reduced_sample_df.csv')
df

#df_sample


# for evaluating the generated sumaries BLEU and BERT score are the metrics used to compare the generated summaries with the original summaries in the dataset. the BERT score is calculated using the BERTScore library instead of a "manual" Bert model implimentation while this implimentation is faster access to the acc embedding-space is lost

# In[32]:


def calculate_bleu(reference, candidate):
    reference = reference.split()
    candidate = candidate.split()
    return sentence_bleu([reference], candidate, smoothing_function=SmoothingFunction().method1)
df['bleu_score'] = df.apply(
    lambda row: calculate_bleu(row['GenSummary'], row['summary']) if pd.notnull(row['summary']) and pd.notnull(row['GenSummary']) else None,
    axis=1
)
df['BERT_score'] = df.apply(
    lambda row: score([row["GenSummary"]], [row["summary"]], model_type="microsoft/codebert-base")[2][0].item(),
    axis=1
)
df


# In[33]:


df.to_csv('/home/profniggastein/PycharmProjects/ReducedDataset/Project_CodeNet/Reduced_sample_df_Evaluated.csv', index=False)
df

