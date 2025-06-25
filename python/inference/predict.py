import os
import sys
import json
import pandas as pd
from cicle import CICLe
from data_hub_client import DataHubClient

####################################################################################################
# Get Environment Variables:                                                                       #
####################################################################################################

DATAHUB_API_KEY = os.getenv('DATAHUB_API_KEY')
DATAHUB_HOST = os.getenv('DATAHUB_HOST')
MODEL_NAMESPACE = os.getenv('MODEL_NAMESPACE')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_VERSION = os.getenv('MODEL_VERSION')
INPUT_DATASET = os.getenv('INPUT_DATASET')
OUTPUT_DATASET = os.getenv('OUTPUT_DATASET')
OUTPUT_DESC = os.getenv('OUTPUT_DESC')
OUTPUT_TAGS = os.getenv('OUTPUT_TAGS')

LLM_API_KEY = os.getenv('LLM_API_KEY')

####################################################################################################
# Helper Functions:                                                                                #
####################################################################################################

def data2json(input:pd.DataFrame, output:pd.DataFrame, path:str):
    # assert matching data:
    assert (input['title'].values == output['title'].values).all()
    
    # copy new columns:
    for col in output.columns: input[col] = output[col].values

    # save data:
    with open(path, 'w') as file:
        json.dump([{k:v for k,v in zip(row.index, row.values)} for _, row in input.iterrows()], file)

def loadLLM_ollama(llm_name:str):
    from ollama import Client

    # create llm client:
    olc = Client(
        host='http://ollama.open-webui.svc.cluster.local:11434',
        headers={}
    )

    # pull llm if necessary:
    if llm_name not in olc.list():
        olc.pull(llm_name)

    # return generation function:
    return lambda x, n, t: [messages + [olc.chat(
        model=llm_name,
        messages=messages,
        options = {'temperature':t, 'max_new_tokens':n}
    )['message']] for messages in x]

def loadLLM_transformers(llm_name:str):
    import torch
    from transformers import pipeline
    from huggingface_hub import login

    # log in to Huggingface:
    login(LLM_API_KEY)

    # create llm pipeline:
    llm = pipeline(
        "text-generation",
        model=llm_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto"
    )

    # get special tokens for later:
    bos_token_id = llm.tokenizer.convert_tokens_to_ids('<|begin_of_text|>')
    eos_token_id = llm.tokenizer.convert_tokens_to_ids('<|eot_id|>')
    pad_token_id = llm.tokenizer.convert_tokens_to_ids('<|eot_id|>')

    # return generation function:
    return lambda x, n, t: [item[0]["generated_text"] for item in llm(x,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        max_new_tokens=n,
        do_sample=t is not None,
        temperature=t,
        top_p=None
    )]

####################################################################################################
# Main Function:                                                                                   #
####################################################################################################

if __name__ == '__main__':
    assert len(sys.argv) == 3
    LLM_NAME = sys.argv[1]
    LABEL_NAME = sys.argv[2]

    # load base classifier:
    cicle = CICLe()

    # load data:
    dhc   = DataHubClient(DATAHUB_HOST, DATAHUB_API_KEY)
    input = dhc.get_dataset_data(INPUT_DATASET)
    input = pd.read_json(input)[['id','title']]

    # create prompt:
    prompt = f'We are looking for food {LABEL_NAME}s in texts. Predict the correct class for the following sample. Only provide the class label. Here are some labelled examples sorted from most probable to least probable:'

    # create generation function:
    #generate = loadLLM_transformers(LLM_NAME)
    generate = loadLLM_ollama(LLM_NAME)

    # predict data:
    output = cicle(input['title'], prompt, generate, batch_size=4)
    data2json(input, output, 'ouput.json')

    # save outputs:
    dhc.create_dataset(
        OUTPUT_DATASET, 'ouput.json', OUTPUT_DESC, [OUTPUT_TAGS]
    )

    # clean up:
    os.remove('ouput.json') 