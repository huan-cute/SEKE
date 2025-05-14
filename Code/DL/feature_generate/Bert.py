import os
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_texts(texts, tokenizer, max_length):
    input_ids, attention_masks = [], []
    encoded_dict = tokenizer.encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 指定包含所有数据集的目录
    dataset_dir = '../dataset/uc/'
    # 获取所有数据集的名称（即每个子文件夹的名称）
    datasets = os.listdir(dataset_dir)
    # datasets = ['Albergate']
    types = ['uc', 'cc']
    max_length = 512
    batch_size = 8


    tokenizer = BertTokenizer.from_pretrained('../HuggingFace/bert-base-uncased')
    model = BertModel.from_pretrained('../HuggingFace/bert-base-uncased')
    model.to(device)
    model.eval()

    for dataset in datasets:
        all_sheets_data = {}
        for type in types:
            folder_path = f'../dataset/{type}/{dataset}/'
            try:
                for filename in os.listdir(folder_path):
                    input_file = os.path.join(folder_path, filename)
                    print(f"Processing file: {input_file}")

                    with open(input_file, 'r', encoding='ISO-8859-1') as f:
                        texts = ' '.join([text.strip() for text in f.readlines()])

                    input_ids, attention_masks = preprocess_texts(texts, tokenizer, max_length)
                    input_ids, attention_masks = input_ids.to(device), attention_masks.to(device)

                    vectors = []
                    with torch.no_grad():
                        for i in tqdm(range(0, len(input_ids), batch_size)):
                            batch_input_ids = input_ids[i:i + batch_size]
                            batch_attention_masks = attention_masks[i:i + batch_size]
                            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
                            batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                            vectors.extend(batch_vectors)

                    vectors_with_filename = [[filename] + list(vector) for vector in vectors]
                    if type not in all_sheets_data:
                        all_sheets_data[type] = []
                    all_sheets_data[type].extend(vectors_with_filename)

            except Exception as e:
                print(f"Error processing {dataset}/{type}: {e}")

        output_file = f'../result/bert_vec_result/{dataset}/{dataset}_bert_vectors.xlsx'
        ensure_dir(output_file)
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            for type, data in all_sheets_data.items():
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=type, index=False)

        print(f"ALBERT vectors have been written to {output_file}")
