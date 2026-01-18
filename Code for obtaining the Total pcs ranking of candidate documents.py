from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings,HuggingFaceEmbeddings
import numpy as np
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma


embeddings=HuggingFaceEmbeddings(model_name=r'/root/autodl-tmp/RAG_langchain/Dmeta-embedding',encode_kwargs = {'normalize_embeddings': True} )

loader = DirectoryLoader(r"/root/autodl-tmp/RAG_langchain/Potential_knowledge_sources"+r"/", show_progress=True,use_multithreading=True)
loaded_docs = loader.load()
print("len(loaded_docs)",len(loaded_docs))
##文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300,chunk_overlap = 50,)
docs = text_splitter.split_documents(loaded_docs)
print("len(docs)",len(docs))
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")

db = FAISS.load_local("faiss_index", embeddings)   #完整版本的知识向量化后的库
 # 指定您的Excel文件路径
excel_file =r'/root/autodl-tmp/RAG_langchain/Dataset_for_builing_DKR/L_1314+SE.xlsx'

# 读取Excel文件
df_o = pd.read_excel(excel_file)
# 假设您要读取的列名分别为'Column1'和'Column2'
# 将这两列分别存入两个列表中
queries = df_o["原始问题"].tolist()
query_code = df_o["序号"].tolist()
data_for_rag=[]
for index, query in enumerate(queries):  
#query = "1、  人民法院审理行政案件，不适用（）。        A 调解        B 开庭审理        C 公开审理        D 两审终审制  "
#docs = db.similarity_search(query
    searched_docs = db.similarity_search_with_score(query,k = 3)
    rank=0
    for doc, score in searched_docs:
        page_content = doc.page_content  # 提取页面内容
       # page = doc.metadata['page']      # 提取页码
        source = doc.metadata # 提取来源
        rank=rank+1
        pcs=1/(score*rank)
        data_for_rag.append([query_code[index],query,page_content, source, score,rank,pcs])
        

import os
import shutil

def empty_folder(folder_path):
    """
    清空指定的文件夹，删除其中的所有文件和子文件夹。
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def copy_file_to_folder(source_file_path, target_folder_path):
    """
    将文件从源路径复制到目标文件夹。

    参数:
    source_file_path: 源文件的完整路径。
    target_folder_path: 目标文件夹的路径。
    """
    # 检查源文件是否存在
    if not os.path.isfile(source_file_path):
        print(f"文件不存在: {source_file_path}")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(target_folder_path):
        os.makedirs(target_folder_path)

    # 构建目标文件的完整路径
    target_file_path = os.path.join(target_folder_path, os.path.basename(source_file_path))

    # 复制文件
    shutil.copy2(source_file_path, target_file_path)
    print(f"文件已复制到: {target_file_path}")






# data = []

# 遍历documents中的每个元素

# for doc, score in searched_docs:
#     page_content = doc.page_content  # 提取页面内容
#    # page = doc.metadata['page']      # 提取页码
#     source = doc.metadata # 提取来源
#     data.append([query,page_content, source, score])

#将列表转换为DataFrame
df = pd.DataFrame(data_for_rag, columns=['code','query','Page Content', 'Sourced_doc', 'Score','rank','pcs'])


#grouped_scores = df.groupby('Source')['pcs'].sum()


df['doc_tpcs'] = df['Sourced_doc'].apply(lambda x: x['source'] if isinstance(x, dict) and 'source' in x else None)

# 现在，使用新列'Source_Key'来分组并计算pcs的总和
grouped_scores = df.groupby('doc_tpcs')['pcs'].sum()

# 将结果转换为DataFrame
grouped_scores_df = grouped_scores.reset_index()
grouped_scores_df.columns = ['doc_tpcs', 'Total pcs']
# 按 'Total pcs' 列排序
sorted_df = grouped_scores_df.sort_values(by='Total pcs', ascending=False)

# 打印排序后的DataFrame

# 保存DataFrame到Excel文件
df.to_csv('extracted_data-wrong.csv', index=False)
grouped_scores_df.to_excel('grouped_scores_df-wrong.xlsx', index=False)


print("excel_file")