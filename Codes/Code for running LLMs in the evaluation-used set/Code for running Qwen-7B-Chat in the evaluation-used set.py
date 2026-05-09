from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen-7B-Chat", trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# Specify hyperparameters for generation. But if you use transformers>=4.32.0, there is no need to do this.
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
messages = []
import pandas as pd
import re
import os
import csv
from bs4 import BeautifulSoup
import pandas as pd
#from docx import Document
import uuid
import pandas as pd
import openpyxl


def score_of_single_choice(answers_from_model, correct_answer):
  score = 0
  number_of_answers=0
  if correct_answer in str(answers_from_model):
    score = 1
  if "A" in str(answers_from_model):
    number_of_answers=number_of_answers+1
  if "B" in str(answers_from_model):
    number_of_answers=number_of_answers+1
  if "C" in str(answers_from_model):
    number_of_answers=number_of_answers+1
  if "D" in str(answers_from_model):
    number_of_answers=number_of_answers+1
  if number_of_answers>1:
    score=0
  return score


def save_df_to_excel(df, file_path, sheet_name):
  # 创建一个Excel写入器对象
  writer = pd.ExcelWriter(file_path)

  # 将DataFrame写入指定工作表
  df.to_excel(writer, sheet_name=sheet_name, index=False)

  # 保存并关闭Excel文件

  writer.close()
def split_correct_answers(string):
  answer = []
  for character in string:
    answer.append(character)
  return answer


def score_of_multi_choice(answers_from_model, correct_answers):
  score = 0
  correct_ones = 0
  missed_ones = 0
  wrong_ones = 0
  # for answer in correct_answers:
  individual_correct_answers = split_correct_answers(correct_answers)
  for individual_answer in individual_correct_answers:
    if individual_answer in str(answers_from_model):
      correct_ones = correct_ones + 1
    if individual_answer not in str(answers_from_model):
      missed_ones = missed_ones + 1
  wrong_answers = set(["A", "B", "C", "D", "E"]).difference(correct_answers)
  #print("wrong_answers", wrong_answers)
  for individual_wrong_answer in wrong_answers:
    if individual_wrong_answer in str(answers_from_model):
      wrong_ones = wrong_ones + 1

  if wrong_ones == 0:
    if missed_ones == 0:
      score = 2
    else:
      score = min(correct_ones * 0.5, 2)
  #print("correct_ones,wrong_ones,missed_ones", correct_ones, wrong_ones, missed_ones)
  return score

def read_excel_column(file_path, sheet_name, column_name):
  # 读取Excel文件
  df = pd.read_excel(file_path, sheet_name=sheet_name)

  # 提取指定列的数据，转换为列表返回
  column_data = df[column_name].tolist()

  return column_data



years=["L_remaining","augmented_prompt10","augmented_prompt20","augmented_prompt25","augmented_prompt30","augmented_prompt40","augmented_prompt50","augmented_prompt75","augmented_prompt100","augmented_prompt150","augmented_prompt200","augmented_prompt250","augmented_prompt300","augmented_prompt350","augmented_prompt400","augmented_prompt507"]
for year in years:
  print("Code of the examination",year)
  Questions=read_excel_column(year+".xlsx", "Sheet1", "augmented_prompt")
  Answers=read_excel_column(year+".xlsx", "Sheet1", "correct_answer")
  #print(Questions)
  df = pd.DataFrame(columns=["Question","Correct_Answer",'Answer1', "Score1"])
  # for Question in Questions
  #  Questi
  for i in range(len(Questions)):
    #print(i)
  # system_message = input("What type of chatbot you want me to be?")
  # messages.append({"role":"system","content":system_message})
  #print("Alright! I am ready to be your friendly chatbot" + "\n" + "You can now type your messages.")
  #message = input("")      
     
      
     
      answers_from_model, history = model.chat(tokenizer, Questions[i], history=None)
      messages = [] 

     


      def final_score(Question,answers_from_model,answers):
          if "五个" in Question:
             score = score_of_multi_choice(answers_from_model, answers)
        # total_score = total_score + score
        #print("score of this question & total scores", score, total_score)
          if "四个" in Question:
              score = score_of_single_choice(answers_from_model, answers)
        # total_score = total_score + score
        #print("score of this question & total scores", score, total_score)
          return score
      answer1=answers_from_model

    
    # df2 = pd.DataFrame([
    # {"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer1,"Score1":final_score(Questions[i],answer1,Answers[i]),"Answer2":response_text_davinci_003.choices[0].text,"Score2":final_score(Questions[i],answer2,Answers[i])}])
      df2 = pd.DataFrame([
    {"Question":Questions[i],"Correct_Answer":Answers[i],"Answer1":answer1,"Score1":final_score(Questions[i],answer1,Answers[i])}])

    # print("No Question",i+1,"\n","Right_Answer:",Answers[i],"\n"+"Answer_from_text_davinci_003:",answer1.strip().replace("\n",""),"\nAnswer_from_:GPT3.5Turbo",answer2.strip().replace("\n",""),"\nAnswer_from_GPT4:",answer3.strip().replace("\n",""))
      print("No Question",i+1,"\n","Right_Answer:",Answers[i],"\nAnswer_from_Qwen-7B-Chat:",answer1.strip().replace("\n",""))
    #df2=pd.DataFrame([['Answer1',response_gpt35turbo["choices"][0]["message"]["content"], response_text_davinci_003.choices[0].text]])

    #row=[response_gpt35turbo["choices"][0]["message"]["content"],response_text_davinci_003.choices[0].text]
    #print (df)
    #print(df2)
      df = pd.concat([df, df2], axis=0)

  save_df_to_excel(df, "Answers_from_knowledge_Qwen-7B-Chat_EEE_"+year+".xlsx", "sheet1")
    #reply = response2["choices"][0]["message"]["content"]
    #print(row)
