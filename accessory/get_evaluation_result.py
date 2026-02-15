import ast
import json
import re
import statistics
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from rouge import Rouge

from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
# from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import os
import argparse


# from nltk.translate.meteor_score import meteor_score
# from nltk.tokenize import word_tokenize

#functions for evaluation of single phrase QA
def metrics_acc_fscore(result):
    """
    Calculate accuracy and macro-averaged F-score from prediction results.
    Args:
        result (list of dict): List of dictionaries, each containing 'answer' (predicted answer) and 'answer_gt' (ground truth answer).
    Returns:
        dict: A dictionary with keys 'accuracy' (float) and 'fscore' (float).
    """
    
    y_pred = [item['answer'] for item in result]   # pred answer
    y_true = [item['answer_gt'] for item in result]  # gt
    accuracy= accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division = 1)
    return {"accuracy": accuracy, "fscore": fscore}

def calc_metric_single_phrase(result):
    """"
    calculate acc and fscore for single phrase tasks
    """
    result_metrics_single_phrase={}
    single_phrase=[item for item in result if item['main_tag']=="single_phrase"]
    class_esd=[item for item in single_phrase if "CoPESD" in item['image']]
    
    class_endo=[item for item in single_phrase if "EndoVis" in item['image']]
    class_endo17=[item for item in class_endo if "EndoVis-17" in item['image']]
    class_endo18=[item for item in class_endo if "EndoVis-18" in item['image']]
    
    class_c80=[item for item in single_phrase if "C80" in item['image']]

    result_metrics_single_phrase['CoPESD'] = metrics_acc_fscore(class_esd)
    result_metrics_single_phrase['EndoVis17'] = metrics_acc_fscore(class_endo17)
    result_metrics_single_phrase['EndoVis18'] = metrics_acc_fscore(class_endo18)
    result_metrics_single_phrase['C80'] = metrics_acc_fscore(class_c80)

    return result_metrics_single_phrase 




#functions for evaluation of grounding task
def parse_bbox_string(s):
    """
    Parses a bounding box string from the input string using regex.
    Args:
        s (str): The input string that may contain a bounding box in the format [x1, y1, x2, y2].
    Returns:
        str: The matched bounding box string if found, otherwise '[-1,-1,-1,-1]' as default.
    """
    
    pattern = r"\[\s*-?\d*\.?\d+\s*,\s*-?\d*\.?\d+\s*,\s*-?\d*\.?\d+\s*,\s*-?\d*\.?\d+\s*\]"
    match = re.search(pattern, s)
    if match:
        try:
            return match.group()
        except (ValueError, SyntaxError):
            return "[-1,-1,-1,-1]"  # default value indicating parse failure
    else:
        return "[-1,-1,-1,-1]"

def IoU_single(box_a, box_b):
    """_summary_

    Args:
        box_a (_type_): pred bbox
        box_b (_type_): gt bbox

    Returns:
        _type_: iou score of one pair of bbox (gt and pred)
    """
    for i in range(0,len(box_a)):
        #set iou to 0 when parse failure of bbox string
        if box_a[i]<0:
            return 0
        # if box_a[i]>1:
        #     box_a[i]=box_a[i]/512
            # box_a[i]=0
    # print(box_a)
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x1 >= x2 or y1 >= y2:
        inter = 0.0
    inter = float((x2 - x1 + 1) * (y2 - y1 + 1))
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = box_a_area + box_b_area - inter
    iou = inter / union
    return iou


def metrics_ap50_miou(result):
    """_summary_

    Args:
        result (_type_): prediction result for grounding task

    Returns:
        _type_: AP@50 and mIoU for grounding task
    """
    
    # if len(result)==0:
    #     return 
    iou_list=[]
    err_cnt=0
    for item in result:
        answer=parse_bbox_string(item['answer'])
        answer_gt=item['answer_gt']
        if answer=="":
            iou_list.append(0)
            err_cnt+=1
            continue
        bbox=ast.literal_eval(answer)
        bbox_gt=ast.literal_eval(answer_gt)
        iou = IoU_single(bbox, bbox_gt)
        iou_list.append(iou)
    acc_50 = len([x for x in iou_list if x >= 0.5]) / len(iou_list)
    miou=sum(iou_list) / len(iou_list)
    return {"AP@50":acc_50,"miou":miou}
    
    
def calc_metric_grounding(result):
    """
    Calculate grounding metrics for different datasets from the result list.
    This function filters the input result list for items with 'main_tag' equal to 'grounding',
    then categorizes them into CoPESD, EndoVis-17, and EndoVis-18 based on the 'image' field.
    It computes AP50 and mIoU metrics for each category using the metrics_ap50_miou function.
    Args:
        result (list of dict): A list of dictionaries, each containing keys like 'main_tag' and 'image'.
    Returns:
        dict: A dictionary with keys 'CoPESD', 'EndoVis17', 'EndoVis18', each mapping to the computed metrics.
    """
    
    result_metrics_grounding={}
    
    grounding=[item for item in result if item['main_tag']=="grounding"]
    grounding_esd=[item for item in grounding if "CoPESD" in item['image']]
    grounding_endo17=[item for item in grounding if "EndoVis-17" in item['image']]
    grounding_endo18=[item for item in grounding if "EndoVis-18" in item['image']]

    result_metrics_grounding['CoPESD'] = metrics_ap50_miou(grounding_esd)
    result_metrics_grounding['EndoVis17'] = metrics_ap50_miou(grounding_endo17)
    result_metrics_grounding['EndoVis18'] = metrics_ap50_miou(grounding_endo18)
    
    return result_metrics_grounding


#functions for sentence answers evaluation (visual QA and region based QA)
def preprocess_text(text,split=False):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    if split:
        return tokens
    else:
        return text

def metrics_bleu(result):
    references = [[preprocess_text(item["answer_gt"],split=True)] for item in result]
    hypotheses = [preprocess_text(item["answer"],split=True) for item in result]

    metrics = {}
    metrics["Bleu_3"] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
    metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    return metrics

def metrics_cider(result):
    references = {i: [item["answer_gt"]] for i, item in enumerate(result)}
    hypotheses = {i: [item["answer"]] for i, item in enumerate(result)}
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, hypotheses)
    return {"CIDEr": cider_score}

def metrics_rouge(result):
    references = [item["answer_gt"] for item in result]
    hypotheses = [item["answer"] for item in result]
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(hypotheses, references, avg=True)
    return {"ROUGE-1": rouge_scores['rouge-1']['f'], "ROUGE-L": rouge_scores['rouge-l']['f']}

def metrics_meteor(result):
    references_meteor = {}
    hypotheses_meteor = {}
    
    for idx, entry in enumerate(result):
        qid = idx
        references_meteor[qid] = [entry['answer_gt']]
        hypotheses_meteor[qid] = [entry['answer']]
    # print(hypotheses_meteor)
    meteor_scorer = Meteor()
    meteor_score, score_per_instance = meteor_scorer.compute_score(references_meteor, hypotheses_meteor)
    return {"METEOR": meteor_score} 


def calc_metric_for_sentence_answer(result,tag):
    """
    calculate BLEU3, BLEU4, CIDER, METEOR, ROUGE-1 and ROUGE-L for sentence answers (visual QA and region based tasks)
    Returns:
        _type_: BLEU3,BLEU4,CIDER,METEOR,ROUGE-1,ROUGE-L
    """
    metrics_sentence={}

    bleu=metrics_bleu(result)
    metrics_sentence.update(bleu)

    cider=metrics_cider(result)
    metrics_sentence.update(cider)

    meteor=metrics_meteor(result)
    metrics_sentence.update(meteor)

    rouge=metrics_rouge(result)
    metrics_sentence.update(rouge)
    
    return metrics_sentence


def calculate_metric_for_visual_QA(result):
    """
    Calculates metrics for visual QA tasks by filtering results into categories (CoPESD, EndoVis-17, EndoVis-18, C80) 
    and computing metrics for each using calc_metric_for_sentence_answer.
    Args:
        result (list): List of result dictionaries containing 'main_tag' and 'image' keys.
    Returns:
        dict: Dictionary with keys 'CoPESD', 'EndoVis17', 'EndoVis18', 'C80' and their corresponding metric results.
    """
    
    visual_QA=[item for item in result if item['main_tag']=="visual_QA"]
    visual_esd=[item for item in visual_QA if "CoPESD" in item['image']]
    visual_endo17=[item for item in visual_QA if "EndoVis-17" in item['image']]
    visual_endo18=[item for item in visual_QA if "EndoVis-18" in item['image']]
    visual_c80=[item for item in visual_QA if "C80" in item['image']]
    
    metric_visual_QA={}
    result_esd=calc_metric_for_sentence_answer(visual_esd,"CoPESD")
    metric_visual_QA['CoPESD']=result_esd
    result_endo17=calc_metric_for_sentence_answer(visual_endo17,"EndoVis17")
    metric_visual_QA['EndoVis17']=result_endo17
    result_endo18=calc_metric_for_sentence_answer(visual_endo18,"EndoVis18")
    metric_visual_QA['EndoVis18']=result_endo18
    result_c80=calc_metric_for_sentence_answer(visual_c80,"C80")
    metric_visual_QA['C80']=result_c80
    return metric_visual_QA


def calculate_metric_for_region_based(result):
    """
    Calculate metrics for region-based items from the result list.
    This function filters the input result list for items with 'main_tag' equal to 'region_based',
    then categorizes them into three groups based on the 'image' field: CoPESD, EndoVis-17, and EndoVis-18.
    For each category, it computes metrics using calc_metric_for_sentence_answer and stores the results
    in a dictionary with keys 'CoPESD', 'EndoVis17', and 'EndoVis18'.
    Args:
        result (list): A list of dictionaries, each containing 'main_tag' and 'image' keys.
    Returns:
        dict: A dictionary with keys 'CoPESD', 'EndoVis17', 'EndoVis18', each mapping to the computed metric.
    """
    
    region_based=[item for item in result if item['main_tag']=="region_based"]
    region_esd=[item for item in region_based if "CoPESD" in item['image']]
    region_endo17=[item for item in region_based if "EndoVis-17" in item['image']]
    region_endo18=[item for item in region_based if "EndoVis-18" in item['image']]
    
    metric_region_based={}
    result_esd=calc_metric_for_sentence_answer(region_esd,"CoPESD")
    metric_region_based['CoPESD']=result_esd
    result_endo17=calc_metric_for_sentence_answer(region_endo17,"EndoVis17")
    metric_region_based['EndoVis17']=result_endo17
    result_endo18=calc_metric_for_sentence_answer(region_endo18,"EndoVis18")
    metric_region_based['EndoVis18']=result_endo18
    return metric_region_based



def calculate_metric_for_description(description_result_path):
    """
    Calculates average scores for CoPESD, EndoVis-17, and EndoVis-18 datasets from a JSON file.
    Args:
        description_result_path (str): Path to the JSON file with description results.
    Returns:
        dict: Dictionary with average scores for each dataset or None if no scores.
    """
    
    result_metrics_description={}
    with open(description_result_path, 'r') as f:
        description_result = json.load(f)
    
    description_result= [item for item in description_result if "score" in item and item['score'] is not None and type(item['score']) is int]

    esd_score=[item['score'] for item in description_result if "CoPESD" in item['image']]
    endo17_score=[item['score'] for item in description_result if "EndoVis-17" in item['image']]
    endo18_score=[item['score'] for item in description_result if "EndoVis-18" in item['image']]

    result_metrics_description['CoPESD']=statistics.mean(esd_score) if len(esd_score)>0 else None
    result_metrics_description['EndoVis17']=statistics.mean(endo17_score) if len(endo17_score)>0 else None
    result_metrics_description['EndoVis18']=statistics.mean(endo18_score) if len(endo18_score)>0 else None
    
    return result_metrics_description

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Calculate metrics for model results.')
    parser.add_argument('--model_test_result_folder', type=str, required=True, help='Path to the model test result folder')
    args = parser.parse_args()
    model_test_result_folder = args.model_test_result_folder
    # model_test_result_folder ="/mnt/data1/bailong/wgk/endochat_revise/test/result/Qwen2.5-VL-7B-Instruct-endochat"
    result_path= os.path.join(model_test_result_folder, "test_result.json")
    description_result_path=os.path.join(model_test_result_folder,"detailed_description_score.json")
    
    with open(result_path, 'r') as f:
        result = json.load(f)

    result_metrics={} #dict that store all metrics
    
    result=[item for item in result if item['answer'] is not None]
    for item in result:
        item['answer_gt']=item['answer_gt'].lower()
        item['answer_gt']=item['answer_gt'].replace("-"," ")
        item['answer_gt']=item['answer_gt'].replace("\n","") # remove \n in the gt answer, which may cause some problem for metric calculation
        if item['answer']!="":
            item['answer']=item['answer'].lower()
            item['answer']=item['answer'].replace("-"," ") #"left-top" and "left top" are equal answers
            item['answer']=item['answer'].replace("\n","") # remove \n in the answer, which may cause some problem for metric calculation
        
        if item['main_tag']=="single_phrase":
            item['answer']=item['answer'].replace(".","").replace("the ","")
            item['answer_gt']=item['answer_gt'].replace(".","").replace("the ","")


    print("start calculating metrics: single phrase")
    result_metrics['single_phrase']=calc_metric_single_phrase(result)
    
    print("start calculating metrics: grounding QA")
    result_metrics['grounding']=calc_metric_grounding(result)
    
    print("start calculating metrics: visual QA")
    result_metrics['visual_QA']=calculate_metric_for_visual_QA(result)
    
    print("start calculating metrics: region based QA")
    result_metrics['region_based_QA']=calculate_metric_for_region_based(result)
    
    print("start calculating metrics: detailed description")
    result_metrics['detailed_description']=calculate_metric_for_description(description_result_path)
    
    with open(os.path.join(model_test_result_folder, "evaluate_result_metrics.json"), 'w') as f:
        json.dump(result_metrics, f, indent=4)
    
    print(f"evaluate finished, result can be found in {model_test_result_folder}/evaluate_result_metrics.json")


