
import numpy as np




def rank_test(predict_file,label_file):
    predict_score = {}
    label_score = {}
    f1 = open(predict_file,'r')
    f2 = open(label_file,'r')
    
    for line in f1.readlines():
        line = line.strip().split()
        img_name = line[0]
        img_score = line[1]
        predict_score[img_name] = float(img_score)

    for line in f2.readlines():
        line = line.strip().split()
        img_name = line[0]
        img_score = line[1]
        label_score[img_name] = float(img_score)
    #print('predict: ',predict_score)
    #print('label: ',label_score)

    img_name_list = list(label_score.keys())
    img_name_list.sort()
    
    count = 0
    positive = 0 
    for i in range(len(img_name_list)):
        for j in range(i+1,len(img_name_list)):
            real_rank = 1 if label_score[img_name_list[i]] >= label_score[img_name_list[j]] else -1
            #print('label: ',img_name_list[i],img_name_list[j],real_rank)
            
            predict_rank = 1 if predict_score[img_name_list[i]] >= predict_score[img_name_list[j]] else -1
            
            #print('predict: ',img_name_list[i],predict_score[img_name_list[i]],img_name_list[j],predict_score[img_name_list[j]],predict_rank)
            #print('***************')
            count += 1
            if real_rank == predict_rank:
                positive += 1
            '''
            else:
                print('label: ',img_name_list[i],label_score[img_name_list[i]],img_name_list[j],label_score[img_name_list[j]],real_rank)
                print('predict: ',img_name_list[i],predict_score[img_name_list[i]],img_name_list[j],predict_score[img_name_list[j]],predict_rank)
            '''
    print('%d/%d'%(positive,count))
    accuracy = positive/count
    print('Accuracy: %f'%accuracy)
    return accuracy

def rank_pair_test(predict_file,label_file):
    predict_score = {}
    label_score = {}
    f1 = open(predict_file,'r')
    f2 = open(label_file,'r')
    
    for line in f1.readlines():
        line = line.strip().split()
        img_name = line[0]
        img_score = line[1]
        predict_score[img_name] = float(img_score)

    for line in f2.readlines():
        line = line.strip().split()
        img_name = line[0]
        img_score = line[1]
        label_score[img_name] = float(img_score)
    
    keys_list = list(label_score.keys())
    keys_list.sort()
    
    cursor = keys_list[0].split('_')[0]
    class_num = 0
    for key in keys_list:      
        if cursor == key.split('_')[0]:
            class_num += 1
        else:
            break
    
    count = 0
    positive = 0
    for idx in range(0,len(keys_list),class_num):
        for i in range(idx,idx+class_num):
            for j in range(i+1,idx+class_num):
                #print(idx,i,j)
                
                real_rank = 1 if label_score[keys_list[i]] >= label_score[keys_list[j]] else -1
                #print('label: ',img_name_list[i],img_name_list[j],real_rank)
            
                predict_rank = 1 if predict_score[keys_list[i]] >= predict_score[keys_list[j]] else -1
            
                #print('predict: ',img_name_list[i],predict_score[img_name_list[i]],img_name_list[j],predict_score[img_name_list[j]],predict_rank)
                #print('***************')
                count += 1
                if real_rank == predict_rank:
                    positive += 1
            
                else:
                    print('label: ',keys_list[i],label_score[keys_list[i]],keys_list[j],label_score[keys_list[j]],real_rank)
                    print('predict: ',keys_list[i],predict_score[keys_list[i]],keys_list[j],predict_score[keys_list[j]],predict_rank)
                
    
    print('%d/%d'%(positive,count))
    accuracy = positive/count
    print('Accuracy: %f'%accuracy)
    return accuracy
        
    
accuracy = rank_test('predict_score.txt','label_score.txt')
rank_pair_test('predict_score.txt','label_score.txt')
