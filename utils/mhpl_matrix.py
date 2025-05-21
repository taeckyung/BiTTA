import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import random
import conf

device = torch.device("cuda:{:d}".format(conf.args.gpu_idx) if torch.cuda.is_available() else "cpu")

def obtain_label(loader, net): # one-shot querying using src-model
    # hyper-params #
    threshold = 0
    distance = "cosine"
    KK = conf.args.mhpl_kk
    epsilon = 1e-5
    ratio = 0.05
    e_n = 5
    alpha = conf.args.mhpl_alpha
    beta = 0.3
    
    h_dict = {}
    loc_dict = {}
    fea_sel_dict = {}
    label_sel_dict = {}
    for cls in range(conf.args.opt['num_class']):
        h_dict[cls] = []
        loc_dict[cls] = []
        fea_sel_dict[cls] = []
        label_sel_dict[cls] = []

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader) # test-dataloader
        # sel_path = iter_test.dataset.imgs
        for _ in range(len(loader)):
            # data = iter_test.next()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            # feas = netB(netF(inputs))
            # feas_uniform = F.normalize(feas)
            # outputs = netC(feas)
            outputs, feas = net[1](net[0](inputs), get_embedding = True)
            feas_uniform = F.normalize(feas)
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)  
    con, predict = torch.max(all_output, 1)
    accuracy_ini = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])


    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > threshold)
    labelset = labelset[0]
    

    dd = cdist(all_fea, initc[labelset], distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    
    accuracy = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    if(len(labelset) < conf.args.opt['num_class']): 
        print("missing classes") 

    #neighbor retrieve
    distance = torch.tensor(all_fea) @  torch.tensor(all_fea).t()
    dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=KK)
    #get the labels of neighbors
    near_label = torch.tensor(pred_label)[idx_near]
    

    #neighbor affinity
    dis_near = dis_near[:,1:]
    neigh_dis = []
    for index in range(len(pred_label)):
        neigh_dis.append(np.mean(np.array(dis_near[index])))
    neigh_dis = np.array(neigh_dis)

    
    pro_clu_near = []
    for index in range(len(near_label)):
        label = np.zeros(conf.args.opt['num_class'])
        count = 0
        for cls in range(conf.args.opt['num_class']):
            cls_filter = (near_label[index] == cls)
            list_loc = cls_filter.tolist()
            list_loc = [i for i,x in enumerate(list_loc) if x ==1 ]
            list_loc = torch.Tensor(list_loc)
            pro = len(list_loc)/len(near_label[index])
            label[cls] = pro
            count += len(list_loc)
            if (count == len(near_label[index])):
                break
        pro_clu_near.append(label)
    # class probability distribution space
    pro_clu_near = torch.tensor(np.array(pro_clu_near))

    #neighbor purity
    ent = torch.sum( - pro_clu_near  * torch.log( pro_clu_near + epsilon), dim=1)
    ent = ent.float()

    closeness = torch.tensor(neigh_dis)
    #neighbor ambient uncertainty
    stand = (- ent) *closeness
    
    loc_label = []
    true_label = []
    binary_mask = []
    sor = np.argsort(stand)
    index = 0
    index_v = 0
    # m active selected samples, ratio
    # SSN =  int(len(pred_label) * ratio)
    SSN = len(loader) * conf.args.ass_num

    while index < SSN:
        near_i = -1
        r_i = sor[index_v]
        idx_ri_near = idx_near[r_i]
        flag_near = False
        #neighbor diversity relaxation
        idx_fir_twi = idx_ri_near[0: e_n]
        for p_i in range (len(idx_fir_twi)):
            if r_i == idx_fir_twi[p_i]:
                continue
            else:
              near_i = idx_fir_twi[p_i]  
            if near_i in loc_label:
                # print("pass")
                index_v = index_v +1
                flag_near = True
                break
        if (flag_near == True):
            continue
        loc_label.append(r_i)
        
        if conf.args.turn_to_binary:
            if pred_label[r_i] == all_label[r_i]:
                binary_mask += [True]
            else:
                binary_mask += [False]
            true_label.append(int(pred_label[r_i]))
        else:
            true_label.append(int(all_label[r_i]))
            pred_label[r_i] = all_label[r_i] # make pred -> true label
        index = index + 1
        index_v = index_v +1
        


    print("needed labeled")
    print(len(loc_label))
    
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}% -> {:.2f}%'.format(accuracy_ini * 100, accuracy * 100, acc * 100)
    print(log_str)
    # args.out_file.write(log_str+'\n')
    # args.out_file.flush()

    pred_true = []
    weight = []
    stand = ent

    for index in range(len(pred_label)):
        label = np.zeros(conf.args.opt['num_class']) # for constructing one-hot encoder
        if index in loc_label: # index of gt
            label[true_label[loc_label.index(index)]] = 1.0
            weight.append(stand[index].tolist()* alpha)
        else:
            label[pred_label[index]] = 1.0
            weight.append(beta)
        pred_true.append(label)

    return pred_true,loc_label,true_label,weight, binary_mask

def obtain_label_rectify(loader, net,loc_label,true_label):
    # hyper-params #
    threshold = 0
    distance = "cosine"
    alpha = 25.0
    beta_af = 0.0
    
    h_dict = {}
    loc_dict = {}
    fea_sel_dict = {}
    label_sel_dict = {}
    for cls in range(conf.args.opt['num_class']):
        h_dict[cls] = []
        loc_dict[cls] = []
        fea_sel_dict[cls] = []
        label_sel_dict[cls] = []

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            # data = iter_test.next()
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            # feas = netB(netF(inputs))
            # feas_uniform = F.normalize(feas)
            # outputs = netC(feas)
            outputs, feas = net[1](net[0](inputs), get_embedding = True)
            feas_uniform = F.normalize(feas)
            
            if start_test:
                all_fea = feas_uniform.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas_uniform.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    con, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        
    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    
    labelset = np.where(cls_count>threshold)
    labelset = labelset[0]
    dd = cdist(all_fea, initc[labelset], distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format( accuracy * 100,acc * 100)
    print(log_str)
    # args.out_file.write(log_str+'\n')
    # args.out_file.flush()
    # print(len(loc_label))
    pred_true = []
    weight = []
    for index in range(len(pred_label)):
        label = np.zeros(conf.args.opt['num_class'])
        if index in loc_label:
            label[true_label[loc_label.index(index)]] = 1.0
            weight.append(alpha)
        else:
            label[pred_label[index]] = 1.0
            weight.append(beta_af)
        pred_true.append(label) 
    return pred_true,weight

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer
    
if __name__ == "__main__":
    pass