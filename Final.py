#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import time
from torch import nn, optim
import collections
import csv
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(device)

def loadData(filename,windows):
    print('Data Reading>>>>>>>>>>>>>>>')
    filename_his = filename+'histone_data.csv'
    print('The path of histone_data',filename_his)
    with open(filename_his) as fi_his:
        csv_reader=csv.reader(fi_his)
        data=list(csv_reader)
        ncols=(len(data[0]))
    fi_his.close()

    nrows=len(data)
    ngenes=nrows/windows
    nfeatures=ncols-1
    print("Number of genes: %d" % ngenes)
    print("Number of entries: %d" % nrows)
    print("Number of HMs: %d" % nfeatures)

    filename_gene = filename+'gene_expression.csv'
    print('The path of gene_expression',filename_gene)
    with open(filename_gene) as fi_gene:
        csv_reader=csv.reader(fi_gene)
        data_gene=list(csv_reader)
        ncols=(len(data_gene[0]))
    fi_gene.close()
    count=0
    attr=collections.OrderedDict()

    
    for i in range(0,nrows-1,windows):
        hm1=torch.zeros(windows,1)
        hm2=torch.zeros(windows,1)
        hm3=torch.zeros(windows,1)
        hm4=torch.zeros(windows,1)
        hm5=torch.zeros(windows,1)
        for w in range(0,windows):
            hm1[w][0]=int(data[i+1+w][1])
            hm2[w][0]=int(data[i+1+w][2])
            hm3[w][0]=int(data[i+1+w][3])
            hm4[w][0]=int(data[i+1+w][4])
            hm5[w][0]=int(data[i+1+w][5])

        geneID=str(data[i+1][0].split("_")[0])
        thresholded_expr = int(data_gene[int(i/100)+1][1])
        attr[count]={
            'geneID':geneID,
            'expr':thresholded_expr,
            'hm1':hm1,
            'hm2':hm2,
            'hm3':hm3,
            'hm4':hm4,
            'hm5':hm5
        }
        count+=1
        
    return attr


class HMData(Dataset):
    def __init__(self,data_cell1,transform=None):
        self.c1=data_cell1
    def __len__(self):
        return len(self.c1)
    def __getitem__(self,i):
        final_data_c1=torch.cat((self.c1[i]['hm1'],self.c1[i]['hm2'],self.c1[i]['hm3'],self.c1[i]['hm4'],self.c1[i]['hm5']),1)
        label=self.c1[i]['expr']
        geneID=self.c1[i]['geneID']
        final_data_c1 = final_data_c1.reshape(1, final_data_c1.shape[0], final_data_c1.shape[1])
        sample={'geneID':geneID,
               'input':final_data_c1,
               'label':label,
               }

        return sample
    
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 20, 5), 
            nn.BatchNorm2d(20),
            nn.ReLU(),
            Reshape(1,20,96),  
            nn.MaxPool2d((1,3), (1,3)), 
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.8),            
            nn.Linear(1*20*32, 120),
            nn.BatchNorm1d(120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 2)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        output = torch.softmax(output,dim = 1)
        return output
    

    
def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    best_loss = float('inf')    

    for epoch in range(num_epochs):
        batch_count = 0
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for XX in train_iter:               
            X = XX['input'].to(device)
            y = XX['label'].to(device) 
            X= X
            y_hat = net(X)   
            if epoch >(num_epochs - 1):
                ans = []
                for t in y_hat:
                    if t[0]>t[1]:
                        ans.append(0)
                    else:ans.append(1)
            l = loss(y_hat, y)
            if l < best_loss:
                best_loss = l
                torch.save(net, 'best_model.pth') 
                
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                        
            if epoch >8:
                print('yhat', y_hat.argmax(dim=1))
                print('y is',y)
                print('geneID is',XX['geneID'])
                print('train_acc_sum is ',train_acc_sum)
                print('input matrix is',XX['input'])
            n += y.shape[0]
            batch_count += 1
            
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


        
        
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for XX in data_iter:              
            X = XX['input'].to(device)
            y = XX['label'].to(device)            

            if isinstance(net, torch.nn.Module):
                net.eval() 
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() 
            else: 
                if('is_training' in net.__code__.co_varnames):                     
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n


# In[2]:


cell_train_dict1=loadData("C:/jupyter/hISTONE/his/",100)
print('Concatenating histone modification and label>>>>>>>>>>>>')
full_dataset = HMData(cell_train_dict1)
print('The length of total dataset',len(full_dataset))


train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size


train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
print('The number of train dataset is:',len(train_dataset))
print('The number of test dataset is:',len(test_dataset))


batch_size = 80
print('Batch size is', batch_size)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

    
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)



net = Net()
print('Net structure:')
print(net)
lr, num_epochs = 0.001, 10
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_ch5(net, train_loader, test_loader, batch_size, optimizer, device, num_epochs)

print('Plesae find the saved net parameters at  C:/jupyter/hISTONE/best_model.pth')

