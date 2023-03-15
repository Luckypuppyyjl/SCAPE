import torch  # 命令行是逐行立即执行的

content = torch.load('/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/checkpoint-99-model1.pth')
dictt=content

i=-1
for key in list(dictt.keys()):
    i = i + 1
    print(i)
    #print(key)
    if 'attn.kv' in key:
        key_n=key.replace('kv', 'k_a',1)
        size_=dictt[key].size()[0]
        if  'bias' in key:
            dictt[key_n] = dictt[key][:int(size_ / 2)].clone()
        else:
           dictt[key_n]=dictt[key][:int(size_/2),:].clone()
        i=i+1
        print(key)

    #if i >= 215 and i <= 348:
          #key_new = key[14:]
          #dictt[key_new] = dictt.pop(key)
    #  continue
    #else:
    #    del dictt[key]

torch.save(dictt, '/data/yjliang/code/Category-Agnostic-Pose-Estimation/P2/Pose-for-Everything/work_dirs/checkpoint-99-model2.pth', _use_new_zipfile_serialization=False)