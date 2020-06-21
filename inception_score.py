import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import os
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from PIL import Image
import numpy as np
from scipy.stats import entropy
import argparse
   

def get_imgs(img_path, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')    

    if transform is not None:
        img = transform(img)

    if normalize is not None:
        img = normalize(img)

    return img


def inception_score(imgs, args, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, num_workers=args.workers)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.float().cuda()
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, path,img_transforms):
        self.transform = img_transforms
        if path.find('birds') != -1 or path.find('flowers') != -1:
            class_names = os.listdir(path)
            image_paths = []
            for class_name in sorted(class_names):
                class_path = os.path.join(path,class_name)
                img_names = os.listdir(class_path)
                for img_name in sorted(img_names):
                    img_path = os.path.join(class_path,img_name)
                    image_paths.append(img_path)
            self.files = image_paths
        else:
            image_paths=[]
            for maindir, subdir, file_name_list in os.walk(path):
                for filename in file_name_list:
                    apath = os.path.join(maindir, filename)
                    image_paths.append(apath)
            self.files = image_paths

    def __getitem__(self, index):
        img_path = self.files[index]
        img = get_imgs(img_path,self.transform)
        
        return img

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate the inception score')
    
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='outputs/test/gan/TestImage')    
    parser.add_argument('--exp_dir', dest='exp_dir', type=str, default='outputs/test/gan')   
    parser.add_argument('--workers',dest='workers',type=int,default=0)
    args = parser.parse_args()   
    

    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    
    root_path = args.data_dir
    epoch_names = os.listdir(root_path)
    save_path = os.path.join(args.exp_dir, 'Inception_score.text')
    info = 'starting evalute the inception score \n'
    with open(save_path, "a") as file:
        file.write(info)
    for epoch_name in epoch_names:
        epoch = int(epoch_name[5:])
        path = os.path.join(root_path,epoch_name)    
        img_transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        data = IgnoreLabelDataset(path,img_transforms)

        # print ("Calculating Inception Score...")
        IS = inception_score(data, args, cuda=True, batch_size=32, resize=True, splits=10)
        info = ' Epoch: [{0}]  IS_mean: {IS_m:.4f} IS_std: {IS_s:.4f} \n'.format(epoch,IS_m=IS[0],IS_s=IS[1])
        print (info)          
        
        with open(save_path, "a") as file:
            file.write(info)



