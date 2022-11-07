import os

import pandas as pd
from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption

class DiffusionDbTrain(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. diffusiondb/images/)
        ann_root (string): directory to store the annotation file
        '''        
        filename = 'diffusiondb_train.csv'

        self.annotation_df = pd.read_csv(os.path.join(ann_root, filename))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_names = {}
        n = 0
        for ix, ann in self.annotation_df.iterrows():
            img_name = ann['image_name']
            if img_name not in self.img_names:
                self.img_names[img_name] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation_df)
    
    def __getitem__(self, index):    
        
        ann = self.annotation_df.iloc[index]
        
        image_path = os.path.join(self.image_root,ann['image_name'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        prompt = self.prompt+pre_caption(ann['prompt'], self.max_words)

        return image, prompt, self.img_names[ann['image_id']]
    
    
class DiffusionDbEval(Dataset):
    def __init__(self, transform, image_root, ann_root, split):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val': 'diffusiondb_val.csv', 'test': 'diffusiondb_test.csv'}
        
        self.annotation_df = pd.read_csv(os.path.join(ann_root, filenames[split]))
        self.transform = transform
        self.image_root = image_root
        
    def __len__(self):
        return len(self.annotation_df)
    
    def __getitem__(self, index):    
        
        ann = self.annotation_df.iloc[index]
        
        image_path = os.path.join(self.image_root,ann['image_name'])
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          
        
        img_id = ann['image_name'][:-len(".png")]
        
        return image, int(img_id)   
