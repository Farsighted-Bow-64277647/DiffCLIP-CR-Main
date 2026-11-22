import numpy as np
import torch
import tifffile
import clip
from PIL import Image

'''
CLIP scorer for cloud intensity vectors
Takes in TIF images which are converted
to PIL format, then plugs them through
CLIP Vision transformer models which "determine"
cloud intensity through text-image feature
similarities
'''

class CLIPinten:
    def __init__(self, device=None):    #CLIP model, variant, prompt setup
        self.device = "cuda" if (torch.cuda.is_available()) else "cpu"
        self.model, self.preprocessor = clip.load("ViT-B/32", self.device)
        self.model.eval()

        self.labels = ["little", "medium", "dense"]     #Prompts for Image-Embedding CLIP
        prompts = [ "A satellite picture of a low-intensity cloudy landscape",
        "A satellite picture of a medium-intensity cloudy landscape","A satellite picture of a high-intensity cloudy landscape"]
        text = clip.tokenize(prompts).to(self.device)   #Embedding prompts
        with torch.no_grad():
            self.text_feats = self.model.encode_text(text)
            self.text_feats /= self.text_feats.norm(dim=-1, keepdim = True)


    def _toPil(self, path):        #Convert imgs to PIL 
        img = tifffile.imread(path)
        img = img.astype(np.float32)

        if (img.ndim==3 and img.shape[-1] >= 3):
            img = img[:, :, :3]
        elif (img.ndim == 2):
            img = np.stack([img]  *   3, axis=-1)
        
        img_min, img_max = img.min(), img.max()
        if (img_max > img_min):
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)
        img  =  (img * 255).astype(np.uint8)


        return Image.fromarray(img, mode="RGB")

    @torch.no_grad()
    def scoring(self, imgPath):
        img_pil   =  self._toPil(imgPath)          #Image convert
        image = self.preprocessor(img_pil).unsqueeze(0).to(self.device) #Image preprocess (pre-encode)
        img_fts = self.model.encode_image(image)                 #Encode image w/ CLIP and extract feats
        img_fts /= img_fts.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_fts @ self.text_feats.T
        probs   =  logits.softmax(dim=-1).squeeze(0).cpu().numpy()      #Ret: probability vector/ argmax
        Class = int(probs.argmax())
        return Class, probs