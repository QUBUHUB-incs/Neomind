# =====================================================
# NeoMind Minimal – Low-end hardware version
# Author: Seriki Yakub
# =====================================================

import os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, random, re, time, glob

# ------------------------------
# 1️⃣ Minimal repo list
# ------------------------------
REPO_URLS = [
    "https://github.com/Web4application/EDQ-AI",
    "https://github.com/Web4application/Brain"
]

def clone_repos(base_dir="repos"):
    os.makedirs(base_dir, exist_ok=True)
    for url in REPO_URLS:
        name = url.rstrip("/").split("/")[-1]
        path = os.path.join(base_dir, name)
        if not os.path.exists(path):
            os.system(f"git clone {url} {path}")
        else:
            os.system(f"cd {path} && git pull")

clone_repos()

# ------------------------------
# 2️⃣ Numeric features (lines, words, chars)
# ------------------------------
def extract_numeric(repo_path):
    features=[]
    for r,_,files in os.walk(repo_path):
        for f in files:
            if f.endswith(('.txt','.md','.py','.json')):
                try:
                    content=open(os.path.join(r,f),'r',errors='ignore').read()
                    features.append([content.count('\n'), len(content.split()), len(content)])
                except: pass
    return features

def build_numeric_data():
    data=[]
    for url in REPO_URLS:
        name = url.rstrip("/").split("/")[-1]
        path = os.path.join("repos", name)
        feats = extract_numeric(path)
        for f in feats:
            vec = f + [0]*(8-len(f)) if len(f)<8 else f[:8]
            data.append(vec)
    return torch.tensor(data, dtype=torch.float)

numeric_data = build_numeric_data()

# ------------------------------
# 3️⃣ Text data (tokenized)
# ------------------------------
def read_texts(repo_path):
    texts=[]
    for r,_,files in os.walk(repo_path):
        for f in files:
            if f.endswith(('.txt','.md','.py','.json')):
                try: texts.append(open(os.path.join(r,f),'r',errors='ignore').read())
                except: pass
    return texts

def tokenize(texts, vocab=None, max_len=16):
    if vocab is None: vocab={}
    tokenized=[]
    for t in texts:
        words=re.findall(r'\b\w+\b',t.lower())
        encoded=[vocab.setdefault(w,len(vocab)+1) for w in words]
        tokenized.append(encoded[:max_len])
    return tokenized, vocab

def build_text_data():
    all_texts=[]
    for url in REPO_URLS:
        name=url.rstrip("/").split("/")[-1]
        path=os.path.join("repos",name)
        all_texts.extend(read_texts(path))
    tokenized,vocab=tokenize(all_texts)
    vocab_size=len(vocab)+1
    max_len=max(len(t) for t in tokenized)
    text_data=torch.zeros(len(tokenized), max_len, dtype=torch.long)
    for i,t in enumerate(tokenized):
        text_data[i,:len(t)]=torch.tensor(t)
    return text_data,vocab_size

text_data,vocab_size=build_text_data()

# ------------------------------
# 4️⃣ Targets (self-supervised)
# ------------------------------
targets=numeric_data.clone()

# ------------------------------
# 5️⃣ Minimal NeoMind network
# ------------------------------
class EDQBranch(nn.Module):
    def __init__(self): super().__init__(); self.l1=nn.Linear(8,32); self.l2=nn.Linear(32,16)
    def forward(self,x): return F.gelu(self.l2(F.gelu(self.l1(x))))

class BrainBranch(nn.Module):
    def __init__(self,vocab_size): super().__init__(); self.embed=nn.Embedding(vocab_size,16,padding_idx=0); self.rnn=nn.GRU(16,16,batch_first=True)
    def forward(self,x): _,h=self.rnn(self.embed(x)); return F.gelu(h.squeeze(0))

class NeoMindMinimal(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.edq=EDQBranch()
        self.brain=BrainBranch(vocab_size)
        self.fuse=nn.Linear(16+16,16)
    def forward(self,x_num,x_text): return self.fuse(torch.cat([self.edq(x_num), self.brain(x_text)],dim=1))

# ------------------------------
# 6️⃣ Versioned weight saving
# ------------------------------
def save_weights(model,base="NeoMind_minimal"):
    existing=glob.glob(f"{base}_v*.pth")
    if existing: versions=[int(re.search(r'_v(\d+)\.pth',f).group(1)) for f in existing]; new_v=max(versions)+1
    else: new_v=1
    fn=f"{base}_v{new_v}.pth"
    torch.save(model.state_dict(),fn); print(f"✅ Weights saved: {fn}"); return fn

# ------------------------------
# 7️⃣ Training
# ------------------------------
def train(model,X_num,X_txt,Y,epochs=5,batch_size=8,lr=1e-3):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); opt=optim.Adam(model.parameters(),lr=lr); loss_fn=nn.MSELoss()
    n=X_num.size(0); idx=list(range(n))
    for ep in range(epochs):
        random.shuffle(idx); epoch_loss=0
        for i in range(0,n,batch_size):
            b=idx[i:i+batch_size]; xb_num=X_num[b].to(device); xb_txt=X_txt[b].to(device); yb=Y[b].to(device)
            opt.zero_grad(); out=model(xb_num,xb_txt); loss=loss_fn(out,yb); loss.backward(); opt.step(); epoch_loss+=loss.item()*xb_num.size(0)
        print(f"Epoch {ep+1} | Loss: {epoch_loss/n:.6f}")
    save_weights(model)
    return model

# ------------------------------
# 8️⃣ Self-updating loop
# ------------------------------
CHECK_INTERVAL=600
def update_loop(model,X_num,X_txt,Y):
    while True:
        updated=False
        for url in REPO_URLS:
            name=url.rstrip("/").split("/")[-1]
            path=os.path.join("repos",name)
            pull=os.system(f"cd {path} && git pull")
            if pull==0: updated=True
        if updated:
            print("🔄 Changes detected. Retraining...")
            numeric_data_new=build_numeric_data()
            text_data_new,_=build_text_data()
            train(model,numeric_data_new,text_data_new,Y,epochs=3)
        else: print("⏳ No changes detected.")
        time.sleep(CHECK_INTERVAL)

# ------------------------------
# 9️⃣ Entrypoint
# ------------------------------
if __name__=="__main__":
    model=NeoMindMinimal(vocab_size)
    model=train(model,numeric_data,text_data,targets,epochs=5)
    update_loop(model,numeric_data,text_data,targets)
