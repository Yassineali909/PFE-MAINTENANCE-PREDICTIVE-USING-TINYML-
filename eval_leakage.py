import os, glob, numpy as np, pandas as pd
from scipy.fft import fft
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
CSV="/mnt/hgfs/csv"; CLASSES=["ball","inner_race","normal","outer_race"]; WINDOW=1024
TEST_SUFFIX="_3"   # fichiers _3 = jeu de test (charge non vue)

def load():
    Xtr,ytr,Xte,yte=[],[],[],[]
    for ci,c in enumerate(CLASSES):
        for f in sorted(glob.glob(os.path.join(CSV,c,"*.csv"))):
            w=pd.read_csv(f)["vibration"].values[:WINDOW].astype(np.float32)
            if len(w)!=WINDOW: continue
            src=os.path.basename(f).split("_w")[0]
            if src.endswith(TEST_SUFFIX): Xte.append(w); yte.append(ci)
            else: Xtr.append(w); ytr.append(ci)
    return (np.array(Xtr),np.array(ytr),np.array(Xte),np.array(yte))

Xtr,ytr,Xte,yte=load()
print("Train: %d  Test(charge _3 non vue): %d" % (len(Xtr),len(Xte)))

def feats(w):
    rms=np.sqrt(np.mean(w**2));pk=np.max(np.abs(w));cr=pk/(rms+1e-9)
    m=np.mean(w);sd=np.std(w);sk=pd.Series(w).skew();ku=pd.Series(w).kurtosis()
    sp=np.abs(fft(w))[:len(w)//2];t5=np.sort(sp)[::-1][:5]
    return np.concatenate([[rms,pk,cr,m,sd,sk,ku],t5])
def norm(w): return (w-np.mean(w))/(np.std(w)+1e-6)

def run_tflite(path, Xin, reshape):
    it=tf.lite.Interpreter(path);it.allocate_tensors()
    inp,out=it.get_input_details()[0],it.get_output_details()[0]
    s,z=inp["quantization"];pr=[]
    for x in Xin:
        q=np.clip(np.round(x/s)+z,-128,127).astype(np.int8).reshape(reshape)
        it.set_tensor(inp["index"],q);it.invoke()
        pr.append(int(np.argmax(it.get_tensor(out["index"])[0])))
    return np.array(pr)

# MLP : features + scaler fit sur TRAIN
Ftr=np.array([feats(w) for w in Xtr]); Fte=np.array([feats(w) for w in Xte])
sc=StandardScaler().fit(Ftr)
mlp=run_tflite("/home/yassine/mlp_bearing_int8.tflite", sc.transform(Fte), (1,12))
# CNN : signal normalise brut
cnn=run_tflite("/home/yassine/cnn1d_bearing_int8.tflite", np.array([norm(w) for w in Xte]), (1,1024,1))
# GRU : signal normalise reshape 32x32
gru=run_tflite("/home/yassine/gru_bearing_int8.tflite", np.array([norm(w).reshape(32,32) for w in Xte]), (1,32,32))
lstm=run_tflite("/home/yassine/lstm_bearing_int8.tflite", np.array([norm(w).reshape(32,32) for w in Xte]), (1,32,32))

print("\n=== SPLIT PAR SOURCE (test = charge _3 jamais vue) ===")
for name,pr in [("MLP",mlp),("CNN",cnn),("GRU",gru),("LSTM",lstm)]:
    print("%-5s acc=%.4f  f1=%.4f" % (name, accuracy_score(yte,pr), f1_score(yte,pr,average="macro")))
    print(confusion_matrix(yte,pr))
