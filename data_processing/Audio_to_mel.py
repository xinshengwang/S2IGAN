import numpy as np
import librosa
import os

def audio_processing(input_file):
    
    y = input_file
    sr = 22050
    window_size = 25
    stride = 10
    input_dim = 40
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = input_dim, n_fft=ws, hop_length=st)
    feat = np.log(feat + 1e-6)

    feat = [feat]
    cmvn = True

    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]

    return np.swapaxes(feat, 0, 1).astype('float32')  

root = 'audio_npy'
save_root = 'audio_mel'
clss_names = os.listdir(root)
for clss_name in sorted(clss_names):
    clss_path = os.path.join(root,clss_name)
    img_names = os.listdir(clss_path)    
    for img_name in sorted(img_names):
        name = img_name.split('.')[0]
        audio_path = os.path.join(clss_path,img_name)        
        audios =np.load(audio_path,allow_pickle=True)
        mels = []
        for audio in audios:            
            mel = audio_processing(audio)            
            mels.append(mel)        
        save_dir = os.path.join(save_root,clss_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + '/' + name +'.npy'
        np.save(save_path,mels)
    print(clss_name)