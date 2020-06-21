import os 
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import pickle
from steps.pre_util import *
import pdb

def train(Models,train_loader, test_loader, args):    
    if cfg.DATA_DIR.find('birds') != -1 or cfg.DATA_DIR.find('flowers') != -1:
        audio_model, image_cnn,image_model,class_model = Models[0],Models[1],Models[2],Models[3]
    else:
        audio_model, image_cnn,image_model = Models[0],Models[1],Models[2]
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    # exp_dir = os.path.join(args.save_root,'pre_train') 
    exp_dir = args.save_root
    save_model_dir = os.path.join(exp_dir,'models')
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # create/load exp
    """
    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_acc = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)
    """

    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if not isinstance(image_cnn, torch.nn.DataParallel):
        image_cnn = nn.DataParallel(image_cnn)

    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
        if not isinstance(class_model, torch.nn.DataParallel):
            class_model = nn.DataParallel(class_model)

        
    epoch = 0
    
    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model_%d.pth" % (exp_dir, epoch)))
        image_model.load_state_dict(torch.load("%s/models/image_model.%d.pth" % (exp_dir, epoch)))
        if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
            class_model.load_state_dict(torch.load("%s/models/class_model.%d.pth" % (exp_dir, epoch)))        
        print("loaded parameters from epoch %d" % epoch)
    

    audio_model = audio_model.to(device)
    image_cnn = image_cnn.to(device)
    image_model = image_model.to(device)
    if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
        class_model = class_model.to(device)    
    # Set up the optimizer
    # audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    image_trainables = [p for p in image_model.parameters() if p.requires_grad] # if p.requires_grad
    
    trainables = audio_trainables + image_trainables
    if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
        class_trainables = [p for p in class_model.parameters() if p.requires_grad]
        trainables += class_trainables
    
    
    if cfg.Loss.deco:
        if not isinstance(deco_model, torch.nn.DataParallel):
            deco_model = nn.DataParallel(deco_model)        
        if epoch != 0:
            deco_model.load_state_dict(torch.load("%s/models/deco_model.%d.pth" % (exp_dir, epoch)))
        deco_model = deco_model.to(device)
        deco_trainables = [p for p in deco_model.parameters() if p.requires_grad]        
        image_trainables = image_trainables + deco_trainables
    
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(trainables, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)
    
    
    
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    # criterion = nn.MSELoss()    
    image_cnn.eval()
    criterion_c = nn.CrossEntropyLoss()    
     
    while epoch<=cfg.TRAIN.MAX_EPOCH:
        epoch += 1
        adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
        end_time = time.time()
        audio_model.train()
        image_model.train()
        if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
            class_model.train()
        
        for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(train_loader):
                        
            # measure data loading time
            data_time.update(time.time() - end_time)
            B = audio_input.size(0)

            audio_input = audio_input.float().to(device)            
            label = label.long().to(device)
            input_length = input_length.float().to(device)
            
            image_input = image_input.float().to(device)
            image_input = image_input.squeeze(1)            
            
            optimizer.zero_grad()

            image_feat = image_cnn(image_input)
            image_output = image_model(image_feat)
            # image_class_output = class_model(image_output)                       
            
            if cfg.SPEECH.model == 'CNN':
                audio_output = audio_model(audio_input)
            else:
                audio_output = audio_model(audio_input,input_length)
            
            if cfg.Loss.clss:
                image_class_output = class_model(image_output) 
                audio_class_output = class_model(audio_output)
                
            loss = 0     
            # batch loss is the matching loss in S2IGAN       
            lossb1,lossb2 = batch_loss(image_output,audio_output,cls_id)
            loss_batch = lossb1 + lossb2
            loss += loss_batch*cfg.Loss.gamma_batch

            # distinctive loss 
            if cfg.Loss.clss:
                loss_c = criterion_c(image_class_output,label) + criterion_c(audio_class_output,label) 
                loss += loss_c*cfg.Loss.gamma_clss

            loss.backward()
            optimizer.step()        

            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            
            if i % 5 == 0:
                # mAP = validate(audio_model, image_model, test_loader)
                print('iteration = %d | loss = %f '%(i,loss))
            """
            if i % 50 == 0:
                recalls = validate(audio_model, image_model, test_loader, args)
            """
            
            #printer
            """
            if global_step % args.n_print_steps == 0 and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss total {loss_meter.val:.4f} ({loss_meter.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return
            """


            end_time = time.time()
            global_step += 1
            
        if epoch % 5 ==0:
            if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
                mAP = validate(audio_model, image_model,image_cnn ,test_loader)
                avg_acc = mAP
                info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f}  mAP: {mAP_:.4f} \n \
                '.format(epoch,loss_meter=loss_meter,mAP_=mAP)
            
            else:
                recalls = validate_all(audio_model, image_model,image_cnn,test_loader,args)
                A_r10 = recalls['A_r10']
                I_r10 = recalls['I_r10']
                A_r5 = recalls['A_r5']
                I_r5 = recalls['I_r5']
                A_r1 = recalls['A_r1']
                I_r1 = recalls['I_r1']
                medr_I2A = recalls['medr_I2A']
                medr_A2I = recalls['medr_A2I']
                avg_acc = (A_r10 + I_r10)/2
                info = ' Epoch: [{0}] Loss: {loss_meter.val:.4f} | \
                *Audio:R@1 {A_r1:.4f} R@5 {A_r5:.4f} R@10 {A_r10:.4f} medr {A_m:.4f}| *Image R@1 {I_r1:.4f} R@5 {I_r5:.4f} R@10 {I_r10:.4f} \
               medr {I_m:.4f} \n'.format(epoch,loss_meter=loss_meter,A_r1=A_r1,A_r5=A_r5, A_r10=A_r10,A_m=medr_A2I,I_r1=I_r1, I_r5=I_r5, I_r10=I_r10, I_m=medr_I2A)  
            print (info)

            
            
            save_path = os.path.join(exp_dir, cfg.result_file)
            with open(save_path, "a") as file:
                file.write(info)
            
            if avg_acc > best_acc:
                best_epoch = epoch
                best_acc = avg_acc

                torch.save(audio_model.state_dict(),
                    "%s/models/best_audio_model.pth" % (exp_dir))
                torch.save(image_model.state_dict(),
                    "%s/models/best_image_model.pth" % (exp_dir))
                if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
                    torch.save(class_model.state_dict(),
                        "%s/models/best_class_model.pth" % (exp_dir))
                torch.save(optimizer.state_dict(), "%s/models/optim_state.pth" % (exp_dir))

            _save_progress()


# mAP: evaltuion for the embedding with a cross modal retreival task
def validate(audio_model, image_model,image_cnn,val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()
    

    if cfg.TRAIN.MODAL=='co-train':
        if not isinstance(image_model, torch.nn.DataParallel):
            image_model = nn.DataParallel(image_model)
        
        image_model = image_model.to(device)
        # switch to evaluate mode
        image_model.eval()
    
    

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    class_ids = []
    with torch.no_grad():
        
        for i, (image_input, audio_input, cls_id, key, input_length,label) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            image_input = image_input.float().to(device)
            input_length = input_length.float().to(device)

            # compute output
            if cfg.TRAIN.MODAL == 'co-train':
                image_feat = image_cnn(image_input)
                image_output = image_model(image_feat)   
            else:
                image_output = image_input
            
            if cfg.SPEECH.model == 'CNN':
                audio_output = audio_model(audio_input)
            else:
                audio_output = audio_model(audio_input,input_length)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)       
            class_ids.append(cls_id)     
            
            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        cls_id = torch.cat(class_ids)
     
        if cfg.DATASET_NAME == 'birds' or cfg.DATASET_NAME == 'flowers':
            recalls  = calc_mAP(image_output,audio_output,cls_id)
        else:
            recalls = calc_recalls(image_output, audio_output)        
            A_r10 = recalls['A_r10']
            I_r10 = recalls['I_r10']
            A_r5 = recalls['A_r5']
            I_r5 = recalls['I_r5']
            A_r1 = recalls['A_r1']
            I_r1 = recalls['I_r1']

            print(' * Audio R@10 {A_r10:.3f} Image R@10 {I_r10:.3f} over {N:d} validation pairs'
                .format(A_r10=A_r10, I_r10=I_r10, N=N_examples), flush=True)
            print(' * Audio R@5 {A_r5:.3f} Image R@5 {I_r5:.3f} over {N:d} validation pairs'
                .format(A_r5=A_r5, I_r5=I_r5, N=N_examples), flush=True)
            print(' * Audio R@1 {A_r1:.3f} Image R@1 {I_r1:.3f} over {N:d} validation pairs'
                .format(A_r1=A_r1, I_r1=I_r1, N=N_examples), flush=True)

    return recalls


def validate_all(audio_model, image_model,image_cnn,val_loader,args):
    

    exp_dir = args.save_root
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()    
    
    if cfg.TRAIN.MODAL=='co-train':
        if not isinstance(image_model, torch.nn.DataParallel):
            image_model = nn.DataParallel(image_model)
        if not isinstance(image_cnn,torch.nn.DataParallel):
            image_cnn = nn.DataParallel(image_cnn)

        image_model = image_model.to(device)
        image_cnn = image_cnn.to(device)
        # switch to evaluate mode
        image_model.eval()
        image_cnn.eval()
    

    # audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))
    # image_model.load_state_dict(torch.load("%s/models/best_image_model.pth" % (exp_dir)))    
    

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    I_embeddings = [] 
    A_embeddings = [] 
    AL_embeddings = []
    frame_counts = []
    I_class_ids = []
    A_class_ids = []
    with torch.no_grad():
        
        for i, (image_input, audio_input, cls_id, key, input_length, label) in enumerate(val_loader):    
            image_input,inverse = torch.unique(image_input,sorted = False,return_inverse = True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(image_input.size(0)).scatter_(0, inverse, perm)
            
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)
            image_input = image_input.squeeze(1)

            audio_input = audio_input.float().to(device)
            image_input = image_input.float().to(device)            

            # compute output
            if cfg.TRAIN.MODAL == 'co-train':
                image_feat = image_cnn(image_input)
                image_output = image_model(image_feat)  
            else:
                image_output = image_input
            
            if cfg.SPEECH.model == 'CNN':
                audio_output = audio_model(audio_input)
            else:
                audio_output = audio_model(audio_input,input_length)
            
            
            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()           

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)  
            I_class_ids.append(cls_id[perm])          
            A_class_ids.append(cls_id)     
            
            batch_time.update(time.time() - end)
            end = time.time()

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)  
        I_ids = torch.cat(I_class_ids) 
        A_ids = torch.cat(A_class_ids)  
        # pdb.set_trace()
        image_output,inverse = torch.unique(image_output,sorted = False,return_inverse = True, dim=0)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(image_output.size(0)).scatter_(0, inverse, perm)   
        I_ids = I_ids[perm]
        recalls = retrieval_evaluation_all(image_output,audio_output,I_ids,A_ids)        
        
    return recalls



def feat_extract_co(audio_model, path,args):
    audio_model = nn.DataParallel(audio_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    exp_dir = args.save_root  #+ '/' + 'pre_train'
    
    audio_model.load_state_dict(torch.load("%s/models/best_audio_model.pth" % (exp_dir)))  #best_audio_model
    

    audio_model = audio_model.to(device)  
    
    audio_model.eval()     
    
    
    # extract speech embeding of train set
    info = 'starting extract speech embedding feature of trainset \n'
    print (info)            
    save_path = os.path.join(exp_dir, 'embedding_extract.txt')
    with open(save_path, "a") as file:
        file.write(info)
    
    filepath = '%s/%s/filenames.pickle' % (path, 'train')
    
    
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)    
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))       

    if path.find('flickr') != -1:        
        data_dir = '%s/flickr_audio' % path
    elif path.find('places') != -1:
        data_dir = '%s/audio' % path
    elif path.find('birds') != -1:
        data_dir = '%s/CUB_200_2011' % path
    elif path.find('flower') != -1:
        data_dir = '%s/Oxford102' % path
    audio_feat = []
    j = 0
    
    for key in filenames:    
        if path.find('flickr') != -1 or path.find('places') != -1:
            audio_file = '%s/mel/%s.npy' % (data_dir, key) 
        else:
            audio_file = '%s/audio_mel/%s.npy' % (data_dir, key)

        audios = np.load(audio_file,allow_pickle=True)
        if len(audios.shape)==2:
            audios = audios[np.newaxis,:,:] 
        num_cap = audios.shape[0]
        if num_cap!=cfg.SPEECH.CAPTIONS_PER_IMAGE:
            print('erro with the number of captions')
            print(audio_file)
        for i in range(num_cap):
            cap = audios[i]
            cap = torch.tensor(cap)
            input_length = cap.shape[0]
            input_length  = torch.tensor(input_length)
            audio_input = cap.float().to(device)  
            audio_input = audio_input.unsqueeze(0)  
            input_length = input_length.float().to(device)        
            input_length = input_length.unsqueeze(0)
            audio_output = audio_model(audio_input,input_length)
            audio_output = audio_output.cpu().detach().numpy()
            if i == 0:
                outputs = audio_output
            else:
                outputs = np.vstack((outputs,audio_output))

        audio_feat.append(outputs)
        
        if j % 50 ==0:
            print('extracted the %ith audio feature'%j)   
        j += 1

    with open("%s/speech_embeddings_train.pickle" % (exp_dir), "wb") as f:
            pickle.dump(audio_feat, f)

    info = 'extracting speech embedding feature of trainset is finished \n'
    print (info)            
    save_path = os.path.join(exp_dir, 'embedding_extract.txt')
    with open(save_path, "a") as file:
        file.write(info)
    
    #extract speech embedding of test set
    info = 'starting extract speech embedding feature of testset \n'
    print (info)            

    with open(save_path, "a") as file:
        file.write(info)
    
    filepath = '%s/%s/filenames.pickle' % (path, 'test')
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)    
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))       

    audio_feat = []
    j = 0
    for key in filenames:    
        if path.find('flickr') != -1 or path.find('places') != -1:
            audio_file = '%s/mel/%s.npy' % (data_dir, key) 
        else:
            audio_file = '%s/audio_mel/%s.npy' % (data_dir, key)
        audios = np.load(audio_file,allow_pickle=True)
        if len(audios.shape)==2:
            audios = audios[np.newaxis,:,:] 
        num_cap = audios.shape[0]
        if num_cap!=cfg.SPEECH.CAPTIONS_PER_IMAGE:
            print('erro with the number of captions')
            print(audio_file)
        for i in range(num_cap):
            cap = audios[i]
            cap = torch.tensor(cap)
            input_length = cap.shape[0]
            input_length  = torch.tensor(input_length)
            audio_input = cap.float().to(device)  
            audio_input = audio_input.unsqueeze(0)  
            input_length = input_length.float().to(device)        
            input_length = input_length.unsqueeze(0)
            audio_output = audio_model(audio_input,input_length)
            audio_output = audio_output.cpu().detach().numpy()
            if i == 0:
                outputs = audio_output
            else:
                outputs = np.vstack((outputs,audio_output))

        audio_feat.append(outputs)
        
        if j % 50 ==0:
            print('extracted the %ith audio feature'%j)   
        j += 1

    info = 'extracting speech embedding feature of testset is finished \n'
    print (info)            
    with open(save_path, "a") as file:
        file.write(info)
    
    with open("%s/speech_embeddings_test.pickle" % (exp_dir), "wb") as f:
            pickle.dump(audio_feat, f)
    
    info = 'speech embedding is saved \n'
    print (info)            
    with open(save_path, "a") as file:
        file.write(info)
