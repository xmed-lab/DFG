import torch
import torch.nn.functional as F
import os,random
from einops import rearrange
from models import get_model
from dataloaders import MyDataset,PatientDataset,MyBatchSampler,MyDataset_refine
from torch.utils.data import DataLoader
from losses import ProtoLoss,MultiClassDiceLoss,PixelPrototypeCELoss

import numpy as np

from utils import IterationCounter, Visualizer, mean_dice, get_indices_of_pairs, mean_dice_new, KeepMaxContour, mean_asd, keepmaxregion

from tqdm import tqdm
import pdb

import sys
from medsam.MedSAM_Infer import get_medsam, medsam_infer_encoder, medsam_infer_decoder, medsam_infer_encoder_batch
#from ckpts import proto
import time
import math

class SAM_Trainer():
    def __init__(self, opt):
        self.opt = opt
        
        #params for MedSAM prototype estimation
        #self.area_thresh_est = 800#200
        #self.area_thresh_deci = 500
        #self.bg_cos_thresh = 0.85
        self.proto = [0]
        self.load_proto = False
        #

        self.exist_thresh = 200#50
        #self.init_thresh = 0.95
        self.init_thresh_delta = 0.005#0.0005#0.001#0.002#0.005
        
        #self.ratio = 2
        self.radius = 4
        self.predefined_featuresize = 256#int(256 / self.ratio)
        self.ind_from, self.ind_to = get_indices_of_pairs(radius=self.radius, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from); self.ind_to = torch.from_numpy(self.ind_to)
        self.refine_order = [1,2,3,4]#[2,3,1,4]
        #self.pixel_thresh = {1:60,2:30,3:20,4:100}
        self.aff_thresh = 0.99 #opt['aff_thresh']
        self.bbox_margin = 3
        self.diffuse_max_step = [0,25,20,20,35]#determined by estimated max area of each class [0,25,20,20,35][0,20,15,15,30]
        self.medsam_model = get_medsam()
        self.tol_twostep = 35
        self.tol_onestep = 15
        
        self.radius_2 = 4
        self.predefined_featuresize_2 = 256
        #self.ratio_2 = int(256/self.predefined_featuresize_2)
        self.ind_from_2, self.ind_to_2 = get_indices_of_pairs(radius=self.radius_2, size=(self.predefined_featuresize_2, self.predefined_featuresize_2))
        self.ind_from_2 = torch.from_numpy(self.ind_from_2); self.ind_to_2 = torch.from_numpy(self.ind_to_2)
        self.diffuse_max_step_2 = [0,25,20,20,35]#determined by estimated max area of each class [0,25,20,20,35][0,20,15,15,30]
        self.dist_thresh_2 = 2.5
        self.max_dist_2 = 0.35#opt['max_dist_2']
        self.tol_2_twostep = 30
        self.tol_2_onestep = 15
        self.tol_2_threestep = 45

        self.step_3 = 2
        self.diffuse_max_step_3 = 6
        self.tol_3 = 15

    def initialize(self):

        ### initialize dataloaders
        if self.opt['patient_level_dataloader']:
            train_dataset = PatientDataset(self.opt['data_root'], self.opt['target_sites'], phase='train', split_train=True)
            patient_sampler = MyBatchSampler(train_dataset,self.opt['batch_size'])
            self.train_dataloader = DataLoader(train_dataset,batch_sampler=patient_sampler,num_workers=self.opt['num_workers'])
        else:
            self.train_dataloader = DataLoader(
                MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='val', split_train=True),
                batch_size=self.opt['batch_size'],
                shuffle=True,
                drop_last=False,
                num_workers=self.opt['num_workers']
            )

        print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='val', split_train=False),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        print('Length of validation dataset: ', len(self.val_dataloader))

        ## initialize the models

        self.model = get_model(self.opt)
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])###
        self.model = self.model.to(self.opt['gpu_id'])
        
        # visualizations
        self.visualizer = Visualizer(self.opt)
        self.set_seed(self.opt['random_seed'])
        self.model_resume()

        self_dict = vars(self)
        filtered_dict = {key: value for key, value in self_dict.items() if key != 'medsam_model'}
        self.visualizer.print(filtered_dict)

        self.decoder_time=0
        self.encoder_time=0
        self.other_time123=0
    
    def initialize_train(self):

        ### initialize dataloaders
        self.train_dataloader = DataLoader(
            MyDataset_refine(self.opt['refine_postprocess_dir'], phase='train'),#'refine_dir'
            batch_size=self.opt['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.opt['num_workers']
        )

        print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='val', split_train=False),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        print('Length of validation dataset: ', len(self.val_dataloader))

        self.total_epochs = self.opt['total_epochs']

        """ for param in self.model.outc.named_parameters():
            param[1].requires_grad_(False) """

        ## optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        ## losses
        self.criterian_pce = PixelPrototypeCELoss(self.opt)
        self.criterian_dc  = MultiClassDiceLoss(self.opt)

        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.model_resume()

    def set_seed(self,seed):
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed for this experiment is {} !'.format(seed))

    def save_models(self, step, dice):
        if step != 0:
            checkpoint_dir = self.opt['checkpoint_dir']
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'saved_models', 'model_step_{}_dice_{:.4f}.pth'.format(step,dice)))

    
    def save_best_models(self, step, dice):
        checkpoint_dir = self.opt['checkpoint_dir']
        for file in os.listdir(os.path.join(checkpoint_dir, 'saved_models')):
            if 'best_model' in file:
                os.remove(os.path.join(checkpoint_dir, 'saved_models', file))
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state,os.path.join(checkpoint_dir, 'saved_models','best_model_step_{}_dice_{:.4f}.pth'.format(step,dice)))


    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params,lr=self.opt['lr'],betas=(0.9, 0.999), weight_decay=0.0005)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, min_lr=1e-7) # maximize dice score
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)#step_size=1, gamma=0.8
        return optimizer, scheduler
    
    def model_resume(self):
        if self.opt['continue_train']:
            if os.path.isfile(self.opt['resume']):
                print("=> Loading checkpoint '{}'".format(self.opt['resume']))
            state = torch.load(self.opt['resume'])
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.start_epoch = state['epoch']
        else:
            self.start_epoch = 0
            print("=> No checkpoint, train from scratch !")

    ###################### training logic ################################
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()
        
        # get losses
        imgs = data[0]
        segs = data[1]

        target_f,predict = self.model(imgs)#,only_feature=True

        """ target_f = rearrange(target_f, 'b c h w -> (b h w) c')
        t2p_loss, p2t_loss = self.criterion_proto(self.source_prototypes,target_f)
        loss = t2p_loss + p2t_loss """
        
        loss_dc  = self.criterian_dc(predict, segs)
        loss = loss_dc

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()


        adapt_losses = {}
        adapt_losses['total_loss'] = loss.detach()

        return predict, adapt_losses
    
    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]
        fea,predict = self.model(imgs)

        self.model.train()

        return fea, predict

    def launch(self):

        self.initialize()
        """ self.get_proto() """
        refine_start_time = time.time()
        self.refine_sam()
        refine_end_time = time.time()
        self.pl_postprocess()
        postprocess_end_time = time.time()

        self.initialize_train()
        train_start_time = time.time()
        self.train_sam()
        train_end_time = time.time()

        refine_time=refine_end_time-refine_start_time
        #self.visualizer.print(f"refine time: {refine_time:.2f} s")

        postprocess_time=postprocess_end_time-refine_end_time
        #self.visualizer.print(f"postprocess time: {postprocess_time:.2f} s")

        train_time=train_end_time-train_start_time
        #self.visualizer.print(f"100epoch total time: {train_time:.2f} s")
        
        
    def refine_sam(self):
        other_time=0
        self.visualizer.print('*'*20)
        self.visualizer.print('refine start')
                
        val_metrics = {}
        sample_dict = {}
        train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
        for it, (val_imgs, val_segs, val_names) in enumerate(train_iterator):

            val_imgs = val_imgs.to(self.opt['gpu_id'])
            val_segs = val_segs.to(self.opt['gpu_id'])
            
            fea, predicts = self.validate_one_step([val_imgs, val_segs])
            #print(val_imgs.shape)#torch.Size([16, 3, 256, 256])
            #print(fea.shape,predicts.shape)#torch.Size([16, 64, 256, 256]) torch.Size([16, 5, 256, 256])
            #print(val_segs.shape)#torch.Size([16, 256, 256])
            #self.visualizer.print(str(val_names))#('0006_9', '0008_14', '0008_37', '0008_8', '0008_23', '0024_47', '0029_3', '0027_23', '0001_67', '0040_51', '0010_52', '0001_44', '0022_35', '0005_66', '0021_25', '0004_13')

            #print(self.source_prototypes)#torch.Size([5, 64])
            #plot_fea(self.opt, val_imgs[:,1].cpu(), fea.detach().cpu(), val_segs.detach().cpu(), self.source_prototypes.detach().cpu(), val_names)
            pl_batch, preds = self.refine_pl_sam(val_imgs, val_segs, fea, predicts, val_names)
            #sys.exit()
            #visuals = {'images':val_imgs[:,1].detach().cpu().numpy(),'preds':torch.argmax(predicts,dim=1).detach().cpu().numpy(),
            #        'gt_segs':val_segs.detach().cpu().numpy()}
            #self.visualizer.display_current_results(self.iter_counter.steps_so_far,visuals)
            #sys.exit()
            start_time = time.time()
            #save refined pl temporally commented
            data_dir = os.path.join(self.opt['data_root'], self.opt['target_sites'][0],'train')
            for i,name in enumerate(val_names):
                sample_name,index = name.split('_')[0],int(name.split('_')[1])
                
                data = np.load(os.path.join(data_dir, name+'.npz'))
                img = data['image']
                seg = data['label']
                pl = pl_batch[i].detach().cpu().numpy()
                np.savez(os.path.join(self.opt['refine_dir'],'{}_{}.npz'.format(sample_name,index)),image = img,pl = pl,label = seg)
                
            ##
            for i,name in enumerate(val_names):

                sample_name,index = name.split('_')[0],int(name.split('_')[1])
                sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(preds[i].detach().cpu(),pl_batch[i].detach().cpu(),           
                                                                               val_segs[i].detach().cpu(),index)]
            end_time = time.time()
            other_time=other_time+(end_time-start_time)
            
        start_time = time.time()    
        pred_results_list = []
        pl_list = []
        gt_segs_list = []
        
        for k in sample_dict.keys():

            sample_dict[k].sort(key=lambda ele: ele[-1])
            preds = []
            pls = []
            targets = []
            for pred,pl,target,_ in sample_dict[k]:
                #if target.sum()==0:
                #    continue
                preds.append(pred)
                pls.append(pl)
                targets.append(target)
            pred_results_list.append(torch.stack(preds,dim=-1))
            pl_list.append(torch.stack(pls,dim=-1))
            gt_segs_list.append(torch.stack(targets,dim=-1))
                
        val_metrics['dice_ori'] = mean_dice_new(pred_results_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
        val_metrics['dice_refine'] = mean_dice_new(pl_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
        self.visualizer.print(val_metrics)
        end_time = time.time()
        other_time=other_time+(end_time-start_time)
        #self.visualizer.print(f"refine other time 0: {other_time:.2f} s")

        #self.visualizer.print(f"encoder time: {self.encoder_time:.2f} s")
        #self.visualizer.print(f"decoder time: {self.decoder_time:.2f} s")
        #self.visualizer.print(f"refine other time123: {self.other_time123:.2f} s")
        
        #sys.exit()
        #self.schedular.step()
    def pl_postprocess(self):
        other_time=0
        self.visualizer.print('pl_postprocess start')

        refine_dir = self.opt['refine_dir']
        val_metrics = {}
        sample_dict = {}
        for data_name in os.listdir(refine_dir):
            start_time = time.time()
            data = np.load(os.path.join(refine_dir, data_name))
            img = data['image']
            seg = data['label']
            pl = data['pl']
            name = data_name[:-4]
            end_time = time.time()
            other_time=other_time+(end_time - start_time)

            sample_name,index = name.split('_')[0],int(name.split('_')[1])
            sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(torch.from_numpy(pl),
                                                                            torch.from_numpy(seg).long(),index)]
            
        pl_list = []
        postprocess_list = []
        gt_segs_list = []
        
        for k in sample_dict.keys():

            sample_dict[k].sort(key=lambda ele: ele[-1])
            pls = []
            targets = []
            for pl,target,_ in sample_dict[k]:
                pls.append(pl)
                targets.append(target)
            pl_list.append(torch.stack(pls,dim=-1))
            gt_segs_list.append(torch.stack(targets,dim=-1))
            postprocess_list.append(keepmaxregion(torch.stack(pls,dim=-1), self.opt['num_classes']))
        
        start_time = time.time()        
        val_metrics['dice_ori'] = mean_dice_new(pl_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
        val_metrics['dice_refine'] = mean_dice_new(postprocess_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
        val_metrics['asd_ori'] = mean_asd(pl_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
        val_metrics['asd_refine'] = mean_asd(postprocess_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
        self.visualizer.print(val_metrics)
        
        for i, k in enumerate(sample_dict.keys()):
            for index in range(postprocess_list[i].size(2)):
                pl = postprocess_list[i][:,:,index].numpy()
                name = '{}_{}.npz'.format(k,index)
                data = np.load(os.path.join(refine_dir, name))
                img = data['image']
                seg = data['label']
                np.savez(os.path.join(self.opt['refine_postprocess_dir'],name),image = img,pl = pl,label = seg)
        end_time = time.time()
        other_time=other_time+(end_time - start_time)
        #self.visualizer.print(f"postprocess other time: {other_time:.2f} s")
        return
        
    def refine_pl_sam(self, val_imgs, val_segs, fea, predicts, val_names):
        fea = F.normalize(fea, p=2, dim=1)
        probs = F.softmax(predicts,dim=1)
        preds = torch.argmax(probs,dim=1)
        #downsample fea and probs
        #fea = F.interpolate(fea,size=(self.predefined_featuresize,self.predefined_featuresize),mode='bilinear')
        #probs = F.interpolate(probs,size=(self.predefined_featuresize,self.predefined_featuresize),mode='bilinear')

        print('aff begin')
        assert fea.size(2) == self.predefined_featuresize and fea.size(3) == self.predefined_featuresize
        ind_from = self.ind_from
        ind_to = self.ind_to

        #fea1 = fea
        fea = fea.view(fea.size(0), fea.size(1), -1)

        ff = torch.index_select(fea, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(fea, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.sum(ft*ff, dim=1)
        del ff
        print('aff end')
        preds_new_batch = []
        ###
        #image_embedding_batch, H, W = medsam_infer_encoder_batch(self.medsam_model, val_imgs.permute(0,2,3,1).cpu().numpy())

        #each image
        for ind in range(aff.size(0)):
            self.visualizer.print(f'{val_names[ind]}')
            #medsam encoder
            start_time = time.time()
            image_embedding, H, W = medsam_infer_encoder(self.medsam_model, val_imgs[ind].permute(1,2,0).cpu().numpy())
            #image_embedding = image_embedding_batch[ind].unsqueeze(0)
            end_time = time.time()
            self.encoder_time=self.encoder_time+(end_time-start_time)
            #continue aff
            """ aff_img = aff[ind]
            aff_img = aff_img.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = fea.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_img = torch.sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                        torch.cat([aff_img, torch.ones([area]), aff_img])).to_dense().cuda() """
            #new continue aff
            aff_img = aff[ind].cpu()
            area = fea.size(2)

            aff_dist = torch.zeros((area, ft.size(2)*2+1))
            aff_indices = torch.zeros((area, ft.size(2)*2+1)).long()
            ind_to = ind_to.view(ft.size(2), -1)
            for col in range(ft.size(2)):
                aff_dist[ind_from, col] = aff_img[col]
                aff_indices[ind_from, col] = ind_to[col]
            for col in range(ft.size(2)):
                aff_dist[ind_to[col], col+ft.size(2)] = aff_img[col]
                aff_indices[ind_to[col], col+ft.size(2)] = ind_from
            aff_dist[torch.arange(0, area).long(), ft.size(2)*2] = 1
            aff_indices[torch.arange(0, area).long(), ft.size(2)*2] = torch.arange(0, area).long()
            aff_dist = aff_dist.cuda()
            aff_indices = aff_indices.cuda()

            #diffuse
            preds_new = []
            cos_sim = [0]*self.opt['num_classes']
            cls_refine = []
            #high_area = [False]*self.opt['num_classes']
            for cls in self.refine_order:
                #if (preds[ind]==cls).sum().item()<self.pixel_thresh[cls]:
                #    preds_new.append((preds[ind]==cls).long())
                #    continue
                init = (probs[ind][cls]>0.8).long()
                self.visualizer.print(f'{ind},{cls},{init.sum().item()}')
                if init.sum() < self.exist_thresh:
                    preds_new.append((preds[ind]==cls).long())
                    continue
                init = (probs[ind][cls]>torch.max(probs[ind][cls]) - self.init_thresh_delta).long()
                if init.sum() == 0:
                    preds_new.append((preds[ind]==cls).long())
                    continue
                init = KeepMaxContour(init.cpu().numpy())
                init = torch.from_numpy(init).cuda()
                init = torch.nonzero(init.view(-1)).squeeze(1)
                region = init
                area = [1e5,1e5]
                seg = [-1,-1]
                sam_refine = False
                for step in range(self.diffuse_max_step[cls]):
                    #1d to 2d
                    region_2d = torch.zeros(probs.size(2)*probs.size(3)).cuda()
                    region_2d[region] = 1
                    region_2d = region_2d.view((probs.size(2),probs.size(3)))

                    xys = torch.nonzero(region_2d)
                    hmin, hmax = max(xys[:,0].min().item()-self.bbox_margin, 0), min(xys[:,0].max().item()+self.bbox_margin, preds.size(1)-1)
                    wmin, wmax = max(xys[:,1].min().item()-self.bbox_margin, 0), min(xys[:,1].max().item()+self.bbox_margin, preds.size(2)-1)
                    #medsam inference
                    box = [wmin, hmin, wmax, hmax]
                    #box=[170, 130, 220, 180]
                    box_np = np.array([box])
                    # transfer box_np t0 1024x1024 scale
                    box_1024 = box_np / np.array([W, H, W, H]) * 1024

                    start_time = time.time()
                    medsam_seg = medsam_infer_decoder(self.medsam_model, image_embedding, box_1024, H, W)
                    end_time = time.time()
                    self.decoder_time=self.decoder_time+(end_time-start_time)
                    medsam_seg = KeepMaxContour(medsam_seg)
                    if medsam_seg is None:
                        #preds_new.append((preds[ind]==cls).long())
                        self.visualizer.print('no contour')
                        region = aff_indices[region][aff_dist[region]>self.aff_thresh].unique()
                        continue
                    area.append(int(medsam_seg.sum()))
                    seg.append(medsam_seg)
                    self.visualizer.print(f'{box}, {area[-1]}')
                    #if abs(area[step+2] - area[step]) < self.tol:###
                    if area[-1]>50 and ((self.diff_area(seg[-1], seg[-3]) < self.tol_twostep) or (self.diff_area(seg[-1], seg[-2]) < self.tol_onestep)):
                        sam_refine = True
                        medsam_seg_new = self.refine_stage2(medsam_seg, image_embedding, H, W, cls)

                        if medsam_seg_new is None:
                            preds_new.append((torch.from_numpy(medsam_seg).cuda())*2)
                        else:
                            preds_new.append((torch.from_numpy(medsam_seg_new).cuda())*2)
                        
                        medsam_seg_final = medsam_seg if medsam_seg_new is None else medsam_seg_new

                        #high_area[cls] = medsam_seg_final.sum()>self.area_thresh_deci
                        
                        #temporally commented
                        """ seg_ctr = self.compute_center(image_embedding, medsam_seg_final, H, W)
                        cls_proto = self.proto[cls]
                        cos_sim[cls] = torch.sum(seg_ctr*cls_proto).item()
                        cls_refine.append(cls) """

                        #preds[ind, cls]
                        start_time = time.time()
                        gt = (val_segs[ind]==cls)
                        self.test_refine(gt, preds[ind]==cls, preds_new[-1]/2, ind, cls, step, box)
                        end_time = time.time()
                        self.other_time123=self.other_time123+(end_time-start_time)
                        break
                    #diffuse
                    #region = torch.nonzero(aff_img[region]>self.aff_thresh)[:,1].unique()
                    #new diffuse
                    region = aff_indices[region][aff_dist[region]>self.aff_thresh].unique()
                #cannot find stable interval
                if not sam_refine:
                    preds_new.append((preds[ind]==cls).long())
            
            #two classes overlap, temporally commented
            """ cls_overlap = []
            for i in range(len(cls_refine)-1):
                for j in range(i+1, len(cls_refine)):
                    pred_i = preds_new[cls_refine[i]-1]/2###refine_order
                    pred_j = preds_new[cls_refine[j]-1]/2###
                    dice = torch.sum(pred_i*pred_j)*2.0 / (torch.sum(pred_i) + torch.sum(pred_j))
                    if dice > 0.3:
                        if cos_sim[cls_refine[i]] > cos_sim[cls_refine[j]]:
                            preds_new[cls_refine[j]-1] = torch.zeros(preds_new[cls_refine[j]-1].shape, device='cuda')
                            self.visualizer.print(f'class{cls_refine[j]} removed')
                        else:
                            preds_new[cls_refine[i]-1] = torch.zeros(preds_new[cls_refine[i]-1].shape, device='cuda')
                            self.visualizer.print(f'class{cls_refine[i]} removed')
                        if cls_refine[i] not in cls_overlap:
                            cls_overlap.append(cls_refine[i])
                        if cls_refine[j] not in cls_overlap:
                            cls_overlap.append(cls_refine[j])
            for cls in cls_overlap:
                cls_refine.remove(cls) """
            #remove background
            '''
            for cls in cls_refine:
                if high_area[cls] and (cos_sim[cls] < self.bg_cos_thresh):
                    preds_new[cls-1] = torch.zeros(preds_new[cls-1].shape, device='cuda')#0
                    self.visualizer.print(f'class{cls} background, removed')
            '''
            """ self.visualizer.print(f'cos_sim:{cos_sim}') """#temporally commented
            
            preds_new_batch.append(torch.stack(preds_new))
        
        preds_new_batch = torch.stack(preds_new_batch)
        preds_new_batch = torch.cat((torch.ones(preds_new_batch.size(0),1,preds_new_batch.size(2),preds_new_batch.size(3),
                                                device='cuda')*0.5, preds_new_batch), dim=1)
        pl_batch = torch.argmax(preds_new_batch,dim=1)
        
        return pl_batch, preds
    
    def refine_stage2(self, medsam_seg, image_embedding, H, W, cls, strict=False):
        self.visualizer.print('stage2 start')
        #resize and normalize 
        fea=F.interpolate(image_embedding,size=(self.predefined_featuresize_2, self.predefined_featuresize_2),mode='bilinear')
        fea = F.normalize(fea, p=2, dim=1)
        
        fea1=fea[0]
        fea1=fea1.reshape((fea1.shape[0],-1))
        fea1 = fea1.permute(1,0)
        #compute centroid and avg cosine distance
        medsam_seg_1d = torch.from_numpy(medsam_seg).cuda().reshape((-1,))
        ctr = torch.mean(fea1[medsam_seg_1d==1], dim=0, keepdim=True)
        ctr = F.normalize(ctr, p=2, dim=1)
        avg_dist = 1-torch.mean((fea1[medsam_seg_1d==1]*ctr).sum(dim=1))
        self.visualizer.print(f'avg_dist:{avg_dist}')
        #aff of medsam feature
        #print('aff2 begin')
        assert fea.size(2) == self.predefined_featuresize_2 and fea.size(3) == self.predefined_featuresize_2
        ind_from = self.ind_from_2
        ind_to = self.ind_to_2

        fea = fea.view(fea.size(0), fea.size(1), -1)

        ff = torch.index_select(fea, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(fea, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff_ft = 1 - torch.sum(ft*ctr.unsqueeze(2).unsqueeze(3), dim=1)
        aff_ff = 1 - torch.sum(ff.expand(-1, -1, ft.size(2), -1)*(ctr.unsqueeze(2).unsqueeze(3)), dim=1)
        #del ff
        #print('aff2 end')

        #continue aff
        aff_ft = aff_ft[0].cpu()
        #aff_ft = aff_ft.view(-1).cpu()
        aff_ff = aff_ff[0].cpu()
        #aff_ff = aff_ff.view(-1).cpu()

        #ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
        #indices = torch.stack([ind_from_exp, ind_to])
        #indices_tp = torch.stack([ind_to, ind_from_exp])
        
        area = fea.size(2)
        #indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

        #aff_2 = torch.sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
        #                            torch.cat([aff_ft, torch.zeros([area]), aff_ff])).cuda()
        aff_dist = torch.ones((area, ft.size(2)*2+1))*1e4
        aff_indices = torch.zeros((area, ft.size(2)*2+1)).long()
        ind_to = ind_to.view(ft.size(2), -1)
        for col in range(ft.size(2)):
            aff_dist[ind_from, col] = aff_ft[col]
            aff_indices[ind_from, col] = ind_to[col]
        for col in range(ft.size(2)):
            aff_dist[ind_to[col], col+ft.size(2)] = aff_ff[col]
            aff_indices[ind_to[col], col+ft.size(2)] = ind_from
        aff_dist[torch.arange(0, area).long(), ft.size(2)*2] = 0
        aff_indices[torch.arange(0, area).long(), ft.size(2)*2] = torch.arange(0, area).long()
        aff_dist = aff_dist.cuda()
        aff_indices = aff_indices.cuda()
        
        #diffuse####
        region = medsam_seg_1d==1
        area = []
        seg = []
        ###test
        xys = torch.nonzero(torch.from_numpy(medsam_seg))
        hmin, hmax = xys[:,0].min().item(), xys[:,0].max().item()
        wmin, wmax = xys[:,1].min().item(), xys[:,1].max().item()
        box = [wmin, hmin, wmax, hmax]
        self.visualizer.print(f'initial box:{box}, area:{medsam_seg.sum()}')
        
        ###test
        box_last = [0,0,0,0]
        for step in range(self.diffuse_max_step_2[cls]):
            
            #1d to 2d
            region_2d = torch.zeros(self.predefined_featuresize_2**2).cuda()
            region_2d[region] = 1
            region_2d = region_2d.view((self.predefined_featuresize_2,self.predefined_featuresize_2))

            xys = torch.nonzero(region_2d)
            hmin, hmax = max(xys[:,0].min().item()-3, 0), min(xys[:,0].max().item()+3, H-1)
            wmin, wmax = max(xys[:,1].min().item()-3, 0), min(xys[:,1].max().item()+3, W-1)
            #medsam inference
            box = [wmin, hmin, wmax, hmax]
            
            if box==box_last:
                #check two/one step stable
                medsam_seg = self.find_twostep_stable(seg)
                if medsam_seg is not None:
                    return medsam_seg
                else:
                    medsam_seg = self.find_onestep_stable(seg)
                    if medsam_seg is not None:
                        return medsam_seg
                #stage3
                medsam_seg_new = self.refine_stage3(box, image_embedding, H, W, seg)
                if medsam_seg_new is not None:
                    return medsam_seg_new
                else:
                    if strict:
                        return None
                    else:
                        return seg[-(1+int(self.diffuse_max_step_3))]
            box_last = box
            #box=[170, 130, 220, 180]
            box_np = np.array([box])
            # transfer box_np t0 1024x1024 scale
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

            start_time = time.time()
            medsam_seg = medsam_infer_decoder(self.medsam_model, image_embedding, box_1024, H, W)
            end_time = time.time()
            self.decoder_time=self.decoder_time+(end_time-start_time)
            medsam_seg = KeepMaxContour(medsam_seg)
            
            area.append(int(medsam_seg.sum()))
            self.visualizer.print(f'{box}, {area[-1]}')
            seg.append(medsam_seg)
            #if (abs(area[step+2] - area[step+1]) < self.tol_2_onestep) or \
            #((abs(area[step+2] - area[step]) < self.tol_2_twostep) and \
            #(abs(area[step+2] - area[step+1]) < self.tol_2_twostep) and \
            #(abs(area[step+1] - area[step]) < self.tol_2_twostep)):
            if self.threestep_stable(seg):
                self.visualizer.print(f'stage2_step:{step}, threestep stable')
                return seg[-2]
            
            #diffuse
            region = aff_indices[region][aff_dist[region]<min(self.dist_thresh_2*avg_dist, self.max_dist_2)].unique()

        medsam_seg = self.find_twostep_stable(seg)
        if medsam_seg is not None:
            return medsam_seg
        else:
            return self.find_onestep_stable(seg)    
        return None        

    def refine_stage3(self, box, image_embedding, H, W, seg):
        self.visualizer.print('stage3 start')
        for step in range(self.diffuse_max_step_3):
            box = [box[0]-self.step_3, box[1]-self.step_3, box[2]+self.step_3, box[3]+self.step_3]
            box_np = np.array([box])
            box_1024 = box_np / np.array([W, H, W, H]) * 1024

            start_time = time.time()
            medsam_seg = medsam_infer_decoder(self.medsam_model, image_embedding, box_1024, H, W)
            end_time = time.time()
            self.decoder_time=self.decoder_time+(end_time-start_time)

            medsam_seg = KeepMaxContour(medsam_seg)
            self.visualizer.print(f'{box}, {int(medsam_seg.sum())}')
            seg.append(medsam_seg)
            #if (abs(area[-1] - area[-2]) < self.tol_2_onestep) or \
            #((abs(area[-1] - area[-3]) < self.tol_2_twostep) and \
            #(abs(area[-1] - area[-2]) < self.tol_2_twostep) and \
            #(abs(area[-2] - area[-3]) < self.tol_2_twostep)):
            if (len(seg)>2 and self.diff_area(seg[-1], seg[-3]) < self.tol_2_twostep) or \
            self.diff_area(seg[-1], seg[-2]) < self.tol_2_onestep:
                self.visualizer.print(f'stage3_step:{step}')
                return medsam_seg
        self.visualizer.print('stage3 finish')
        return None

    def test_refine(self, gt, pred, pred_new, ind, cls, step, box):
        self.visualizer.print(f'ind:{ind},cls:{cls},step:{step}')
        xys = torch.nonzero(gt)
        if gt.sum()==0:
            gt_box=None
        else:
            gt_box = [xys[:,1].min().item(), xys[:,0].min().item(), xys[:,1].max().item(), xys[:,0].max().item()]
        self.visualizer.print(f'gt box:{gt_box},box:{box}')
        self.visualizer.print(f'gt area: {gt.sum().item()}')
        self.visualizer.print(f'original area: {pred.sum().item()}')
        self.visualizer.print(f'refine area: {pred_new.sum().item()}')
        dice_ori = torch.sum(pred[gt])*2.0 / (torch.sum(pred) + torch.sum(gt))
        dice_refine = torch.sum(pred_new[gt])*2.0 / (torch.sum(pred_new) + torch.sum(gt))
        self.visualizer.print(f'dice_ori:{dice_ori.item()},dice_refine:{dice_refine.item()}')
        if dice_refine.item() < 0.8:
            self.visualizer.print(f'dice_refine is less than 0.8 cls{cls}')
        elif dice_refine.item() < 0.9:
            self.visualizer.print(f'dice_refine is less than 0.9 cls{cls}')                     

    def diff_area(self, medsam_seg_1, medsam_seg_2):
        diff = medsam_seg_1 != medsam_seg_2
        return int(diff.sum())   

    def threestep_stable(self, seg):
        if len(seg) < 4:
            return False
        return self.diff_area(seg[-1], seg[-4]) < self.tol_2_threestep

    def find_twostep_stable(self, seg):
        for i in range(2, len(seg)):
            if self.diff_area(seg[i], seg[i-2]) < self.tol_2_twostep:
                self.visualizer.print(f'stage2_step:{i}, twostep stable')
                return seg[i]
        return None

    def find_onestep_stable(self, seg):
        for i in range(1, len(seg)):
            if self.diff_area(seg[i], seg[i-1]) < self.tol_2_onestep:
                self.visualizer.print(f'stage2_step:{i}, onestep stable')
                return seg[i]
        return None                 

    def get_proto(self):
        if self.load_proto:
            proto = torch.load(self.opt['proto_dir'])
            self.proto = proto
            return
        
        self.visualizer.enable_print = False        
        centers = [[]]*self.opt['num_classes']
        train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
        for it, (val_imgs, val_segs, val_names) in enumerate(train_iterator):

            val_imgs = val_imgs.to(self.opt['gpu_id'])
            val_segs = val_segs.to(self.opt['gpu_id'])
            
            fea, predicts = self.validate_one_step([val_imgs, val_segs])
            #print(val_imgs.shape)#torch.Size([16, 3, 256, 256])
            #print(fea.shape,predicts.shape)#torch.Size([16, 64, 256, 256]) torch.Size([16, 5, 256, 256])
            #print(val_segs.shape)#torch.Size([16, 256, 256])
            #self.visualizer.print(str(val_names))#('0006_9', '0008_14', '0008_37', '0008_8', '0008_23', '0024_47', '0029_3', '0027_23', '0001_67', '0040_51', '0010_52', '0001_44', '0022_35', '0005_66', '0021_25', '0004_13')

            #print(self.source_prototypes)#torch.Size([5, 64])
            
            self.get_centers(val_imgs, val_segs, fea, predicts, val_names, centers)

        for cls in range(1,self.opt['num_classes']):
            ctrs = torch.stack(centers[cls])
            ctr = torch.mean(ctrs, dim=0, keepdim=True)
            ctr = F.normalize(ctr, p=2, dim=1)
            self.proto.append(ctr.squeeze())

        self.visualizer.enable_print = True
        self.visualizer.print('proto computation end')
        torch.save(self.proto, self.opt['proto_dir'])
        #self.visualizer.print(self.proto)

    
    def get_centers(self, val_imgs, val_segs, fea, predicts, val_names, centers):
        
        fea = F.normalize(fea, p=2, dim=1)
        probs = F.softmax(predicts,dim=1)
        preds = torch.argmax(probs,dim=1)
        #downsample fea and probs
        #fea = F.interpolate(fea,size=(self.predefined_featuresize,self.predefined_featuresize),mode='bilinear')
        #probs = F.interpolate(probs,size=(self.predefined_featuresize,self.predefined_featuresize),mode='bilinear')

        print('aff begin')
        assert fea.size(2) == self.predefined_featuresize and fea.size(3) == self.predefined_featuresize
        ind_from = self.ind_from
        ind_to = self.ind_to

        #fea1 = fea
        fea = fea.view(fea.size(0), fea.size(1), -1)

        ff = torch.index_select(fea, dim=2, index=ind_from.cuda(non_blocking=True))
        ft = torch.index_select(fea, dim=2, index=ind_to.cuda(non_blocking=True))

        ff = torch.unsqueeze(ff, dim=2)
        ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))

        aff = torch.sum(ft*ff, dim=1)
        del ff
        print('aff end')
        preds_new_batch = []
        #each image
        for ind in range(aff.size(0)):
            self.visualizer.print(f'{val_names[ind]}')
            #medsam encoder
            image_embedding, H, W = medsam_infer_encoder(self.medsam_model, val_imgs[ind].permute(1,2,0).cpu().numpy())
            
            #continue aff
            """ aff_img = aff[ind]
            aff_img = aff_img.view(-1).cpu()

            ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
            indices = torch.stack([ind_from_exp, ind_to])
            indices_tp = torch.stack([ind_to, ind_from_exp])

            area = fea.size(2)
            indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

            aff_img = torch.sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                        torch.cat([aff_img, torch.ones([area]), aff_img])).to_dense().cuda() """
            #new continue aff
            aff_img = aff[ind].cpu()
            area = fea.size(2)

            aff_dist = torch.zeros((area, ft.size(2)*2+1))
            aff_indices = torch.zeros((area, ft.size(2)*2+1)).long()
            ind_to = ind_to.view(ft.size(2), -1)
            for col in range(ft.size(2)):
                aff_dist[ind_from, col] = aff_img[col]
                aff_indices[ind_from, col] = ind_to[col]
            for col in range(ft.size(2)):
                aff_dist[ind_to[col], col+ft.size(2)] = aff_img[col]
                aff_indices[ind_to[col], col+ft.size(2)] = ind_from
            aff_dist[torch.arange(0, area).long(), ft.size(2)*2] = 1
            aff_indices[torch.arange(0, area).long(), ft.size(2)*2] = torch.arange(0, area).long()
            aff_dist = aff_dist.cuda()
            aff_indices = aff_indices.cuda()

            #diffuse
            preds_new = []
            for cls in self.refine_order:
                init = (probs[ind][cls]>0.8).long()
                #self.visualizer.print(f'{ind},{cls},{init.sum().item()}')
                if init.sum() < self.area_thresh_est:
                    continue
                init = (probs[ind][cls]>torch.max(probs[ind][cls]) - self.init_thresh_delta).long()
                if init.sum() == 0:
                    continue
                init = KeepMaxContour(init.cpu().numpy())
                init = torch.from_numpy(init).cuda()
                init = torch.nonzero(init.view(-1)).squeeze(1)
                region = init
                area = [1e5,1e5]
                seg = [-1,-1]
                for step in range(self.diffuse_max_step[cls]):
                    #1d to 2d
                    region_2d = torch.zeros(probs.size(2)*probs.size(3)).cuda()
                    region_2d[region] = 1
                    region_2d = region_2d.view((probs.size(2),probs.size(3)))

                    xys = torch.nonzero(region_2d)
                    hmin, hmax = max(xys[:,0].min().item()-self.bbox_margin, 0), min(xys[:,0].max().item()+self.bbox_margin, preds.size(1)-1)
                    wmin, wmax = max(xys[:,1].min().item()-self.bbox_margin, 0), min(xys[:,1].max().item()+self.bbox_margin, preds.size(2)-1)
                    #medsam inference
                    box = [wmin, hmin, wmax, hmax]
                    #box=[170, 130, 220, 180]
                    box_np = np.array([box])
                    # transfer box_np t0 1024x1024 scale
                    box_1024 = box_np / np.array([W, H, W, H]) * 1024

                    medsam_seg = medsam_infer_decoder(self.medsam_model, image_embedding, box_1024, H, W)
                    medsam_seg = KeepMaxContour(medsam_seg)
                    area.append(int(medsam_seg.sum()))
                    seg.append(medsam_seg)
                    self.visualizer.print(f'{box}, {area[-1]}')
                    #if abs(area[step+2] - area[step]) < self.tol:###
                    if (self.diff_area(seg[step+2], seg[step]) < self.tol_twostep) or (self.diff_area(seg[step+2], seg[step+1]) < self.tol_onestep):
                        medsam_seg_new = self.refine_stage2(medsam_seg, image_embedding, H, W, cls, strict=True)
                        
                        if medsam_seg_new is None:
                            break
                        else:
                            medsam_seg_final = torch.from_numpy(medsam_seg_new).cuda()
                        
                        ctr = self.compute_center(image_embedding, medsam_seg_final, H, W)
                        
                        centers[cls].append(ctr)
                        #preds[ind, cls]
                        break
                    #diffuse
                    region = aff_indices[region][aff_dist[region]>self.aff_thresh].unique()
                #time.sleep(12e4)###
                continue
        return
    
    def compute_center(self, image_embedding, medsam_seg, H, W):
        fea=F.interpolate(image_embedding,size=(H, W),mode='bilinear')
        fea = F.normalize(fea, p=2, dim=1)
        
        fea=fea[0]
        fea=fea.reshape((fea.shape[0],-1))
        fea = fea.permute(1,0)

        medsam_seg_1d = medsam_seg.reshape((-1,))
        ctr = torch.mean(fea[medsam_seg_1d==1], dim=0, keepdim=True)
        ctr = F.normalize(ctr, p=2, dim=1)
        return ctr.squeeze()

    def train_sam(self):
        other_time=0
        for epoch in range(self.start_epoch,self.total_epochs):
            train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
            train_losses = {}
            for it, (images,segs,_) in enumerate(train_iterator):
                # pdb.set_trace()
                images = images.to(self.opt['gpu_id'])
                segs = segs.to(self.opt['gpu_id'])
                
                with self.iter_counter.time_measurement("train"):
                    predicts, losses = self.train_one_step([images, segs])
                    for k,v in losses.items():
                        train_losses[k] = v + train_losses.get(k,0) 
                    train_iterator.set_description(f'Train Epoch [{epoch}/{self.total_epochs}]')
                    #train_iterator.set_postfix(ce_loss = train_losses['train_ce'].item()/(it+1), dc_loss = train_losses['train_dc'].item()/(it+1), total_loss = train_losses['train_total'].item()/(it+1))
                    train_iterator.set_postfix(total_loss = train_losses['total_loss'].item()/(it+1))
                
                with self.iter_counter.time_measurement("maintenance"):
                    if self.iter_counter.needs_displaying():
                        
                        if isinstance(predicts, dict):
                            predicts = predicts['seg']
                        """ visuals = {'images':images[:,1].detach().cpu().numpy(),'preds':torch.argmax(predicts,dim=1).detach().cpu().numpy(),
                                   'gt_segs':segs.detach().cpu().numpy()}
                        self.visualizer.display_current_results(self.iter_counter.steps_so_far,visuals) """
                        self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, losses)
                
                self.iter_counter.record_one_iteration()
            self.iter_counter.record_one_epoch()
            start_time = time.time()
            if self.iter_counter.needs_evaluation():
                val_losses = None
                val_metrics = {}
                sample_dict = {}
                sample_list = []
                val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))
                for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

                    val_imgs = val_imgs.to(self.opt['gpu_id'])
                    val_segs = val_segs.to(self.opt['gpu_id'])

                    if val_losses is None:
                        _, predict = self.validate_one_step([val_imgs, val_segs])
                        preds = torch.argmax(predict,dim=1)
                    else:
                        predict, losses = self.validate_one_step([val_imgs, val_segs])
                        for k,v in losses.items():
                            val_losses[k] += v
                    val_iterator.set_description(f'Eval Epoch [{epoch}/{self.total_epochs}]')
                    #val_iterator.set_postfix(ce_loss = val_losses['val_ce'].item()/(it+1), dc_loss = val_losses['val_dc'].item()/(it+1))

                    if 'eval_2d' not in self.opt:
                        for i,name in enumerate(val_names):

                            sample_name,index = name.split('_')[0],int(name.split('_')[1])
                            sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(preds[i].detach().cpu(),val_segs[i].detach().cpu(),index)]
                    else:
                        for i,name in enumerate(val_names):
                            sample_list.append((preds[i].detach().cpu(),val_segs[i].detach().cpu()))    
                #for k, v in val_losses.items():
                #    val_losses[k] = v/(len(self.val_dataloader)+1)   
                    
                if 'eval_2d' not in self.opt:
                    pred_results_list = []
                    gt_segs_list = []
                    for k in sample_dict.keys():

                        sample_dict[k].sort(key=lambda ele: ele[2])
                        preds = []
                        targets = []
                        for pred,target,_ in sample_dict[k]:
                            #if target.sum()==0:
                            #    continue
                            preds.append(pred)
                            targets.append(target)
                        pred_results_list.append(torch.stack(preds,dim=-1))
                        gt_segs_list.append(torch.stack(targets,dim=-1))
                else:
                    pred_results_list = [t[0] for t in sample_list]
                    gt_segs_list = [t[1] for t in sample_list]
                
                if 'Curvas' in self.opt['experiment_name']:
                    pred_results_list_new = []
                    for pred in pred_results_list:
                        pred[pred==2] = 1
                        pred[pred==3] = 2

                val_metrics['dice'] = mean_dice_new(pred_results_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])   
                val_metrics['asd'] = mean_asd(pred_results_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])        
  

                if val_metrics['dice']['dice_avg'] > self.best_avg_dice:
                    self.best_avg_dice = val_metrics['dice']['dice_avg']
                    #self.save_best_models(self.iter_counter.epochs_so_far,val_metrics['dice']['dice_avg'])
                #else:
                if self.iter_counter.needs_saving():
                    self.save_models(self.iter_counter.epochs_so_far,val_metrics['dice']['dice_avg'])
                
                #self.visualizer.plot_current_losses(self.iter_counter.epochs_so_far, val_losses)
                self.visualizer.plot_current_metrics(self.iter_counter.epochs_so_far, val_metrics['dice'],'Dice_metrics')
                self.visualizer.plot_current_metrics(self.iter_counter.epochs_so_far, val_metrics['asd'],'Asd_metrics')
            self.schedular.step()
            
            end_time = time.time()
            other_time = other_time +(end_time - start_time)
        #self.visualizer.print(f"100epoch other time: {other_time:.2f} s")
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_fea(opt, imgs, feas, segs_256, weights, names, preds, index):
    cls=2
    feas=F.interpolate(feas,size=(64,64),mode='bilinear').reshape((feas.shape[0],feas.shape[1],-1))
    feas = F.normalize(feas, p=2, dim=1)
    segs=F.interpolate(segs_256.float().unsqueeze(1),size=(64,64),mode='nearest').long().squeeze()
    segs_1d=segs.reshape((segs.shape[0],-1))
    #print(feas.shape,segs.shape)#torch.Size([16, 64, 4096]) torch.Size([16, 64, 64])
    
    feas = feas.permute(0,2,1)
    weights = F.normalize(weights, p=2, dim=1)

    label2cls = {0:'background',1:'Spleen',2:'R.Kidney',3:'L.Kidney',4:'Liver'}
    for ind in range(feas.shape[0]):
        if ind!=index:
            continue
        img = imgs[ind]#h,w
        fea = feas[ind]#hw,c
        fea = torch.cat((fea, weights), 0)
        seg = segs[ind]#h,w
        seg_1d = segs_1d[ind]#hw

        #print(ind,seg.unique())
        '''
        if (seg==cls).sum()==0:
            continue
        h_indices, w_indices = np.where((seg==cls).numpy())
        h_min, h_max = np.min(h_indices), np.max(h_indices)
        w_min, w_max = np.min(w_indices), np.max(w_indices)

        seg_bbox = np.zeros_like(seg)
        seg_bbox[h_min:h_max+1,w_min:w_max+1] = 1
        seg_bbox_1d = seg_bbox.reshape(-1)
        '''
        #print(fea[-5:])
        proj = TSNE().fit_transform(fea.numpy())
        #print(proj[-5:])
        #sys.exit()
        weight = proj[-5:]#self.opt['num_classes']
        proj = proj[:-5]

        fig = plt.figure(figsize=(4,4))#(8,8),dpi=400
        """ axes = plt.subplot(221)
        axes.imshow(img) """
        '''
        axes.add_patch(
        plt.Rectangle((w_min*4, h_min*4), (w_max-w_min)*4, (h_max-h_min)*4, edgecolor="m", facecolor=(0, 0, 0, 0), lw=2)
        )
        '''
        """ axes = plt.subplot(222)
        axes.imshow(segs_256[ind]) """
        
        colors = ['c', 'b', 'm', 'y', 'r', 'g', 'k']
        markers = ['o','o','o','^','^','^']

        axes = plt.subplot()#223
        plt.xticks(())
        plt.yticks(())
        #print(proj[:,0].min(),proj[:,0].max(),proj[:,1].min(),proj[:,1].max())
        #print(weight)
        for i in range(5):
            if (seg_1d==i).sum()>0:
                select = seg_1d == i
                sc = axes.scatter(proj[:,0][select],proj[:,1][select],lw=0,s=1,
                                c=colors[i], label=label2cls[i])#marker='o',i
            #plt.legend()
            """ sc = axes.scatter(weight[i,0],weight[i,1],s=35,
                            c=colors[i], marker='*',zorder=1) """    
        '''
        axes = plt.subplot(224)
        plt.xticks(())
        plt.yticks(())
        for i in range(2):

            select = seg_bbox_1d == i
            sc = axes.scatter(proj[:,0][select],proj[:,1][select],s=2,
                            c=colors[i], marker='o', label=i)
            plt.legend()
        '''
        """ axes = plt.subplot(224)
        axes.imshow(preds[ind]) """
        fig.tight_layout()
        plt.savefig(os.path.join('/home/zhuaiaa/medsam/MedSAM/pl_ablation', f'{names[ind]}', f'{names[ind]}.png'))
        plt.savefig(os.path.join('/home/zhuaiaa/medsam/MedSAM/pl_ablation', f'{names[ind]}', f'{names[ind]}.svg'))
        plt.close(fig)

