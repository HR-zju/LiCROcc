import torch
from torchmetrics.metric import Metric
import numpy as np

class SSCMetrics(Metric):
    def __init__(self, n_classes=20, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        
        self.n_classes = n_classes
        
        self.add_state('tps', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        self.add_state('fps', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        self.add_state('fns', default=torch.zeros(
            self.n_classes), dist_reduce_fx='sum')
        
        self.add_state('completion_tp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fp', default=torch.zeros(1), dist_reduce_fx='sum')
        self.add_state('completion_fn', default=torch.zeros(1), dist_reduce_fx='sum')
    
    def compute_single(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask_0 = y_true != 255
        mask_dis = y_true == 50
        
        # 添加 mask 
        # 获取 x 和 y 轴的范围  
        x_range = torch.arange(-51.2, 51.2, (51.2 * 2) / 512)  
        y_range = torch.arange(-51.2, 51.2, (51.2 * 2) / 512)  
        
        # 获取范围内的索引  
        x_indices = ((x_range >= -20) & (x_range < 20)).nonzero()  
        y_indices = ((y_range >= -20) & (y_range < 20)).nonzero()  # 8400

        # x_indices = ((abs(x_range) >= 20) & (abs(x_range) < 30)).nonzero()  
        # y_indices = ((abs(y_range) >= 20) & (abs(y_range) < 30)).nonzero()  

        # x_indices = ((abs(x_range) >= 30) & (abs(x_range) < 50)).nonzero()  
        # y_indices = ((abs(y_range) >= 30) & (abs(y_range) < 50)).nonzero()  # 4000

        # 更新 mask  
        # for x in x_indices:
        #     for y in y_indices:
        #         mask_dis[:, x, y, :] = True 

        x_indices = np.array(x_indices)  
        y_indices = np.array(y_indices)  
        xx, yy = np.meshgrid(x_indices, y_indices, indexing='ij')  
        mask_dis[:, xx, yy, :] = True 
        # mask[:, (x_indices == False), (y_indices == False), :] = False  
        # breakpoint()
        mask = mask_0  #&  mask_dis 
        # breakpoint()


        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface

        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        # # evaluate semantic completion
        # mask = y_true != 255
        mask = mask_0  #& mask_dis 

        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        
        ret = (tp.cpu().numpy(), fp.cpu().numpy(), fn.cpu().numpy(), tp_sum.cpu().numpy(), fp_sum.cpu().numpy(), fn_sum.cpu().numpy())
        
        return ret
        
    def update(self, y_pred, y_true, nonempty=None, nonsurface=None):
        # evaluate completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        # todo在这里更改mask
        # breakpoint()
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn
        
        # # evaluate semantic completion
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum
    
    def compute(self):
        precision = self.completion_tp / (self.completion_tp + self.completion_fp)
        recall = self.completion_tp / (self.completion_tp + self.completion_fn)
        iou = self.completion_tp / \
                (self.completion_tp + self.completion_fp + self.completion_fn)
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        
        output = {
            "precision": precision,
            "recall": recall,
            "iou": iou.item(),
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": iou_ssc[1:].mean().item(),
        }
        
        return output

    def get_score_completion(self, predict, target, nonempty=None):
        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = torch.zeros_like(predict)
        b_true = torch.zeros_like(target)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = torch.sum((y_true == 1) & (y_pred == 1))
            fp = torch.sum((y_true != 1) & (y_pred == 1))
            fn = torch.sum((y_true == 1) & (y_pred != 1))
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.view(_bs, -1)  # (_bs, 129600)
        predict = predict.view(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        tp_sum = torch.zeros(_C).type_as(predict)
        fp_sum = torch.zeros(_C).type_as(predict)
        fn_sum = torch.zeros(_C).type_as(predict)

        for idx in range(_bs):
            y_true = target[idx]  # GT
            y_pred = predict[idx]
            
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].view(-1)
                valid_mask = (nonempty_idx == 1) & (y_true != 255)
                y_pred = y_pred[valid_mask]
                y_true = y_true[valid_mask]
            
            for j in range(_C):  # for each class
                tp = torch.sum((y_true == j) & (y_pred == j))
                fp = torch.sum((y_true != j) & (y_pred == j))
                fn = torch.sum((y_true == j) & (y_pred != j))
                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum