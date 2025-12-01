import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
from matplotlib.colors import ListedColormap
from PIL import Image
from models.nn_unet import NnUNet
from models.unetr import UNETR2D
from models.swin_unet import SwinUNet
from models.pairwise_conv_avg_model import PairwiseConvAvgModel
from models.NoiseContext import ContextNoiseUNet
import os
import matplotlib.pyplot as plt
from util.shapecheck import ShapeChecker
from functools import lru_cache
from scipy.ndimage import distance_transform_edt as distance_transform
from glob import glob
import pandas as pd
from dataloaders import reading_data_tg3k,reading_training_data_fetal, reading_camus_data, reading_busi_data, reading_busbra, get_data_jnu,get_frame_labels, read_and_split_busi_data, read_and_split_busbra_data, read_data_jnu
from models.nn_unet import NnUNet
from models.unetr import UNETR2D
from models.swin_unet import SwinUNet
from models.NoiseContext import ContextNoiseUNet
from models.multiverseg.models.network import MultiverSegNet

cmap = ListedColormap([ # 1: TP
    'black',
    "green",    # 2: FP (green)
    "red"      # 3: FN
])

class LightningModel(pl.LightningModule):
    def __init__(self,net,name):
        super().__init__()
        self.best_dice = 0.0
        self.save_dir="exp_1"
        os.makedirs(self.save_dir, exist_ok=True)

        self.net = net
        self.name = name.lower()
   
        
        # Loss functi
    def forward(self, target, context_in, context_out):
        sc = ShapeChecker()
        y_pred = self.net(target, context_in, context_out)
       
        #y_pred = self.net(target_in)
        sc.check(y_pred, "B C H W")
        return y_pred 

    def training_step(self, batch, batch_idx):
        target_images, target_masks, context_images, context_masks = batch
        target_images=target_images.to("cuda")
        target_masks=target_masks.to("cuda")
        context_images=context_images.to("cuda")
        context_masks=context_masks.to("cuda")
        # print(f"target_images.shape: {target_images.shape}, target_masks.shape: {target_masks.shape}, context_images.shape: {context_images.shape}, context_masks.shape: {context_masks.shape}")
        # exit()
        #gt_dt = masks_to_distance_maps(target_masks.squeeze(1)).to("cuda")
        pred_masks = self(target_images.squeeze(1), context_images.squeeze(1), context_masks.squeeze(1))        #edge_pred = sobel_edges(pred_masks)
        #edge_gt   = sobel_edges(target_masks.squeeze(1).float())
        #edge_loss = F.mse_loss(pred_dt, gt_dt, reduction='mean')
        #edge_pred = boundary_mask(pred_masks > 0.5)
        #edge_gt   = boundary_mask(target_masks.squeeze(1))
        #shape_loss = shape_prior_loss(pred_masks,target_masks.squeeze(1))
        # Dice-style overlap on boundaries
        #intersection = (edge_pred * edge_gt).sum()
        #denom = edge_pred.sum() + edge_gt.sum() + 1e-6
        #boundary_loss = 1 - (2 * intersection / denom)
        #loss = self.criterion(pred_masks, target_masks.squeeze(1))  + 0.1*edge_loss
        loss = self.criterion(pred_masks,target_masks.squeeze(1))
        metrics = self._calculate_metrics(pred_masks, target_masks)
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_images, target_masks, context_images, context_masks = batch
        # Forward pass
        pred_masks = self(target_images.squeeze(1), context_images.squeeze(1), context_masks.squeeze(1))
        # Calculate loss
        #shape_loss = shape_prior_loss(pred_masks,target_masks.squeeze(1))

        loss = self.criterion(pred_masks, target_masks.squeeze(1)) 
        # Calculate metrics
        metrics = self._calculate_metrics(pred_masks, target_masks)

        # Log everything
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)


        return loss


    def test_step(self, batch, batch_idx):
        target_images, target_masks, context_images, context_masks = batch

        # Forward pass
        pred_masks = self(target_images.squeeze(1), context_images.squeeze(1), context_masks.squeeze(1))

        # Calculate loss
        loss = self.criterion(pred_masks, target_masks.squeeze(1)) 

        # Calculate metrics
        metrics = self._calculate_metrics(pred_masks, target_masks)

        # Log everything
        self.log_dict({f"test_{k}": v for k, v in metrics.items()})
        self.log("test_loss", loss)
        img = target_images[0,0].detach().cpu()     # shape [C,H,W] or [H,W]
        gt_mask = target_masks[0,0].detach().cpu()
        pred_mask = pred_masks[0,0].detach().cpu()  # convert logits to [0,1]

        # convert to numpy
        img_np = img.permute(1,2,0).numpy() if img.ndim==3 else img.numpy()
        gt_np = gt_mask.squeeze().numpy()
        pred_np = (pred_mask > 0.5).squeeze().numpy().astype(float)  # binarize
        # Rotate all images 90° clockwise
        #img_np = np.rot90(img_np, k=3)   # 90° clockwise
        #gt_np = np.rot90(gt_np, k=3)
        #pred_np = np.rot90(pred_np, k=3)
        metrics_path = os.path.join(self.save_dir, "test_metrics.txt")

# Append Dice and IoU for each sample
        with open(metrics_path, "a") as f:
            f.write(f" Batch {batch_idx} "
                    f"Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}\n")
        # plot side by side
        fig, axes = plt.subplots(1,3, figsize=(12,4))
        axes[0].imshow(img_np, cmap="gray")
        axes[0].set_title("Input Image")
        axes[1].imshow(gt_np, cmap="gray")
        axes[1].set_title("GT Mask")
        axes[2].imshow(pred_np, cmap="gray")
        axes[2].set_title("Predicted Mask")
        for ax in axes: ax.axis("off")

        # save
        save_path = os.path.join(self.save_dir, f"sample_{self.current_epoch}_{batch_idx}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)

        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                              lr=1e-5,
                              weight_decay=1e-7)
        return optimizer

    def _calculate_metrics(self, pred_logits, target, save_dir="outputs", batch_idx=0):
        target = target.squeeze(1).squeeze(1)
        pred_logits = pred_logits.squeeze(1)
        pred = torch.sigmoid(pred_logits)
        
        preds = pred > 0.5
        targets = target > 0.5
        smooth = 1e-6  # Smoothing factor to avoid division by zero
        
        # Calculate TP, FP, FN, TN
        tp = (preds & targets).float().sum()
        fp = (preds & ~targets).float().sum()
        fn = (~preds & targets).float().sum()
        tn = (~preds & ~targets).float().sum()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + fp + fn + tn + smooth)
        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        specificity = tn / (tn + fp + smooth)
        iou = tp / (tp + fp + fn + smooth)
        dice = (2. * tp + smooth) / (preds.float().sum() + targets.float().sum() + smooth)
 
        # Visualization
        os.makedirs(save_dir, exist_ok=True)
        pred_np = preds[0].cpu().detach().numpy()
        target_np = targets[0].cpu().detach().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(pred_np, cmap='gray')
        ax1.set_title('Prediction')
        ax1.axis('off')
        
        ax2.imshow(target_np, cmap='gray')
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        metrics = {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item(),
            'iou': iou.item(),
            'dice': dice.item(),
        }

        plt.suptitle(f'Batch {batch_idx} - Dice: {metrics["dice"]:.3f}')
        plt.tight_layout()
        plt.savefig("displayed_image.png", bbox_inches='tight', pad_inches=0)
        plt.close()
        
        return metrics




class TrainDataset(Dataset):
    def __init__(self, data, context_dataset, context_size=4, return_context_only=False, device='cuda'):
        self.data = data
        self.context_size = context_size
        self.return_context_only = return_context_only
        self.device = device
        self.context_dataset = context_dataset
       
        
        assert len(self.data) >= context_size + 1

    def __len__(self):
        return len(self.data) - self.context_size

           

    def __getitem__(self, idx):
        if self.return_context_only:
            # When used as context dataset, only return img and mask
            img, mask = self.data[idx]
            img = torch.tensor(np.ascontiguousarray(img), dtype=torch.float32, device="cpu").unsqueeze(0)
            mask = torch.tensor(np.ascontiguousarray(mask), dtype=torch.float32, device="cpu").unsqueeze(0)

    #        return img, mask
        else:
            # Get target sample
            target_img, target_mask = self.data[idx]
            target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32, device="cpu").unsqueeze(0)  # [1, C, H, W]
            target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32, device="cpu").unsqueeze(0)  # [1, C, H, W]

            
            # Get context samples (different from target and sequential)
            # We'll take the next 'context_size' samples after the target
            context_indices = range(idx + 1, idx + 1 + self.context_size)
            
            context_imgs = []
            context_masks = []
            for context_idx in context_indices:
                c_img, c_mask = self.data[context_idx]
                c_img = torch.tensor(np.ascontiguousarray(c_img), dtype=torch.float32, device='cpu').unsqueeze(0)  # [1, C, H, W]
                c_mask = torch.tensor(np.ascontiguousarray(c_mask), dtype=torch.float32, device='cpu').unsqueeze(0) 
                context_imgs.append(c_img)
                context_masks.append(c_mask)
                
            # Stack context samples along the sequence dimension
            context_img = torch.stack(context_imgs, dim=0)  # [4, 1, C, H, W]
            context_mask = torch.stack(context_masks, dim=0)  # [4, 1, C, H, W]            
            

        
            # Add batch dimension and adjust dimensions
            context_img = context_img.unsqueeze(0)  # [1, 4, 1, H, W]
            context_mask = context_mask.unsqueeze(0)  # [1, 4, 1, H, W]

            return (
            target_img.unsqueeze(0),       # [1, 4, H, W]
            target_mask.unsqueeze(0),            # [1, 1, H, W]
            context_img,      # [1, 4, 4, H, W]
            context_mask       # [1, 4, 1, H, W]
            )


        


class EvalDataset(Dataset):
    """Evaluation dataset with same channel padding as TrainDataset and 4 context samples"""
    def __init__(self, target_data, context_dataset, context_size=4):
        self.target_data = target_data
        self.context_dataset = context_dataset
        self.context_size = context_size
   
    def __len__(self):
        return len(self.target_data)
    
    def __getitem__(self, idx):
        # Process target (with padding)
        target_img, target_mask = self.target_data[idx]

        target_img = torch.tensor(np.ascontiguousarray(target_img), dtype=torch.float32)  # [H, W]
        target_mask = torch.tensor(np.ascontiguousarray(target_mask), dtype=torch.float32)  # [H, W]

    

   
        # --- Find k closest context samples based on L2 distance ---
        distances = []
        for context_img, context_mask in self.context_dataset:         
            ctx_tensor = torch.tensor(np.ascontiguousarray(context_img), dtype=torch.float32)
            distances.append(torch.norm(target_img - ctx_tensor).item())
            
        sorted_indices = np.argsort(distances)[:self.context_size]

        # Select the top-k most similar context samples
        context_imgs = []
        context_masks = []
        for i in sorted_indices:
            ctx_img, ctx_mask = self.context_dataset[i]
            context_imgs.append(np.ascontiguousarray(ctx_img))
            context_masks.append(np.ascontiguousarray(ctx_mask))

        # Convert to tensors
        context_imgs_tensor = torch.stack([
            torch.tensor(img, dtype=torch.float32) for img in context_imgs
        ])  # [C, H, W]
        context_masks_tensor = torch.stack([
            torch.tensor(mask, dtype=torch.float32) for mask in context_masks
        ])  # [C, H, W]
    
  
        return (
            target_img.unsqueeze(0).unsqueeze(0),
            target_mask.unsqueeze(0).unsqueeze(0),
            context_imgs_tensor.unsqueeze(1) ,
            context_masks_tensor.unsqueeze(1)  
        )

class UltrasoundDataModule(LightningDataModule):
    def __init__(self, X_train, X_init, X_val, X_test, batch_size=32,num_workers=0,no_edges=True):
        super().__init__()
        self.X_train = X_train
        self.X_init = X_init
        self.X_val = X_val
        self.X_test = X_test
        self.batch_size = batch_size
        self.num_workers=num_workers
        self.no_edges=no_edges
    def setup(self, stage=None):
        # Training set
        self.train_dataset = TrainDataset(self.X_train, self.X_init,context_size=4)

        
        
        # Validation/Test sets
        self.val_dataset = EvalDataset(self.X_val, self.X_init, context_size=4)
        self.test_dataset = EvalDataset(self.X_test, self.X_init, context_size=4)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,num_workers=self.num_workers)


def calculate_metrics(pred_logits, target,name):
        if name.lower() == "sam2-sgp":
            target = target.squeeze(1)
        else :             
            target = target.squeeze(1).squeeze(1)

        pred_logits = pred_logits.squeeze(1)
        pred = pred_logits
        preds = pred > 0.5
        targets = target > 0.5
        smooth = 1e-6  # Smoothing factor to avoid division by zero
        
        # Calculate TP, FP, FN, TN
        tp = (preds & targets).float().sum()
        fp = (preds & ~targets).float().sum()
        fn = (~preds & targets).float().sum()
        tn = (~preds & ~targets).float().sum()

        iou = tp / (tp + fp + fn + smooth)
        dice = (2. * tp + smooth) / (preds.float().sum() + targets.float().sum() + smooth)
 
        # Visualization
        
        
        return dice,iou




# ------------------------------------------------------------------
# Unified model wrapper (auto-detect Lightning vs raw checkpoint)
# ------------------------------------------------------------------

class BaseSegModel(torch.nn.Module):
    def __init__(self, model_name, checkpoint_path=None, device="cuda"):
        super().__init__()
        self.model_name = model_name.lower()

        # Initialize architecture (for fallback)
        self.model = self._create_model_architecture(model_name)

        if checkpoint_path and os.path.exists(checkpoint_path):
         

            try:    # Detect Lightning checkpoint
                print(f"⚡ Loading Lightning checkpoint for {model_name}")
                lightning_module = LightningModel.load_from_checkpoint(
                    checkpoint_path,
                    net=self.model,          # provide architecture
                    name=self.model_name     # any args your LightningModel needs
                )
                self.model = lightning_module
    
            except Exception as e:
                print(f"⚠️ Failed to load {model_name} from {checkpoint_path}: {e}")
        else:
            print(f"⚠️ No checkpoint found for {model_name}, using random weights.")

        self.model.to(device)
        self.model.eval()

    def _create_model_architecture(self, name):
        name = name.lower()
        if name == "multiversegnet":
            return MultiverSegNet(encoder_blocks=[64, 64, 64, 64])
        elif name == "icl-noiseunet":
            return ContextNoiseUNet()
        elif name == "nn-unet":
            return NnUNet(in_channels=1, out_channels=1, base_num_features=16, num_pool=4, ndim=2, deep_supervision=False)
        elif name == "unet-transformer":
            return UNETR2D(img_shape=(192, 192), input_dim=1, output_dim=1)
        elif name == "swin-unet":
            return SwinUNet()
        else:
            raise ValueError(f"Unknown model architecture: {name}")

    @torch.no_grad()
    def forward(self, target, context_in=None, context_out=None):
            pred = self.model(target, context_in, context_out)
            return pred

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())

def model_size_mb(model: nn.Module):
    # assumes float parameters
    param_count = sum(p.numel() for p in model.parameters())
    bytes_per_param = next(model.parameters()).element_size()  # usually 4 bytes FP32 or 2 bytes FP16
    total_bytes = param_count * bytes_per_param
    return total_bytes / (1024 * 1024)

def test_all_models(data_module, checkpoints, save_dir="comparisons", device="cuda"):
    os.makedirs(save_dir, exist_ok=True)

    # Load all models together
    print("\n🧩 Loading all models...\n")
    models = {
        name: BaseSegModel(name, ckpt, device=device)
        for name, ckpt in checkpoints.items()
    }


    test_loader = data_module.test_dataloader()
    metrics_path = os.path.join(save_dir, "metrics.csv")

    with open(metrics_path, "w") as f:
        f.write("batch_idx,model,dice,iou\n")

    print("🚀 Starting model comparison...\n")
    all_metrics = []
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        target, mask, context_img, context_mask = batch
        sam2_pred = torch.ones_like(target)  

        if sam2_pred.sum() > 0:
            img = target[0,0].detach().cpu()
            img_np = img.numpy()
            mask_np = mask[0, 0].detach().cpu().numpy()

            img_np=img_np[0,:,:]
            mask_np=mask_np[0,:,:]
            

            for i, (name, model) in enumerate(models.items(),start=2):
                if name.lower() == "swin-unet":
                    model.eval()
                    target_resized = F.interpolate(target.squeeze(1), size=(224, 224), mode="bilinear", align_corners=False)
                    preds = model(target_resized.to("cuda"),context_img.to("cuda").squeeze(1), context_mask.to("cuda").squeeze(1))
                    preds = F.interpolate(preds, size=(192, 192), mode="bilinear", align_corners=False)
                elif name.lower() == "sam2-sgp": 
                    preds = sam2_pred.to("cuda")
                    print(f"sam2_pred sum: {sam2_pred.sum()}")

                else :
                    model.eval()
                    preds = model(target.to("cuda").squeeze(1), context_img.to("cuda").squeeze(1), context_mask.to("cuda").squeeze(1))
                    
                    
                dice,iou = calculate_metrics(preds, mask.to("cuda"),name)
                dice = dice.item()
                iou = iou.item()
                preds= preds[0,0].detach().cpu()  # convert logits to [0,1]
                pred_np = (preds> 0.5).squeeze().numpy().astype(float)  # binarize
        

                all_metrics.append({"batch_idx": batch_idx, "model": name, "dice": dice, "iou": iou})

                with open(metrics_path, "a") as f:
                    f.write(f"{batch_idx},{name},{dice:.4f},{iou:.4f}\n")

            
                # 3-channel RGB image for overlay
                pixel_map = np.zeros_like(pred_np, dtype=np.uint8)
                pixel_map[(mask_np == 0) & (pred_np == 1)] = 1  # FP
                pixel_map[(mask_np == 1) & (pred_np == 0)] = 2  # FN   
                

            mask_img = Image.fromarray(mask_np*255)
            mask_img = mask_img.convert("L")
            mask_img.save(os.path.join(save_dir,f"mask_batch_{batch_idx:03d}.png"))
            pred_img = Image.fromarray(pred_np*255)
            pred_img = pred_img.convert("L")
            pred_img.save(os.path.join(save_dir,f"pred_batch_{batch_idx:03d}.png"))
            #plt.close(fig)

    print(f"\n✅ Comparison complete! Metrics saved to {metrics_path}\n")
    df = pd.DataFrame(all_metrics)
    df.to_csv(metrics_path, index=False)

    # Compute and save mean Dice/IoU per model
    summary = df.groupby("model")[["dice", "iou"]].mean().reset_index()

    summary_path = os.path.join(save_dir, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
  

    print("\n✅ Comparison complete!")
    print(f"📊 Metrics saved to: {metrics_path}")
    print(f"📈 Summary saved to: {summary_path}\n")


def test_inference_models(data_module, checkpoints, save_dir="comparisons", device="cuda",max_batches=100):
    os.makedirs(save_dir, exist_ok=True)

    # Load all models together
    print("\n🧩 Loading all models...\n")
    models = {
        name: BaseSegModel(name, ckpt, device=device)
        for name, ckpt in checkpoints.items()
    }
    
    test_loader = data_module.test_dataloader()
    metrics_path = os.path.join(save_dir, "metrics.csv")



    print("🚀 Starting model comparison...\n")
    all_metrics = []
    with open(metrics_path, "w") as f:
        f.write("model,inference_time,params,model_size_MB\n")
    for i, (name, model) in enumerate(models.items()):
        model.eval()
        num_params = count_parameters(model)
        size_mb = model_size_mb(model)
        start_time = time.time()   # <--- track total inference start
        valid_count = 0

        for batch_idx, batch in enumerate(test_loader):
            target, mask, context_img, context_mask, sam2_pred = batch
            if sam2_pred.sum() > 0:
                valid_count += 1
                if valid_count > max_batches:
                    break
                with torch.no_grad():
                    preds = model(target.to("cuda").squeeze(1), context_img.to("cuda").squeeze(1), context_mask.to("cuda").squeeze(1))

                    img = target[0,0].detach().cpu()
                    img_np = img.numpy()
                    mask_np = mask[0, 0].detach().cpu().numpy()

                    img_np=img_np[0,:,:]
                    mask_np=mask_np[0,:,:]
                    img_np = np.rot90(img_np, k=3)   # 90° clockwise
                    mask_np = np.rot90(mask_np, k=3)
            
                        
                    dice,iou = calculate_metrics(preds, mask.to("cuda"),name)
                    dice = dice.item()
                    iou = iou.item()
                    preds= preds[0,0].detach().cpu()  # convert logits to [0,1]
                    pred_np = (preds> 0.5).squeeze().numpy().astype(float)  # binarize
                    
                    all_metrics.append({"batch_idx": batch_idx, "model": name, "dice": dice, "iou": iou})

                
            
                # 3-channel RGB image for overlay
        inference_time = time.time() - start_time
        print(valid_count)
        with open(metrics_path, "a") as f:
            f.write(f"{name},{inference_time:.4f},{num_params},{size_mb}\n")

  

    df = pd.DataFrame(all_metrics)

    # Compute and save mean Dice/IoU per model
    summary = df.groupby("model")[["dice", "iou"]].mean().reset_index()
    summary_path = os.path.join(save_dir, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\n✅ Comparison complete!")
    print(f"📊 Metrics saved to: {metrics_path}")
    print(f"📈 Summary saved to: {summary_path}\n")
    print(summary)
# ------------------------------------------------------------------
# Entry Point
# ------------------------------------------------------------------

X, Y, V = read_and_split_busi_data(busi_root='./BUSI')
X_init = X.copy()
data_module = UltrasoundDataModule(X, X_init, V, Y, batch_size=1, num_workers=4, no_edges=True)
data_module.setup()

    # Define all model checkpoints
checkpoints = {
        
        "ICL-NoiseUNet":  "{model_path}",
        "MultiverSegNet":  "{model_path}",
        "nn-UNet": "{model_path}",
        "UNET-Transformer":  "{model_path}",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
test_all_models(data_module, checkpoints, save_dir="model", device=device)

