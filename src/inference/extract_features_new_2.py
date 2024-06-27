import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel

from data.dataset import AbdominalInfDataset
from data.transforms import get_transfos
from data.loader import define_loaders
from model_zoo.models import define_model
from util.metrics import rsna_loss
from util.torch import load_model_weights, sync_across_gpus
from params import IMAGE_TARGETS

def print_memory_usage():
    # Get total system memory and current memory used
    svmem = psutil.virtual_memory()
    print(f"Total: {svmem.total / (1024 ** 3):.2f} GB", f"Used : {svmem.used / (1024 ** 3):.2f} GB")
 

def save_results(preds_accumulator, fts_accumulator, batches_processed, exp_folder, fold_name):
    """
    Save accumulated predictions and features to numpy files with unique filenames.

    Args:
        preds_accumulator (list): List of accumulated prediction arrays.
        fts_accumulator (list): List of accumulated feature arrays.
        batches_processed (int): Total number of batches processed.
        exp_folder (str): Path to the experiment folder.
        fold_name (str): Name of the fold for saving the files.
    """
    for i, (preds, fts) in enumerate(zip(preds_accumulator, fts_accumulator)):
        # Generate unique filenames
        preds_file = exp_folder + f"pred_val_batch_{fold_name}_{batches_processed}_{i}.npy"
        fts_file = exp_folder + f"fts_val_batch_{fold_name}_{batches_processed}_{i}.npy"
        
        np.save(preds_file, preds)
        np.save(fts_file, fts)
        
    preds_accumulator = []
    fts_accumulator = [] 
    memory_used = print_memory_usage()
    print(f"Saved predictions and features after {batches_processed} batches {memory_used}")

 
class Config:
    """
    Placeholder to load a config from a saved json
    """
    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


def predict_distributed(
    model,
    dataset,
    loss_config,
    batch_size=64,
    use_fp16=False,
    num_workers=8,
    distributed=True,
    world_size=0,
    local_rank=0,
    exp_folder=None, 
):
    """
    Make predictions using a PyTorch model on a given dataset.
    Uses DDP.

    Args:
        model (nn.Module): PyTorch model for making predictions.
        dataset (torch.utils.data.Dataset): Dataset used for prediction.
        loss_config (dict): Configuration for the loss function.
        batch_size (int, optional): Batch size for inference. Defaults to 64.
        use_fp16 (bool, optional): Whether to use mixed-precision (fp16) inference. Defaults to False.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        distributed (bool, optional): Whether to use distributed inference. Defaults to True.
        world_size (int, optional): Number of GPUs used in distributed inference. Defaults to 0.
        local_rank (int, optional): Local rank for distributed inference. Defaults to 0.

    Returns:
        Tuple: If `local_rank` is 0, returns a tuple containing:
            - preds (numpy.ndarray): Predictions from the model.
            - fts (numpy.ndarray): Extracted features from the model.
        Otherwise, returns (0, 0) for non-zero `local_rank`.
    """
    model.eval()
    preds, fts = [], []

    loader = define_loaders(
        dataset,
        dataset,
        batch_size=batch_size,
        val_bs=batch_size,
        num_workers=num_workers,
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )[1]
    # variable to save result 
    batches_processed = 0 
    save_every = 200 
    preds_accumulator, fts_accumulator = [], [] 
    save_counter = 0
    
    with torch.no_grad():
        for img, _, _ in tqdm(loader, disable=(local_rank != 0)):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, ft = model(img.cuda(), return_fts=True)

            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)
            elif loss_config["activation"] == "patient":
                y_pred[:, :2] = y_pred[:, :2].sigmoid()
                y_pred[:, 2:5] = y_pred[:, 2:5].softmax(-1)
                y_pred[:, 5:8] = y_pred[:, 5:8].softmax(-1)
                y_pred[:, 8:] = y_pred[:, 8:].softmax(-1)

            preds_accumulator.append(y_pred.detach())
            fts_accumulator.append(ft.detach())
            save_counter+=1
            
            if save_counter % save_every == 0:
                save_results(
                        preds_accumulator, fts_accumulator, batches_processed, exp_folder, fold_name=fold_name
                    )
                # Reset accumulators and counter
                preds_accumulator, fts_accumulator = [], [] 
    # Final save of remaining results
    
    if preds_accumulator:
        save_results(
            preds_accumulator, fts_accumulator, batches_processed, exp_folder, fold_name=fold_name
        ) 
        preds_accumulator, fts_accumulator = [], []  
    
     
                
    # preds = torch.cat(preds, 0)
    # fts = torch.cat(fts, 0)

    # if distributed:
    #     fts = sync_across_gpus(fts, world_size)
    #     preds = sync_across_gpus(preds, world_size)
    #     torch.distributed.barrier()

    # if local_rank == 0:
    #     preds = preds.cpu().numpy()
    #     fts = fts.cpu().numpy()
    #     return preds, fts
    # else:
    #     return 0, 0


def kfold_inference(
    df_patient,
    df_img,
    exp_folder,
    debug=False,
    use_fp16=False,
    save=False,
    num_workers=8,
    batch_size=None,
    distributed=False,
    config=None,
    save_after=100,
):
    """
    Perform k-fold inference on a dataset using a trained model.

    Args:
        df_patient (pd.DataFrame): Dataframe containing patient information.
        df_img (pd.DataFrame): Dataframe containing image information.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): If True, enable debug mode. Defaults to False.
        use_fp16 (bool, optional): Whether to use mixed-precision (fp16) inference. Defaults to False.
        save (bool, optional): If True, save inference results. Defaults to False.
        num_workers (int, optional): Number of workers for data loading. Defaults to 8.
        batch_size (int, optional): Batch size for inference. Defaults to None.
        distributed (bool, optional): Whether to use distributed inference. Defaults to False.
        config (Config, optional): Configuration object for the experiment. Defaults to None.
    """
    if config is None:
        config = Config(json.load(open(exp_folder + "config.json", "r")))

    if "fold" not in df_patient.columns:
        folds = pd.read_csv(config.folds_file)
        df_patient = df_patient.merge(folds, how="left")
        df_img = df_img.merge(folds, how="left")

    for fold in config.selected_folds:
        if config.local_rank == 0:
            print(f"\n- Fold {fold + 1}")

        model = define_model(
            config.name,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            use_gem=config.use_gem,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            n_channels=config.n_channels,
            reduce_stride=config.reduce_stride,
            increase_stride=config.increase_stride if hasattr(config, "increase_stride") else False,
            pretrained=False,
        )
        model = model.cuda().eval()

        weights = exp_folder + f"{config.name}_{fold}.pt"
        model = load_model_weights(model, weights, verbose=config.local_rank == 0)

        if distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[config.local_rank],
                find_unused_parameters=False,
                broadcast_buffers=False,
            )

        df_val = df_img[df_img["fold"] == fold].reset_index(
            drop=True
        )

        transforms = get_transfos(
            augment=False,
            resize=config.resize,
            crop=config.crop,
        )

        dataset = AbdominalInfDataset(
            df_val,
            transforms=transforms,
            frames_chanel=config.frames_chanel if hasattr(config, "frames_chanel") else 0,
            n_frames=config.n_frames if hasattr(config, "n_frames") else 1,
            stride=config.stride if hasattr(config, "stride") else 1,
        )
        
        import os
        import glob 
            # Clear previous run  results for this fold
        for f in glob.glob(exp_folder + f"pred_val_batch_{fold}_*.npy"):
            os.remove(f)
        for f in glob.glob(exp_folder + f"fts_val_batch_{fold}_*.npy"):
            os.remove(f)  
        print("done remove old data, start runing prediction")
        predict_distributed(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"] if batch_size is None else batch_size,
            use_fp16=use_fp16,
            num_workers=num_workers,
            distributed=True,
            world_size=config.world_size,
            local_rank=config.local_rank,
        )
        print("Done inference,!!!! Start loading the data")
        for preds_file in sorted(glob.glob(exp_folder + f"pred_val_batch_{fold}_*.npy")):
            preds.append(np.load(preds_file))
    
        for fts_file in sorted(glob.glob(exp_folder + f"fts_val_batch_{fold}_*.npy")):
            fts.append(np.load(fts_file))

        preds = np.concatenate(preds, axis=0)
        fts = np.concatenate(fts, axis=0) 
         
        
        if config.local_rank == 0:
            pred, fts = pred[: len(dataset)], fts[: len(dataset)]

        if save and config.local_rank == 0:
            np.save(exp_folder + f"pred_val_{fold}.npy", pred)

            pred_cols = []
            for i, tgt in enumerate(IMAGE_TARGETS):
                df_val[f"pred_{tgt}"] = pred[: len(df_val), i]
                pred_cols.append(f"pred_{tgt}")
            df_val_patient = (
                df_val[["patient_id"] + pred_cols].groupby("patient_id").mean()
            )

            df_val_patient = df_val_patient.merge(
                df_patient[df_patient["fold"] == fold], on="patient_id", how="left"
            )

            print()
            for tgt in IMAGE_TARGETS:
                auc = roc_auc_score(df_val_patient[tgt], df_val_patient[f"pred_{tgt}"])
                print(f"- {tgt} auc : {auc:.3f}")

            losses, avg_loss = rsna_loss(
                df_val_patient[
                    ["pred_bowel_injury", "pred_extravasation_injury"]
                ].values,
                df_val_patient,
            )
            for k, v in losses.items():
                print(f"- {k.split('_')[0][:8]} loss\t: {v:.3f}")