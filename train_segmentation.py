# general libs
import logging
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

# densetorch wrapper
import densetorch as dt

# configuration for light-weight refinenet
from arguments import get_arguments
from data import get_datasets, get_transforms
from network import get_segmenter
from optimisers import get_optimisers, get_lr_schedulers

#iamge processing
import cv2
import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
models_dir = os.path.join(project_root, 'models')
sys.path.append(models_dir)

# import network
from network import get_segmenter
from models.mobilenet import mbv2


# Load .npy file
def load_npy_file(file_path):
    data = np.load(file_path, allow_pickle=True)
    print(f"Loaded data type: {type(data)}")

    if isinstance(data, np.ndarray):
        # Try to extract dictionary from numpy array
        try:
            data = data.item()  # Convert numpy array to dictionary
        except AttributeError:
            raise ValueError("The loaded data is not a dictionary.")
    
    if not isinstance(data, dict):
        raise ValueError("The loaded data is not a dictionary.")
    
    return data





#process .npy image
def process_npy_file(input_dir,output_dir,device,model,input_size=512,img_mean=[0.485, 0.456, 0.406],img_std=[0.229, 0.224, 0.225]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
          if file.endswith(".npy"): 
            file_path = os.path.join(root, file)
            print("Processing file: {}".format(file_path))
            #load image
            image = load_npy_file(file_path)
            #process image
            image = process_image(image, input_size,img_mean,img_std)
            image =torch.tensor(image, dtype=torch.float32).unsqueeze(0) #clear batch dimension
            segemented_image = run_inference(model,image,device)
            output_file_path = os.path.join(output_dir,file.replace(".npy",".png"))  
            cv2.imwrite(output_file_path,segemented_image)
            print("Segmentation done and saved as {}".format(output_file_path)) 


def setup_network(args, device):
    logger = logging.getLogger(__name__)
    #create a segmenter model
    segmenter = get_segmenter(
        enc_backbone=args.enc_backbone,
        enc_pretrained=args.enc_pretrained,
        num_classes=args.num_classes,
    ).to(device)
    if device == "cuda":
        segmenter = nn.DataParallel(segmenter)
    logger.info(
        " Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M".format(
            args.enc_backbone,
            args.enc_pretrained,
            dt.misc.compute_params(segmenter) / 1e6,
        )
    )
    #define the loss functions
    training_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_label).to(device)
    validation_loss = dt.engine.MeanIoU(num_classes=args.num_classes)
    return segmenter, training_loss, validation_loss


def setup_checkpoint_and_maybe_restore(args, model, optimisers, schedulers):
    saver = dt.misc.Saver(
        args=vars(args),
        ckpt_dir=args.ckpt_dir,
        best_val=0,
        condition=lambda x, y: x > y,
    )  # keep checkpoint with the best validation score
    (
        epoch_start,
        _,
        model_state_dict,
        optims_state_dict,
        scheds_state_dict,
    ) = saver.maybe_load(
        ckpt_path=args.ckpt_path,
        keys_to_load=["epoch", "best_val", "model", "optimisers", "schedulers"],
    )
    if epoch_start is None:
        epoch_start = 0
    dt.misc.load_state_dict(model, model_state_dict)
    if optims_state_dict is not None:
        for optim, optim_state_dict in zip(optimisers, optims_state_dict):
            optim.load_state_dict(optim_state_dict)
    if scheds_state_dict is not None:
        for sched, sched_state_dict in zip(schedulers, scheds_state_dict):
            sched.load_state_dict(sched_state_dict)
    return saver, epoch_start


def setup_data_loaders(args):
    train_transforms, val_transforms = get_transforms(
        crop_size=args.crop_size,
        shorter_side=args.shorter_side,
        low_scale=args.low_scale,
        high_scale=args.high_scale,
        img_mean=args.img_mean,
        img_std=args.img_std,
        img_scale=args.img_scale,
        ignore_label=args.ignore_label,
        num_stages=args.num_stages,
        augmentations_type=args.augmentations_type,
        dataset_type=args.dataset_type,
    )
    train_sets, val_set = get_datasets(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_list_path=args.train_list_path,
        val_list_path=args.val_list_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        masks_names=("segm",),
        dataset_type=args.dataset_type,
        stage_names=args.stage_names,
        train_download=args.train_download,
        val_download=args.val_download,
    )
    train_loaders, val_loader = dt.data.get_loaders(
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        train_set=train_sets,
        val_set=val_set,
        num_stages=args.num_stages,
    )
    return train_loaders, val_loader


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type=args.enc_optim_type,
        enc_lr=args.enc_lr,
        enc_weight_decay=args.enc_weight_decay,
        enc_momentum=args.enc_momentum,
        dec_optim_type=args.dec_optim_type,
        dec_lr=args.dec_lr,
        dec_weight_decay=args.dec_weight_decay,
        dec_momentum=args.dec_momentum,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=args.enc_lr_gamma,
        dec_lr_gamma=args.dec_lr_gamma,
        enc_scheduler_type=args.enc_scheduler_type,
        dec_scheduler_type=args.dec_scheduler_type,
        epochs_per_stage=args.epochs_per_stage,
    )
    return optimisers, schedulers


def process_image(data, input_size=512, img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225]):
    if isinstance(data, dict):
        image = data.get('rgb')  # Extract the image data from the dictionary
        if image is None:
            raise ValueError("Dictionary does not contain 'rgb' key.")
    elif isinstance(data, np.ndarray):
        image = data
    else:
        raise ValueError("Expected numpy array or dictionary but got something else.")
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Expected numpy array but got something else.")
    
    # Ensure the image has 3 channels
    if image.shape[-1] != 3:
        raise ValueError(f"Expected image with 3 channels but got {image.shape[-1]} channels.")
    
    # Convert image to float32
    image = image.astype(np.float32)
    
    # Resize image
    image = cv2.resize(image, (input_size, input_size))
    
    # Normalize image
    image = (image / 255.0 - img_mean) / img_std
    
    # Transpose image to match the expected input shape [channels, height, width]
    image = np.transpose(image, (2, 0, 1))
    
    return image





def run_inference(model,image,device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        output = output.squeeze(0) #remove batch dimension
        output = output.argmax(1) #get the class index with the highest probability
        output = output.cpu().numpy()  #convert to numpy array
    return output



def main():
    args = get_arguments()
    logger = logging.getLogger(__name__)
    torch.backends.cudnn.deterministic = True
    dt.misc.set_seed(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup the network loading the pre-trained model
    segmenter,_,_= setup_network(args, device=device)
    # setup the checkpoint and restore the model if needed
    saver,_= setup_checkpoint_and_maybe_restore(
        args, model=segmenter, optimisers=[], schedulers=[]
        )
    
    #prepare the data loaders
    input_dir = "/home/yanwenli/light-weight-refinenet/test/data"
    output_dir = "/home/yanwenli/light-weight-refinenet/test/out_put_image"
    #process all .npy files
    process_npy_file(input_dir,output_dir,device,segmenter,device)
    
   


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()
