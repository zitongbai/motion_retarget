import glob
import os
import sys
import numpy as np
import joblib
from tqdm import tqdm
from scipy.spatial.transform import Rotation as sRot
import hydra
from omegaconf import DictConfig

import torch
import torch.multiprocessing as mp
from torch.autograd import Variable

from smpl_sim.utils import torch_utils
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from smpl_sim.utils.smoothing_utils import gaussian_kernel_1d, gaussian_filter_1d_batch

from phc.torch_humanoid_batch import Humanoid_Batch

def load_motion_data(motion_path):
    """Load motion data from a .npz file.

    Args:
        motion_path (str): Path to the .npz file.

    Returns:
        dict: A dictionary containing the motion data, or None if loading failed.
    """
    with open(motion_path, 'rb') as f:
        try:
            entry_data = dict(np.load(f, allow_pickle=True))
        except Exception as e:
            print(f"Error loading {motion_path}: {e}")
            return None
        
    if 'mocap_framerate' not in entry_data:
        return None
    
    # dict_keys(['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses'])
    
    frame_rate = entry_data['mocap_framerate']
    root_trans = entry_data['trans']
    pose_aa = np.concatenate(
        [entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))],
        axis=-1
    )
    betas = entry_data['betas']
    gender = entry_data['gender']
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": frame_rate
    }


def process_motion(motion_names, motion_path_dict, cfg):
    device = torch.device("cpu")

    # load forward kinematics model
    robot_fk = Humanoid_Batch(cfg.robot)
    num_augment_joint = len(cfg.robot.extend_config)    # TODO: understand what is this
    
    # load SMPL parser
    smpl_parser = SMPL_Parser(
        model_path="model/smpl", 
        gender = "neutral"
    )
    smpl_shape, smpl_scale = joblib.load(
        f"data/{cfg.robot.humanoid_type}/smpl_shape.pkl"
    )
    
    # get key joint names and indices for both robot and SMPL
    joint_names_robot = robot_fk.body_names_augment
    key_joint_names_robot = [pair[0] for pair in cfg.robot.joint_matches]
    key_joint_indices_robot = [joint_names_robot.index(name) for name in key_joint_names_robot]
    
    key_joint_names_smpl = [pair[1] for pair in cfg.robot.joint_matches]
    key_joint_indices_smpl = [SMPL_BONE_ORDER_NAMES.index(name) for name in key_joint_names_smpl]

    retarget_data_dict = {}
    pbar = tqdm(motion_names, position=0, leave=True)
    for motion_name in pbar:
        motion_raw_data = load_motion_data(motion_path_dict[motion_name])
        if motion_raw_data is None:
            continue
        
        # sample the motion data to 30 fps
        raw_fps = motion_raw_data['fps']
        desired_fps = 30
        skip = int(raw_fps // desired_fps)
        
        root_pos = motion_raw_data['trans'][::skip]
        pose_aa = motion_raw_data['pose_aa'][::skip]
        
        root_pos = torch.from_numpy(root_pos).float().to(device)
        pose_aa = torch.from_numpy(pose_aa).float().to(device)
        
        num_frames = pose_aa.shape[0]
        
        if num_frames < 10:
            print(f"Skipping {motion_name} due to insufficient frames: {num_frames}")
            continue
        
        with torch.no_grad():
            # Use the loaded pose_aa to compute forward kinematics for SMPL 
            # with the optimzed shape and scale
            verts_smpl, joint_pos_smpl = smpl_parser.get_joints_verts(
                pose_aa, smpl_shape, root_pos
            )
            root_pos_smpl = joint_pos_smpl[:, 0:1]
            joint_pos_smpl = smpl_scale * (joint_pos_smpl - root_pos_smpl) + root_pos_smpl
            joint_pos_smpl[..., 2] -= verts_smpl[0, :, 2].min().item()  # align the ground plane
        
            root_pos_smpl = joint_pos_smpl[:, 0].clone()
            
            root_quat_smpl = torch.from_numpy((sRot.from_rotvec(pose_aa[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat()).float()    # can't directly use this 
            root_rot_smpl = torch.from_numpy(sRot.from_quat(torch_utils.calc_heading_quat(root_quat_smpl)).as_rotvec()).float() # so only use the heading. 

        # prepare the variables for optimization
        dof_pos_var = Variable(
            torch.zeros((1, num_frames, robot_fk.num_dof, 1)), 
            requires_grad=True,
        )
        root_pos_offset_var = Variable(
            torch.zeros(1,3), 
            requires_grad=True,
        )
        root_rot_var = Variable(
            root_rot_smpl.clone(), 
            requires_grad=True,
        )
        # optimizer
        optimizer = torch.optim.Adam(
            [dof_pos_var, root_pos_offset_var, root_rot_var], 
            lr=0.02
        )
        
        filter_kernel_size = 5
        filter_sigma = 0.75
        
        for iteration in range(cfg.get("fitting_iterations", 500)):
            # prepare the angle-axis of each joint for robot
            pose_aa_robot = torch.cat(
                [
                    root_rot_var[None, :, None], 
                    robot_fk.dof_axis * dof_pos_var, 
                    torch.zeros((1, num_frames, num_augment_joint, 3), device=device)
                ], 
                axis=2
            )
            # compute forward kinematics for robot
            fk_return_robot = robot_fk.fk_batch(
                pose_aa_robot, 
                root_pos_smpl[None, ] + root_pos_offset_var
            )
            
            if num_augment_joint > 0:
                key_joint_pos_robot = fk_return_robot.global_translation_extend[:, :, key_joint_indices_robot]
            else:
                key_joint_pos_robot = fk_return_robot.global_translation[:, :, key_joint_indices_robot]
            key_joint_pos_smpl = joint_pos_smpl[:, key_joint_indices_smpl]
            # compute the difference of key joints position between SMPL and robot
            diff = key_joint_pos_robot - key_joint_pos_smpl
            
            # compute the loss: norm of the difference and a regularization term for dof_pos_var
            loss = diff.norm(dim=-1).mean() + 0.01 * 0.01 * torch.mean(torch.square(dof_pos_var))
            
            # update the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # apply dof saturation
            dof_pos_var.data.clamp_(
                min = robot_fk.joints_range[:, 0, None], 
                max = robot_fk.joints_range[:, 1, None]
            )
            
            # filter the dof positions
            # I don't know why the operation is so complex here
            # refer to the original PHC repository for more details
            dof_pos_var.data = gaussian_filter_1d_batch(
                dof_pos_var.squeeze().transpose(1, 0)[None, ],
                filter_kernel_size,
                filter_sigma
            ).transpose(2, 1)[..., None]
            
            pbar.set_description(f"{motion_name[:10]}-Iter: {iteration} \t {loss.item() * 1000:.3f}")
        
        # after optimization
        
        # apply the dof saturation
        dof_pos_var.data.clamp_(
            min=robot_fk.joints_range[:, 0, None], 
            max=robot_fk.joints_range[:, 1, None]
        )
        
        # optimized angle-axis of each joint for robot
        pose_aa_robot_opt = torch.cat(
            [
                root_rot_var[None, :, None], 
                robot_fk.dof_axis * dof_pos_var, 
                torch.zeros((1, num_frames, num_augment_joint, 3), device=device)
            ], 
            axis=2
        )
        
        # optimized root position of robot
        root_pos_opt = (root_pos_smpl + root_pos_offset_var).clone()
        
        # move to the ground plane
        combined_mesh = robot_fk.mesh_fk(
            pose_aa_robot_opt[:, :1].detach(), 
            root_pos_opt[None, :1].detach()
        )
        height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
        root_pos_opt[..., 2] -= height_diff
        
        # also save the smpl joint positions for later use
        joint_pos_smpl_dump = joint_pos_smpl.detach().cpu().numpy().copy()
        joint_pos_smpl_dump[..., 2] -= height_diff
        
        # save the retargeted data
        retarget_data_dict[motion_name] = {
            "root_pos": root_pos_opt.squeeze().detach().cpu().numpy(),
            "root_rot": sRot.from_rotvec(root_rot_var.detach().numpy()).as_quat(),
            "pose_aa": pose_aa_robot_opt.squeeze().detach().cpu().numpy(),
            "dof_pos": dof_pos_var.squeeze().detach().cpu().numpy(),
            "fps": desired_fps, # 30
            "joint_pos_smpl": joint_pos_smpl_dump,
        }
        
    return retarget_data_dict
        
        
@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    
    torch.set_num_threads(1)
    mp.set_sharing_strategy('file_descriptor')
    
    if "dataset_dir" not in cfg:
        raise ValueError("Please specify the dataset_dir in the config.")
    dataset_dir = cfg.dataset_dir
    
    # Get all .npz files in the dataset directory
    
    all_files = glob.glob(f"{dataset_dir}/**/*.npz", recursive=True)
    print(f"Found {len(all_files)} files in {dataset_dir}.")
    # remove the dataset_dir in the path
    motion_path_dict = {}
    for i in range(len(all_files)):
        motion_name = os.path.relpath(all_files[i], dataset_dir)
        # remove the .npz extension
        motion_name = motion_name[:-4]  # remove the .npz extension
        # replace special characters with underscores
        motion_name = motion_name.replace("/", "_").replace(" ", "_").replace("-", "_")
        motion_path_dict[motion_name] = all_files[i]
    motion_names = list(motion_path_dict.keys())

    num_jobs = 30
    chunk = np.ceil(len(motion_names) / num_jobs).astype(int)
    jobs = [motion_names[i:i + chunk] for i in range(0, len(motion_names), chunk)]
    jobs_args = [
        (jobs[i], motion_path_dict, cfg) 
        for i in range(len(jobs))
    ]
    
    if len(jobs_args) == 1:
        retarget_data_dict = process_motion(*jobs_args[0])
    else:
        try:
            pool = mp.Pool(num_jobs)
            retarget_data_dict_list = pool.starmap(process_motion, jobs_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
        retarget_data_dict = {}
        for retarget_data_dict_chunk in retarget_data_dict_list:
            retarget_data_dict.update(retarget_data_dict_chunk)

    # save the retargeted data
    os.makedirs(f"data/{cfg.robot.humanoid_type}/retargeted", exist_ok=True)
    output_file_name = cfg.get("output_file_name", "retargeted_motion")
    output_file_path = f"data/{cfg.robot.humanoid_type}/retargeted/{output_file_name}.pkl"
    joblib.dump(
        retarget_data_dict, 
        output_file_path
    )
    print(
        f"Retargeted motion data saved to {output_file_path}. "
        f"Total {len(retarget_data_dict)} motions processed."
    )


if __name__ == "__main__":
    main()