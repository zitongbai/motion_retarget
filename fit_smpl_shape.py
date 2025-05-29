import os
import sys
import joblib
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

import torch
from torch.autograd import Variable

from scipy.spatial.transform import Rotation as sRot

from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES

from phc.torch_humanoid_batch import Humanoid_Batch

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cpu")
    
    # load forward kinematics model
    robot_fk = Humanoid_Batch(cfg.robot)
    
    # get key joint names and indices for both robot and SMPL
    joint_names_robot = robot_fk.body_names_augment
    key_joint_names_robot = [pair[0] for pair in cfg.robot.joint_matches]
    key_joint_indices_robot = [joint_names_robot.index(name) for name in key_joint_names_robot]
    
    key_joint_names_smpl = [pair[1] for pair in cfg.robot.joint_matches]
    key_joint_indices_smpl = [SMPL_BONE_ORDER_NAMES.index(name) for name in key_joint_names_smpl]
    
    # prepare stand pose of axis-angle for SMPL
    stand_pose_aa_smpl = torch.zeros(
        (1, len(SMPL_BONE_ORDER_NAMES), 3), dtype=torch.float32, device=device
    )
    modifier = cfg.robot.smpl_pose_modifier
    for joint in modifier.keys():
        euler_angle = eval(modifier[joint])
        aa = sRot.from_euler("xyz", euler_angle, degrees=False).as_rotvec()
        stand_pose_aa_smpl[:, SMPL_BONE_ORDER_NAMES.index(joint), :] = torch.tensor(
            aa, dtype=torch.float32, device=device
        ).view(1, 3)
    stand_pose_aa_smpl = stand_pose_aa_smpl.reshape(-1, len(SMPL_BONE_ORDER_NAMES)*3)

    # load SMPL model
    smpl_parser = SMPL_Parser(
        model_path="model/smpl", 
        gender = "neutral"
    )
    
    # compute forward kinematics for SMPL, and get the root translation
    trans = torch.zeros((1,3), dtype=torch.float32, device=device)
    beta = torch.zeros((1, 10), dtype=torch.float32, device=device) # 10 shape parameters
    _, joint_pos = smpl_parser.get_joints_verts(stand_pose_aa_smpl, beta, trans)
    root_trans_offset = joint_pos[:, 0] # I don't know why here, just do it. 

    # prepare stand pose of axis-angle for robot
    # I don't know why there are so many dimensions here, just following the original code
    stand_pose_aa_robot = torch.zeros(
        (1, 1, 1, robot_fk.num_bodies, 3), dtype=torch.float32, device=device
    )

    # compute forward kinematics for robot
    fk_return_robot = robot_fk.fk_batch(stand_pose_aa_robot, root_trans_offset[None, 0:1])
    
    # prepare variables for optimization
    shape_var = Variable(torch.zeros((1, 10), dtype=torch.float32, device=device), requires_grad=True)  # 10 shape parameters
    scale_var = Variable(torch.ones([1], dtype=torch.float32, device=device), requires_grad=True)  # scale factor
    
    # optimizer
    optimizer = torch.optim.Adam([shape_var, scale_var], lr=0.1)
    
    train_iterations = 3000
    print(f"Training for {train_iterations} iterations...")
    pbar = tqdm(range(train_iterations))
    for i in pbar:
        optimizer.zero_grad()
        
        # compute forward kinematics for SMPL with current shape parameters
        _, joint_pos_smpl = smpl_parser.get_joints_verts(stand_pose_aa_smpl, shape_var, trans[0:1])
        root_pos_smpl = joint_pos_smpl[:, 0]
        joint_pos_smpl = scale_var * (joint_pos_smpl - root_pos_smpl) + root_pos_smpl
        
        # compute difference of key joints position between SMPL and robot
        if len(cfg.robot.extend_config) > 0:
            key_joint_pos_robot = fk_return_robot.global_translation_extend[:, :, key_joint_indices_robot]
        else:
            key_joint_pos_robot = fk_return_robot.global_translation[:, :, key_joint_indices_robot]
        key_joint_pos_smpl = joint_pos_smpl[:, key_joint_indices_smpl]
        diff = key_joint_pos_robot - key_joint_pos_smpl
        
        loss = diff.norm(dim = -1).square().sum()
        pbar.set_description_str(f"Iteration {i}: Loss = {loss.item():.4f}")

        loss.backward()
        optimizer.step()
    
    print("Optimization finished.")
    print(f"Final shape parameters: {shape_var.detach().cpu().numpy()}")
    print(f"Final scale factor: {scale_var.detach().cpu().item()}")
    
    os.makedirs(f"data/{cfg.robot.humanoid_type}", exist_ok=True)
    joblib.dump(
        (shape_var.detach(), scale_var.detach()), 
        f"data/{cfg.robot.humanoid_type}/smpl_shape.pkl"
    )
    
    if cfg.get("vis", False):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 24})  # 增大全局字体

        robot_key_joint_pos = fk_return_robot.global_translation_extend[0, :, key_joint_indices_robot, :].detach().cpu().numpy()
        robot_key_joint_pos = robot_key_joint_pos - robot_key_joint_pos[:, 0:1]

        smpl_key_joint_pos = joint_pos_smpl[:, key_joint_indices_smpl].detach().cpu().numpy()
        smpl_key_joint_pos = smpl_key_joint_pos - smpl_key_joint_pos[:, 0:1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(robot_key_joint_pos[0, :, 0],
               robot_key_joint_pos[0, :, 1],
               robot_key_joint_pos[0, :, 2],
               label='Robot Key Joints', color='blue')

        ax.scatter(smpl_key_joint_pos[0, :, 0],
               smpl_key_joint_pos[0, :, 1],
               smpl_key_joint_pos[0, :, 2],
               label='Fitted SMPL Key Joints', color='red')

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax.set_xlabel('X', fontsize=18)
        ax.set_ylabel('Y', fontsize=18)
        ax.set_zlabel('Z', fontsize=18)

        ax.legend(fontsize=16)
        plt.show()


if __name__ == "__main__":
    main()

        