import os
import sys
import time
import numpy as np
import joblib
import hydra
from omegaconf import DictConfig

import torch

import mujoco
import mujoco.viewer


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])


def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Pause/Resume")
        paused = not paused
    elif chr(keycode) == "[":
        print("next")
        motion_id += 1
        if motion_id >= num_motions:
            motion_id = 0
        curr_motion_key = motion_data_keys[motion_id]
        print("Current motion:", curr_motion_key)
    elif chr(keycode) == "]":
        print("previous")
        motion_id -= 1
        if motion_id < 0:
            motion_id = num_motions - 1
        curr_motion_key = motion_data_keys[motion_id]
        print("Current motion:", curr_motion_key)
    else:
        print("not mapped", chr(keycode))
    
    
@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg : DictConfig) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    device = torch.device("cpu")
    humanoid_xml = cfg.robot.asset.assetFileName
    
    curr_start = 0
    num_motions = 1
    motion_id = 0
    motion_acc = set()
    time_step = 0
    dt = 1/30
    paused = False
    
    motion_file_name = cfg.get("motion_file_name", "retargeted_motion")
    motion_file = f"data/{cfg.robot.humanoid_type}/retargeted/{motion_file_name}.pkl"
    print(motion_file)
    motion_dict = joblib.load(motion_file)
    motion_data = motion_dict["retarget_data"]
    motion_data_keys = list(motion_data.keys())
    
    num_motions = len(motion_data_keys)
    print(f"Number of motions: {num_motions}")
    
    joint_names_robot = motion_dict["joint_names_robot"]
    joint_names_smpl = motion_dict["joint_names_smpl"]

    print("Joint names (robot):", joint_names_robot)
    print("Joint names (SMPL):", joint_names_smpl)
    
    num_joints_robot = len(joint_names_robot)
    num_joints_smpl = len(joint_names_smpl)

    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
    
    # debug_joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
    # debug_joint_names_qpos = {
    #     jnt_name: mj_model.joint(jnt_name).qposadr for jnt_name in debug_joint_names
    # }
    # for key in debug_joint_names_qpos:
    #     print(f"Joint {key} has qpos address {debug_joint_names_qpos[key]}")

    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(num_joints_smpl):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.03, np.array([1, 0, 0, 1]))
        for _ in range(num_joints_robot):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.03, np.array([0, 1, 0, 1]))

        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            curr_time = int(time_step/dt) % curr_motion['dof_pos'].shape[0]
            
            mj_data.qpos[:3] = curr_motion['root_pos'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            mj_data.qpos[7:] = curr_motion['dof_pos'][curr_time]
                
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt

            # joint_gt = motion_data[curr_motion_key]['joint_pos_smpl']
            
            # for i in range(joint_gt.shape[1]):
            #     viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]

            joint_pos_smpl = curr_motion['joint_pos_smpl']
            joint_pos_robot = curr_motion['joint_pos_robot']
            for i in range(num_joints_smpl):
                viewer.user_scn.geoms[i].pos = joint_pos_smpl[curr_time, i]
            for i in range(num_joints_robot):
                viewer.user_scn.geoms[i + num_joints_smpl].pos = joint_pos_robot[curr_time, i]

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()






