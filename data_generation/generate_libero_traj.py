import os
import h5py
import pickle as pkl
import numpy as np
from pathlib import Path
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from einops import rearrange
from flow_utils import sample_from_mask, sample_double_grid

DATASET_PATH = Path("/your_data")
BENCHMARKS = ["libero_object"]
SAVE_DATA_PATH = Path("/your_path/libero/libero_traj")
img_size = (128, 128)

# create save directory
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# benchmark for suite
benchmark_dict = benchmark.get_benchmark_dict()
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("all-MiniLM-L6-v2")
lang_model = AutoModel.from_pretrained("all-MiniLM-L6-v2")

cotracker = torch.hub.load(os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_co-tracker_main/"), "cotracker2", source="local")
cotracker = cotracker.eval().cuda()

def track_video_iterative(video, init_grid_size, clip_length=16):
    T, C, H, W = video.shape
    video = torch.from_numpy(video).cuda().float()

    init_grid_points = sample_double_grid(init_grid_size, device="cuda").unsqueeze(0)
    init_grid_points[:, :, 0] *= W
    init_grid_points[:, :, 1] *= H
    init_grid_points = init_grid_points.int()
    query_frame = torch.zeros(1, init_grid_points.shape[1], 1).cuda()
    init_grid_points = torch.cat([query_frame, init_grid_points], dim=-1)   # [1, N, 3]
    
    pred_tracks = torch.zeros(T, clip_length, (init_grid_size**2)*2, 2)
    pred_visibility = torch.zeros(T, clip_length, (init_grid_size**2)*2)

    for start_frame_idx in range(T):
        end_idx = start_frame_idx + clip_length
        if end_idx > T:
            remain = end_idx - T
            video_clip = video[start_frame_idx:]
            last_frame = video[-1].unsqueeze(0).repeat(remain, 1, 1, 1)
            video_clip = torch.cat([video_clip, last_frame], dim=0)
        else:
            video_clip = video[start_frame_idx:end_idx]

        clip_tracks, clip_vis = cotracker(video_clip[None], queries=init_grid_points, backward_tracking=True) # [1, T, N, 2]
        pred_tracks[start_frame_idx, :, :, :] = clip_tracks
        pred_visibility[start_frame_idx, :, :] = clip_vis
        last_valid_clip = (clip_tracks, clip_vis)
    
    return pred_tracks, pred_visibility


# Total number of tasks
num_tasks = 0
for benchmark in BENCHMARKS:
    benchmark_path = DATASET_PATH / benchmark
    num_tasks += len(list(benchmark_path.glob("*.hdf5")))

tasks_stored = 0
for benchmark in BENCHMARKS:
    print(f"############################# {benchmark} #############################")
    benchmark_instance = benchmark_dict[BENCHMARKS[0]]()
    num_tasks = benchmark_instance.get_num_tasks()
    print(f"############################# {num_tasks} #############################")

    datasets_default_path = '/your_path/libero'
    all_demo_files = [os.path.join(datasets_default_path, benchmark_instance.get_task_demonstration(i)) for i in range(num_tasks)]
    save_benchmark_path = SAVE_DATA_PATH / benchmark
    save_benchmark_path.mkdir(parents=True, exist_ok=True)

    # Init env benchmark suite
    task_suite = benchmark_dict[benchmark]()

    for task_file in benchmark_path.glob("*.hdf5"):
        print(f"Processing {tasks_stored+1}/{num_tasks}: {task_file}")
        data = h5py.File(task_file, "r")["data"]

        # Init env
        task_name = str(task_file).split("/")[-1][:-10]
        # get task id from list of task names
        task_id = task_suite.get_task_names().index(task_name)
        # create environment
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": img_size[0],
            "camera_widths": img_size[1],
        }
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        render_image = obs["agentview_image"][::-1, :]

        observations = []
        states = []
        actions = []
        rewards = []

        # for demo in list(data.keys())[:10]:
        for demo in list(data.keys()):
            print(f"Processing {demo}")
            demo_data = data[demo]

            observation = {}
            observation["robot_states"] = np.array(
                demo_data["robot_states"], dtype=np.float32
            )

            # render image offscreen
            pixels, depths = [], []
            pixels_ego, depths_ego = [], []
            joint_states, eef_states, gripper_states = [], [], []
            for i in range(len(demo_data["states"])):
                obs = env.regenerate_obs_from_state(demo_data["states"][i])
                img = obs["agentview_image"][::-1]
                img_ego = obs["robot0_eye_in_hand_image"][::-1]

                joint_state = obs["robot0_joint_pos"]
                eef_state = np.concatenate(
                    [obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
                )
                gripper_state = obs["robot0_gripper_qpos"]
                # append
                pixels.append(img)
                pixels_ego.append(img_ego)
                joint_states.append(joint_state)
                eef_states.append(eef_state)
                gripper_states.append(gripper_state)
            
            observation["pixels"] = np.array(pixels, dtype=np.uint8)
            observation["pixels_egocentric"] = np.array(pixels_ego, dtype=np.uint8)
            observation["joint_states"] = np.array(joint_states, dtype=np.float32)
            observation["eef_states"] = np.array(eef_states, dtype=np.float32)
            observation["gripper_states"] = np.array(gripper_states, dtype=np.float32)
            rgb = observation["pixels"]
            rgb = rearrange(rgb, "t h w c -> t c h w")
            T, C, H, W = rgb.shape
            pred_tracks, pred_vis = track_video_iterative(rgb, 8, clip_length=16)
            observation["tracks"] = pred_tracks.cpu().numpy() 
            observation["vis"] = pred_vis.cpu().numpy() 

            rgbego = observation["pixels_egocentric"]
            rgbego = rearrange(rgbego, "t h w c -> t c h w")
            T, C, H, W = rgbego.shape
            pred_egotracks, pred_egovis = track_video_iterative(rgbego, 8, clip_length=16)

            observation["egotracks"] = pred_egotracks.cpu().numpy() 
            observation["egovis"] = pred_egovis.cpu().numpy() 
            observations.append(observation)
            states.append(np.array(demo_data["states"], dtype=np.float32))
            actions.append(np.array(demo_data["actions"], dtype=np.float32))
            rewards.append(np.array(demo_data["rewards"], dtype=np.float32))

        inputs = tokenizer(env.language_instruction, return_tensors='pt', padding=True, truncation=True)
        outputs = lang_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        task_emb = last_hidden_state[:, 0, :]  
        # save data
        save_data_path = save_benchmark_path / (
            str(task_file).split("/")[-1][:-10] + ".pkl"
        )
        with open(save_data_path, "wb") as f:
            pkl.dump(
                {
                    "observations": observations,
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "task_emb": task_emb,
                },
                f,
            )
        print(f"Saved to {str(save_data_path)}")

        tasks_stored += 1
