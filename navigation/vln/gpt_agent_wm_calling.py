import os
import sys
import numpy as np
import quaternion
from collections import defaultdict
from GPT.one_stage_prompt_manager import OneStagePromptManager
from .agent_base import BaseAgent
from GPT.api import gpt_infer
import json
import re
from diffusers.utils import export_to_video
import argparse
from scipy.spatial.transform import Rotation as R
from stable_virtual_camera.demo import svc_main, Model
import torch
# from numpy import quaternion
import multiprocessing
from typing import Dict, List, Optional
import cv2
os.environ["PYTORCH_SDP_FORCE_FALLBACK"] = "1"
import math 
import copy
import pickle

from PIL import Image

def resize_to_short_side(img, target_short=512):
    h, w = img.shape[:2]
    if min(h, w) == target_short:          # already the right size
        return img
    scale = target_short / float(min(h, w))
    new_w, new_h = int(math.ceil(w * scale)), int(math.ceil(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

class ActionSpace:
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

def _run_one_candidate(action_list, magnitude,
                        img_path, step_dir, action_folder_name,
                        model_args,
                        model, # one model for each process
                        forward_size, turn_size):
    action_folder = os.path.join(step_dir, action_folder_name)
    os.makedirs(action_folder, exist_ok=True)

    # ---------- generate video ----------
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    video = np.tile(img_rgb[None, ...], (len(action_list) + 1, 1, 1, 1))
    export_to_video(video, os.path.join(action_folder, "video.mp4"), fps=1)

    # ---------- sample trajectory ----------
    pos   = np.zeros(3, dtype=float)
    theta = np.array([np.radians(-20.0), 0.0, 0.0], dtype=float)
    traj, traj_json = _build_trajectory(action_list, pos, theta,
                                        forward_size, turn_size)
    
    c2ws = traj["camera_extrinsic"]
    T = len(c2ws)

    effective_num_targets = min(model_args.num_targets, T - 1)
    effective_num_targets = max(1, effective_num_targets)
    
    
    with open(os.path.join(action_folder, 'episode.pkl'), 'wb') as f:
        pickle.dump(traj, f)
    with open(os.path.join(action_folder, 'episode.json'), 'w') as f:
        json.dump(traj_json, f, indent=2)

    if action_list[0] == ActionSpace.MOVE_FORWARD:
        traj_prior = f"move-forward-{magnitude}"
    elif action_list[0] == ActionSpace.TURN_LEFT:
        traj_prior = f"turn-left-{magnitude}"
    else:
        traj_prior = f"turn-right-{magnitude}"

    svc_main(
        model=model,
        data_path=action_folder,
        task=model_args.task,
        replace_or_include_input=model_args.replace_or_include_input,
        traj_prior=traj_prior,
        cfg=model_args.cfg,
        guider=model_args.guider,
        L_short=model_args.L_short,
        # num_targets=model_args.num_targets,
        num_targets=effective_num_targets,
        
        use_traj_prior=model_args.use_traj_prior,
        output_path=action_folder,
        chunk_strategy=model_args.chunk_strategy,
        c2ws=traj['camera_extrinsic'],
        scene = [img_path] 
    )
    return traj, traj_json


def _build_trajectory(action_list, pos, theta, forward_size, turn_size):
    try:
        trajectory = {
            'camera_pose': [],
            'camera_rotation': [],
            'camera_rotation_euler': [],
            'action': [],
            'camera_extrinsic': [],
        }
        trajectory_json = {
            'camera_pose': [],
            'camera_rotation_euler': [],
            'action': [],
        }

        def _append(a):
            r   = R.from_euler('xyz', theta, degrees=False)
            x, y, z, w = r.as_quat()
            quat = np.quaternion(w, x, y, z)
            trajectory['camera_pose'].append(np.round(pos, 4))
            trajectory['camera_rotation'].append(quat)
            trajectory['camera_rotation_euler'].append(np.round(theta, 4))
            trajectory['action'].append(a)
            trajectory_json['camera_pose'].append(np.round(pos, 4).tolist())
            trajectory_json['camera_rotation_euler'].append(np.round(theta, 4).tolist())
            trajectory_json['action'].append(a)
            rot_mat = r.as_matrix()
            c2w = np.eye(4)
            c2w[:3, :3] = rot_mat
            c2w[:3, 3] = pos
            trajectory['camera_extrinsic'].append(np.round(c2w, 6))

        _append(0)                                # first frame (no-op)
        for action in action_list:                # subsequent actions
            if action == ActionSpace.TURN_LEFT:
                theta[1] -= np.radians(turn_size)
            elif action == ActionSpace.TURN_RIGHT:
                theta[1] += np.radians(turn_size)
            elif action == ActionSpace.MOVE_FORWARD:
                dx = forward_size * np.sin(theta[1])
                dz = forward_size * np.cos(theta[1])
                pos += np.array([dx, 0.0, dz])
            else:
                raise ValueError(f"Unknown action: {action}")
            _append(action)

        return trajectory, trajectory_json

    except Exception as e:
        print(f"❌ Error in _run_one_candidate:")
        print(f"    action_list: {action_list}")
        return None, None
    
class GPTNavAgent(BaseAgent):
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }
    for k, v in env_actions.items():
        env_actions[k] = [[vx] for vx in v]

    def __init__(self, args, env, rank=0):
        super().__init__(env)
        self.args = args
        
        self.model_args = args #get_svc_args()
        
        self.global_model = Model()
        self._build_prompt_manager()

        # Logs
        sys.stdout.flush()
        self.logs = defaultdict(list)
    
    def _build_prompt_manager(self):
        self.prompt_manager = OneStagePromptManager(self.args)
        print('Model version:', self.args.llm)

    def make_equiv_action(self, a_t, obs, traj=None):

        def take_action(i, name):
            if type(name) is int:       # Go to the next viewpoint
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx']) # j+1: idx for navigable location

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append([state.location.viewpointId])


    def parse_json_list_from_str(self, s: str):
        """
        Parse a JSON list from a string that may be wrapped in ```json ... ```.

        Returns:
            list[dict]
        """
        # 去掉 ```json 和 ```
        s = re.sub(r"^```json\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s)

        # 解析 JSON
        return json.loads(s)

    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        batch_size = len(obs)

        # pred_file = os.path.join(self.args.pred_dir, "case_InstrID_%s.json" % obs[0]['instr_id'])
        # if os.path.exists(pred_file):
        #     print('Path already exists, load from ', pred_file)
        #     traj = json.load(open(pred_file))
        #     traj['path'] = [[obs[0]['viewpoint']]]
        #     return [traj]
        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'a_t': {},
        } for ob in obs]

        if traj[0]['instr_id'] in self.results:
            return [None]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

        self.prompt_manager.history = ['' for _ in range(self.args.batch_size)]
        self.prompt_manager.nodes_list = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.node_imgs = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.graph = [{} for _ in range(self.args.batch_size)]
        self.prompt_manager.trajectory = [[] for _ in range(self.args.batch_size)]
        self.prompt_manager.planning = [["Navigation has just started, with no planning yet."] for _ in range(self.args.batch_size)]

        # policy_prompt = self.prompt_manager.make_r2r_policy()
        
        for t in range(self.args.max_action_len):
            if t == self.args.max_action_len:
                break

            cand_inputs = self.prompt_manager.make_action_prompt(obs, previous_angle)
            if self.args.response_format == 'str':
                nav_input = self.prompt_manager.make_r2r_prompts(cand_inputs=cand_inputs, obs=obs, t=t)
            elif self.args.response_format == 'json':
                nav_input = self.prompt_manager.make_r2r_json_prompts(cand_inputs=cand_inputs, obs=obs, t=t)
            else:
                raise NotImplemented

            image_list = self.prompt_manager.node_imgs[0]
            environment_prompts = nav_input["prompts"][0]
            print('-------------------- Environment Prompts --------------------')
            print(environment_prompts)

            if self.args.llm == 'gpt-4-vision-preview' and self.args.response_format == 'str':
                # GPT-4V only supports string mode output
                nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                               self.args.llm, self.args.max_tokens)
                print('-------------------- Output --------------------')
                print(nav_output)
                nav_output = [nav_output]
                a_t = self.prompt_manager.parse_action(nav_output=nav_output,
                                                       only_options_batch=nav_input["only_options"],
                                                       t=t)
                self.prompt_manager.parse_planning(nav_output=nav_output)

            # elif self.args.llm == 'gpt-4o-2024-05-13' and self.args.response_format == 'json':
            elif self.args.llm == 'gpt-4o' and self.args.response_format == 'json':
                if len(image_list) > 20:
                    # GPT-4o currently does not support queries with more than 20 images
                    a_t = [0]
                    print('Exceed image limit and stop!')
                else:
                    
                    # add here adaptive WM calling here
   
                    # import pdb; pdb.set_trace()
                    instruction = obs[0]["instruction"]
                    policy_prompt, ins = self.prompt_manager.make_r2r_policy(instruction) 
                    # import pdb; pdb.set_trace() 
                    try:
                        policy = gpt_infer(policy_prompt, ins, image_list,
                                           self.args.llm, self.args.max_tokens, extra='Generate the JSON plan list now.')
                    
                    except Exception as e:
                        print('Error during LLM inference:', e)
                        a_t = [0]
                        print('LLM inference failed, stop!')
                        break

                    try:
                        policy_json = self.parse_json_list_from_str(policy[0])
                    except Exception as e:
                        print('Error during parsing policy JSON:', e)
                        policy_json = []
                        
                    print('-------------------- Policy JSON --------------------')
                    print(policy_json)
                    
                    # import pdb; pdb.set_trace()
                    
                    if len(policy_json) ==0:
                        print('-------------------- Image List --------------------')
                        print(image_list)
                         
                        try:
                            nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, image_list,
                                                    self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                        except Exception as e:
                            print('Error during LLM inference:', e)
                            a_t = [0]
                            print('LLM inference failed, stop!')
                            break
                        
                        json_output = json.loads(nav_output)
                        a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], t)
                        self.prompt_manager.parse_json_planning(json_output)
                        print('-------------------- Output --------------------')
                        print(nav_output)
                        
                    else:
                        action_mapping = {"F":1, "L":2, "R":3}
                        action_str_mapping = {"F":"move-forward 0.25", "L":"turn-left 9", "R":"turn-right 9"}
                        
                        gen_img_dict = {}
                        for policy in policy_json:
                            
                            try:
                                image_id = policy["img_ids"]
                                
                                try:
                                    image_id = int(image_id)
                                except:
                                    image_id = int(image_id[0])
                                
                                if (int(image_id)+1) > len(image_list):
                                    continue
                                
                                actions = policy["actions"]
                                uid = obs[0]['instr_id']
                                action_ids = [action_mapping[a] for a in actions]
                                action_seq_strs = [action_str_mapping[a] for a in actions]

                                save_dir = os.path.join(self.args.output_dir, uid, f"step_{t}", f"{image_id}")
                                os.makedirs(os.path.join(save_dir), exist_ok=True)
                                # --- primary image ----------------------------------------------------------
                                primary_img_path = os.path.join(save_dir, "img_0.png")
                                img = cv2.imread(image_list[int(image_id)])
                                img = resize_to_short_side(img, target_short=512)
                                cv2.imwrite(primary_img_path, img)
                                saved, ordered, folder_name = self._simulate_one_sequence(
                                    image_path=primary_img_path,
                                    step_idx=0,
                                    action_ids=action_ids,
                                    action_seq_strs=action_seq_strs,
                                    model_args=self.model_args,
                                    save_dir=save_dir,
                                    sampling_interval_angle=9,
                                    sampling_interval_meter=0.25,
                                )
                                
                                border_w = 10
                                gen_img_paths = [primary_img_path] + [paths[0] for _, paths in ordered]
                                imgs = [Image.open(p).convert("RGB") for p in gen_img_paths]
                                
                                widths, heights = zip(*(img.size for img in imgs))
                                total_width = sum(widths) + border_w * (len(imgs) - 1)
                                max_height = max(heights)
                                
                                canvas = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
                                x = 0
                                for i, img in enumerate(imgs):
                                    canvas.paste(img, (x, 0))
                                    x += img.size[0]
                                    if i < len(imgs) - 1:
                                        x += border_w  # 

                                out_path =  os.path.join(self.args.output_dir, uid, f"step_{t}", f"{image_id}", "combined.png")
                                canvas.save(out_path)
                                gen_img_dict[image_id] = out_path
                                
                            except Exception as e:
                                print('Error during world model simulation:', e)
                                continue
                            # import pdb; pdb.set_trace()
                            
                        new_image_list = image_list.copy()
                        for img_id, gen_img_path in gen_img_dict.items():
                            new_image_list[int(img_id)] = gen_img_path
                        
                        print('-------------------- New Image List --------------------')
                        print(new_image_list)
                        try:
                            nav_output, tokens = gpt_infer(nav_input["task_description"], environment_prompts, new_image_list,
                                                    self.args.llm, self.args.max_tokens, response_format={"type": "json_object"})
                        except Exception as e:
                            print('Error during LLM inference:', e)
                            a_t = [0]
                            print('LLM inference failed, stop!')
                            break
                        
                        json_output = json.loads(nav_output)
                        a_t = self.prompt_manager.parse_json_action(json_output, nav_input["only_options"], t)
                        self.prompt_manager.parse_json_planning(json_output)
                        print('-------------------- Output --------------------')
                        print(nav_output)

            else:
                raise NotImplemented

            for i in range(batch_size):
                traj[i]['a_t'][t] = a_t[i]

            # Determine stop actions
            a_t_stop = [a_t_i == 0 for a_t_i in a_t]

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if a_t_stop[i] or ended[i]:
                    cpu_a_t.append(-1)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(a_t[i] - 1)

            self.make_equiv_action(cpu_a_t, obs, traj)
            obs = self.env._get_obs()

            previous_angle = [{'heading': ob['heading'],
                               'elevation': ob['elevation']} for ob in obs]

            # we only implement batch_size=1
            if a_t[0] == 0:
                break

            self.prompt_manager.make_history(a_t, nav_input, t)

        return traj
    
    def _simulate_one_sequence(
        self,
        image_path: str,
        step_idx: int,
        action_ids: List[int],
        action_seq_strs: List[str],
        model_args,
        save_dir: str,
        sampling_interval_angle: int,
        sampling_interval_meter: float,
    ):
        """
        Uniformly sample ONE frame per action according to action length.
        No keyframes, no diff, no ffmpeg.

        Idea:
        - pred.mp4 may be padded / repeated / 30fps
        - We ignore that.
        - We treat pred.mp4 as a continuous timeline.
        - Each action occupies a proportion of the total action length.
        - We sample the CENTER frame of each action's proportion.
        """
        import os
        import cv2
        import numpy as np
        from typing import Dict, List, Tuple

        # -------------------------------------------------
        # 0) render pred.mp4 (unchanged)
        # -------------------------------------------------
        step_dir = save_dir #os.path.join(save_dir, f"step_{step_idx}")
        # os.makedirs(step_dir, exist_ok=True)

        folder_name = "_".join(s.replace(" ", "_") for s in action_seq_strs)
        folder_name = folder_name[:180]

        turn_size = model_args.frame_interval * 3
        forward_size = model_args.frame_interval * (0.25 / 3)

        self._take_action_with_world_model(
            step_idx=step_idx,
            img_path=image_path,
            action_lists=[action_ids],
            magnitudes=[0.0],  # placeholder
            step_dir=step_dir,
            action_folder_name_list=[folder_name],
            forward_size=forward_size,
            turn_size=turn_size,
            num_workers=1,
        )

        video_path = os.path.join(step_dir, folder_name, "pred.mp4")
        if not os.path.exists(video_path):
            return {}, [], folder_name

        out_dir = os.path.join(step_dir, folder_name)
        os.makedirs(out_dir, exist_ok=True)

        # -------------------------------------------------
        # 1) helpers
        # -------------------------------------------------
        def _parse_action(action_str: str):
            verb, mag = action_str.split()
            return verb, float(mag)

        def _is_forward(verb: str) -> bool:
            return verb == "move-forward"

        # -------------------------------------------------
        # 2) compute action lengths (base grid length)
        # -------------------------------------------------
        action_infos = []  # (action_key, action_len)

        cum_turn_left = 0.0
        cum_turn_right = 0.0
        cum_forward = 0.0

        for a_str in action_seq_strs:
            verb, magnitude = _parse_action(a_str)

            if verb == "turn-left":
                cum_turn_left += magnitude
                action_key = f"turn-left {cum_turn_left:.1f}"
                arr = np.arange(0, magnitude + 1, turn_size)
            elif verb == "turn-right":
                cum_turn_right += magnitude
                action_key = f"turn-right {cum_turn_right:.1f}"
                arr = np.arange(0, magnitude + 1, turn_size)
            else:
                cum_forward += magnitude
                action_key = f"move-forward {cum_forward:.2f}"
                arr = np.arange(0, magnitude + 1e-3, forward_size)

            # length in "steps"
            action_len = max(len(arr) - 1, 1)
            action_infos.append((action_key, action_len))

        total_action_len = sum(l for _, l in action_infos)

        # -------------------------------------------------
        # 3) open video
        # -------------------------------------------------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}, [], folder_name

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return {}, [], folder_name

        # -------------------------------------------------
        # 4) uniform sampling by action length
        # -------------------------------------------------
        saved: Dict[str, List[str]] = {}
        ordered: List[Tuple[str, List[str]]] = []

        cum_len = 0.0

        for action_key, action_len in action_infos:
            # center of this action in [0, 1]
            center_ratio = (cum_len + 0.5 * action_len) / total_action_len

            # map to frame index
            frame_idx = int(round(center_ratio * (total_frames - 1)))
            frame_idx = max(0, min(frame_idx, total_frames - 1))

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                cum_len += action_len
                continue

            safe_prefix = action_key.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(out_dir, f"{safe_prefix}.png")
            cv2.imwrite(out_path, frame)

            saved[action_key] = [out_path]
            ordered.append((action_key, saved[action_key]))

            cum_len += action_len

        cap.release()
        return saved, ordered, folder_name
    
    
    def _take_action_with_world_model(self, step_idx, img_path, action_lists, magnitudes, step_dir, action_folder_name_list, forward_size=0.25, turn_size=9, num_workers=1):
        """
        parallel inference
        `num_workers`
        """
        tasks = [(alist, mag, img_path, step_dir, action_folder_name,
                copy.deepcopy(self.model_args),
                self.global_model, 
                forward_size, turn_size)
                for alist, mag, action_folder_name in zip(action_lists, magnitudes, action_folder_name_list)]

        if num_workers is None:
            num_workers = min(len(tasks), os.cpu_count() or 1)

        with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
            results = pool.starmap(_run_one_candidate, tasks)