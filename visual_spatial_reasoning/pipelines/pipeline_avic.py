from utils.api import ChatAPI, AzureConfig
from utils.prompt_formatting import *
from tqdm import tqdm
import argparse
import json
import random
import os
import cv2
import sys
from typing import Dict, List, Optional
from stable_virtual_camera.demo import svc_main, Model
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import pickle
import copy
os.environ["PYTORCH_SDP_FORCE_FALLBACK"] = "1"
from diffusers.utils import export_to_video
import quaternion
from pipeline_baseline import PipelineBase
import torch
from numpy import quaternion
import multiprocessing
import re
from collections import Counter
from utils.qwen_policy import Qwen3VLPolicy

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


def ordered_to_action_consequences(
    ordered,
    sampling_interval_angle: int,
    sampling_interval_meter: float,
):
    """
    ordered: List[(action_key, [img_paths])]

    return:
      {
        "Turn Right": {"9 degrees": path, "18 degrees": path, ...},
        "Turn Left":  {...},
        "Move Forward": {"0.25 meters": path, ...},
      }
    """
    action_consequences = {}

    for action_key, paths in ordered:
        is_forward = action_key.startswith("move-forward")
        interval = sampling_interval_meter if is_forward else sampling_interval_angle
        unit = "meters" if is_forward else "degrees"

        # group name
        if "turn-right" in action_key:
            group = "Turn Right"
        elif "turn-left" in action_key:
            group = "Turn Left"
        else:
            group = "Move Forward"

        # parse cumulative magnitude from action_key, e.g. "turn-right 18.0" -> 18.0
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*$", action_key)
        
        # base_val = float(m.group(1)) if m else 0.0  # fallback
        if is_forward:
            base_val = 0.25
        else:
            base_val = 9 

        # ensure we merge into existing group dict (DON'T overwrite)
        sub = action_consequences.setdefault(group, {})

        for i, p in enumerate(paths):
            # if there are multiple frames for the same action_key, keep increasing label
            val = base_val + i * interval

            if unit == "meters":
                label = f"{val:.2f} meters"
            else:
                # if you want integer degrees:
                label = f"{int(round(val))} degrees"

            # avoid overwrite if label already exists
            if label in sub:
                suffix = 2
                new_label = f"{label} ({suffix})"
                while new_label in sub:
                    suffix += 1
                    new_label = f"{label} ({suffix})"
                label = new_label

            sub[label] = p

    return action_consequences
    


class SpatialVQAPipelineSVC(PipelineBase):
    """Class-based refactor of the original *source* script, keeping identical
    functionality but adopting the object-oriented structure used elsewhere
    in the code-base (see PipelineBase / PipelineSiyuan).
    """
    # ---------------------------------------------------------------------
    #  INIT
    # ---------------------------------------------------------------------
    def __init__(
        self,
   ):
        super().__init__()
        model_args = self.model_args
        self.global_model = Model()
         
        # new knobs
        self.model_args.num_policy_samples = getattr(self.model_args, "num_policy_samples", 5)
        self.model_args.policy_majority_threshold = getattr(self.model_args, "policy_majority_threshold", 0.5)  # >0.5 means strict majority
        self.model_args.max_wm_candidates = getattr(self.model_args, "max_wm_candidates", 5)  # cap WM runs even if more plans sampled

        # ---- Optional Qwen-VL policy model (Qwen3-VL or Qwen2.5-VL) ----
        # When --policy_model_type is qwen3vl / qwen2.5vl, the gating + action
        # planning ("policy") step is produced by a local (optionally
        # LoRA/GRPO-trained) Qwen-VL model instead of the prompted GPT VLM.
        # Default "gpt" keeps the training-free behaviour unchanged.
        self.qwen_policy = None
        if getattr(model_args, "policy_model_type", "gpt") in ("qwen3vl", "qwen2.5vl"):
            lora_ckpt = getattr(model_args, "policy_lora_ckpt", None) or None
            print(f"[Policy] Loading {model_args.policy_model_type} policy: "
                  f"{model_args.policy_model_name}"
                  + (f"  + LoRA: {lora_ckpt}" if lora_ckpt else ""))
            self.qwen_policy = Qwen3VLPolicy(
                model_name=model_args.policy_model_name,
                interval_meter=model_args.sampling_interval_meter,
                interval_angle=model_args.sampling_interval_angle,
                max_atomic_actions=model_args.max_action_ids_cap,
                lora_ckpt=lora_ckpt,
            )

    # ------------------------------------------------------------------
    #  PUBLIC ENTRY-POINT
    # ------------------------------------------------------------------
    def _sample_policies(self, question, primary_img_path, helper_img_path, n: int):
        """Return list of (policy_dict, policy_text). Only keep parsed policies.

        Dispatches between the GPT (self.vlm) policy and the Qwen-VL policy
        based on --policy_model_type.
        """
        if self.qwen_policy is not None:
            return self._sample_policies_qwen(
                question, primary_img_path, helper_img_path, n
            )

        # GPT policy (training-free path)
        policies = []
        for _ in range(n):
            sys_prompt, content = self.vlm.format_prompt(
                prompt_type="policy_plan",
                question=question["question"],
                answer_choices=question["answer_choices"],
                images=[primary_img_path, helper_img_path] if helper_img_path else [primary_img_path],
            )
            policy_text = self.vlm.run_prompt("policy_plan", sys_prompt, content)
            pol = self._parse_policy_json(policy_text)
            print('[Policy Planning]:', pol)
            if pol:
                policies.append((pol, policy_text))
        return policies

    def _sample_policies_qwen(self, question, primary_img_path, helper_img_path, n: int):
        """Sample policies from the local Qwen-VL model.

        Returns the same shape as the GPT path: list of (policy_dict, policy_text)
        where policy_dict is {"decision": ..., "actions": <atomic_actions>, "reason": ...}.
        """
        images = [primary_img_path, helper_img_path] if helper_img_path else [primary_img_path]

        # n samples means we need diverse outputs -> need temperature > 0
        temperature = self.model_args.policy_temperature
        if n > 1 and temperature <= 0:
            print(
                f"[Policy] policy_temperature={temperature} but num_policy_samples={n}; "
                "outputs will be near-identical. Set --policy_temperature > 0 for diversity."
            )

        policies = []
        for i in range(n):
            result = self.qwen_policy.plan(
                question=question["question"],
                answer_choices=question["answer_choices"],
                images=images,
                max_new_tokens=self.model_args.policy_max_new_tokens,
                temperature=temperature,
                top_p=self.model_args.policy_top_p,
                max_retries=self.model_args.max_tries_gpt,
            )
            print(f"[Policy Planning {i}]: decision={result['decision']} "
                  f"reason={result['reason'][:80]} parse_ok={result['parse_ok']}")

            if not result["parse_ok"]:
                continue

            # Translate to the dict shape the rest of the pipeline expects.
            pol = {
                "decision": result["decision"],
                "actions": result["atomic_actions"],
                "reason": result["reason"],
            }
            policies.append((pol, result["raw_response"]))
        return policies

    def _majority_vote_decision(self, policies):
        """policies: list of (policy_dict, policy_text). Return decision + counts."""
        if not policies:
            return "skip", {"skip": 0, "call_wm": 0}
        votes = [p[0].get("decision", "skip") for p in policies]
        c = Counter(votes)
        # normalize decision keys
        skip_n = c.get("skip", 0)
        call_n = c.get("call_wm", 0)
        decision = "call_wm" if call_n > skip_n else "skip"
        return decision, {"skip": skip_n, "call_wm": call_n}
    
    
    def _score_one_candidate_ordered(self, question, primary_img_path, helper_img_path, ordered):
        """
        ordered: [(action_key, [img_paths...]), ...] from _simulate_one_sequence
        Returns a scalar score for selecting best trajectory.
        """
        scores = None
        for _ in range(self.model_args.max_tries_gpt):
            # import pdb; pdb.set_trace()
            sys_prompt, content = self.vlm.format_prompt(
                prompt_type="prompt_scores_policy_plan",
                question=question["question"],
                answer_choices=question["answer_choices"],
                images=[primary_img_path, helper_img_path] if helper_img_path else [primary_img_path],
                action_consequences=ordered,
            )
             
            # import pdb; pdb.set_trace()
        
            resp = self.vlm.run_prompt("prompt_scores_policy_plan", sys_prompt, content)
            
            # import pdb; pdb.set_trace()
            
            score = self._process_score(resp)

            if score != "out of control":
                break
            
        # import pdb; pdb.set_trace()

        if score == "out of control" or score is None:
            return None, "score_parse_fail"


        return score, "ok"

    
    
    def run(self) -> None:
        """Main evaluation loop (mirrors the logic of the original script)."""
        for question in self.questions:
            # ------------------------------------------------------------------
            #  Quick filters & deduplication
            # ------------------------------------------------------------------
            if question["question_type"] in ["other"]:
                print(f"[SpatialVQA] Skipping question {question['database_idx']} - not a spatial VQA task.")
                self.results["skip_indices"].append(question["database_idx"])
                self.save_results()
                continue

            if self.model_args.question_type != "None" and question["question_type"] != self.model_args.question_type:
                print(f"[SpatialVQA] Skipping question {question['database_idx']} - not a {self.model_args.question_type} task.")
                self.results["skip_indices"].append(question["database_idx"])
                self.save_results()
                continue
                
            if len(question["img_paths"]) > self.model_args.max_images:
                print(f"[SpatialVQA] Skipping question {question['database_idx']} - only one image supported.")
                self.results["skip_indices"].append(question["database_idx"])
                self.save_results()
                continue

            qid = question["database_idx"]
            if (
                qid in self.results["skip_indices"]
                or any(qid in result["correct"] for result in self.results["progress"].values())
                or any(qid in result["wrong"] for result in self.results["progress"].values())
            ):
                print(f"[SpatialVQA] Skipping already processed question {qid}.")
                continue

            os.makedirs(os.path.join(self.model_args.output_dir, f"{qid}"), exist_ok=True)

            # ------------------------------------------------------------------
            #  Set-up per-question output folder & initial image(s)
            # ------------------------------------------------------------------
            save_dir = os.path.join(self.model_args.output_dir, f"{qid}")

            os.makedirs(os.path.join(save_dir, f"step_0"), exist_ok=True)

            # --- primary image ----------------------------------------------------------
            primary_img_path = os.path.join(save_dir, "step_0", "img_0.png")
            img = cv2.imread(question["img_paths"][0])
            if self.model_args.vlm_model_name == "OpenGVLab/InternVL3-14B":
                # resize to 512x512
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            else:
                img = resize_to_short_side(img, target_short=512)
            cv2.imwrite(primary_img_path, img)

            # --- optional helper image --------------------------------------------------
            helper_img_path = None
            if len(question["img_paths"]) > 1:
                helper_img_path = os.path.join(save_dir, "step_0", "helper_img.png")
                helper = cv2.imread(question["img_paths"][1])
                if self.model_args.vlm_model_name == "OpenGVLab/InternVL3-14B":
                    helper = cv2.resize(helper, (512, 512), interpolation=cv2.INTER_LINEAR)
                else:
                    helper = resize_to_short_side(helper, target_short=512)
                cv2.imwrite(helper_img_path, helper)

            # ------------------------------------------------------------------
            #  Dialogue loop (LLM <-> environment)
            # ------------------------------------------------------------------
            response, result, action_list, magnitude = None, "out of control", [], None
            # ----------------------------------------------------------
            #  Query Policy Model for Action Planning
            # ----------------------------------------------------------
            policy = None
            policies = self._sample_policies(
                question=question,
                primary_img_path=primary_img_path,
                helper_img_path=helper_img_path,
                n=self.model_args.num_policy_samples
            )

            if not policies:
                # fallback
                final_decision = "skip"
                vote_stats = {"skip": 1, "call_wm": 0}
            else:
                final_decision, vote_stats = self._majority_vote_decision(policies)
            
            print("[POLICY VOTES]", vote_stats, "=>", final_decision)
             
            # import pdb; pdb.set_trace()
            
            call_plans = []
            if final_decision == "call_wm":
                for pol, pol_text in policies:
                    if pol.get("decision") == "call_wm" and pol.get("actions"):
                        call_plans.append(pol)
                    seen = set()
                uniq_plans = []
                for pol in call_plans:
                    key = tuple((a.get("type"), float(a.get("value", 0))) for a in pol["actions"])
                    if key not in seen:
                        seen.add(key)
                        uniq_plans.append(pol)

                # Cap #WM candidates
                call_plans = uniq_plans[: self.model_args.max_wm_candidates]

            else:
                call_plans = []
                
                
            policy = {
                "decision": final_decision,
                "vote_stats": vote_stats,
                "num_samples": len(policies),
                "num_call_candidates": len(call_plans),
            }
            
            wm_success = False
            best_ordered = None
            best_plan = None
            best_folder = None
            best_score = -1e9
            best_score_meta = None

            if policy["decision"] == "call_wm" and len(call_plans) > 1:
                for cand_i, cand_pol in enumerate(call_plans):
                # try:
                    action_ids, action_seq_strs = self._plan_to_action_ids(cand_pol["actions"])
                    # (optional) cap length
                    action_ids = action_ids[: getattr(self.model_args, "max_action_ids_cap", len(action_ids))]
                    action_seq_strs = action_seq_strs[: getattr(self.model_args, "max_action_ids_cap", len(action_seq_strs))]

                    saved, ordered, folder_name = self._simulate_one_sequence(
                        image_path=primary_img_path,
                        step_idx=0,
                        action_ids=action_ids,
                        action_seq_strs=action_seq_strs,
                        model_args=self.model_args,
                        save_dir=save_dir,
                        sampling_interval_angle=self.model_args.sampling_interval_angle,
                        sampling_interval_meter=self.model_args.sampling_interval_meter,
                    )
                                        
                    new_ordered = [(action_seq_strs[i], ordered[i][1]) for i in range(len(ordered))]

                    # Score this candidate’s generated views
                    cand_score, status = self._score_one_candidate_ordered(
                        question=question,
                        primary_img_path=primary_img_path,
                        helper_img_path=helper_img_path,
                        ordered=new_ordered,
                    )
                    

                    print(f"[CAND {cand_i}] score={cand_score} status={status} plan={action_seq_strs}")

                    if cand_score is not None and cand_score > best_score:
                        best_score = cand_score
                        best_ordered = ordered
                        best_plan = cand_pol
                        best_folder = folder_name
                        best_score_meta = {"status": status, "score": cand_score, "plan": action_seq_strs}
                        best_action_seq_strs = action_seq_strs

                    
            elif policy["decision"] == "call_wm" and len(call_plans) == 1:
                action_ids, action_seq_strs = self._plan_to_action_ids(call_plans[0]["actions"])
                # (optional) cap length
                action_ids = action_ids[: getattr(self.model_args, "max_action_ids_cap", len(action_ids))]
                action_seq_strs = action_seq_strs[: getattr(self.model_args, "max_action_ids_cap", len(action_seq_strs))]

                # import pdb; pdb.set_trace()

                saved, ordered, folder_name = self._simulate_one_sequence(
                    image_path=primary_img_path,
                    step_idx=0,
                    action_ids=action_ids,
                    action_seq_strs=action_seq_strs,
                    model_args=self.model_args,
                    save_dir=save_dir,
                    sampling_interval_angle=self.model_args.sampling_interval_angle,
                    sampling_interval_meter=self.model_args.sampling_interval_meter,
                ) 
                best_score = 0
                best_ordered = ordered
                best_plan = call_plans[0]
                best_folder = folder_name
                best_score_meta = {"status": 'ok', "score": 0., "plan": action_seq_strs}
                best_action_seq_strs = action_seq_strs 
                # import pdb; pdb.set_trace()
                
            policy["best_plan"] = best_plan
            policy["best_score"] = best_score_meta
            policy["best_folder"] = best_folder
            print(policy)
            # --- if skip => final QA directly ---
            all_images = [primary_img_path] + ([helper_img_path] if helper_img_path else [])
            generated_views = []
            
            if best_ordered is None:
                policy["decision"] = "skip"
            
            if policy['decision'] != 'skip':

                action_consequences = [(best_action_seq_strs[i], best_ordered[i][1]) for i in range(len(best_ordered))]                
                sys_prompt, content = self.vlm.format_prompt(
                    prompt_type="answer_scaling",
                    question=question["question"],
                    answer_choices=question["answer_choices"],
                    images=[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path],
                    action_consequences=action_consequences,
                )
                response = self.vlm.run_prompt("answer_scaling_no_cot", sys_prompt, content)
                print("[LLM RESPONSE]", response)
                result = self._process_answer(response, question)
                # print("[RESULT After Process]", result)
            else:

                sys_prompt, content = self.vlm.format_prompt(prompt_type="answer_baseline",
                    question=question["question"],
                    answer_choices=question["answer_choices"],
                    images=[primary_img_path, helper_img_path] if len(question['img_paths']) > 1 else [primary_img_path],
                )
                response = self.vlm.run_prompt("answer_baseline", sys_prompt, content)
                print("[LLM]", response)
                result = self._process_answer(response, question)
                                
            # import pdb; pdb.set_trace() 
                    
            self._dump_llm_interaction(save_dir, 0, question, response, result, None, None
                                       , planning=policy, policies=policies)
            
            if result == "out of control":
                result = "wrong"
                
            if result in ("correct", "wrong"):
                self.results["progress"][question["question_type"]][result].append(qid)
            
            all_types = self.results["progress"].keys()
            correct_total = sum(len(self.results["progress"][t]["correct"]) for t in all_types)
            wrong_total = sum(len(self.results["progress"][t]["wrong"]) for t in all_types)
            
            self.results["current"] = f"{correct_total + wrong_total + len(self.results['skip_indices'])} / {len(self.questions)}"
            
            self.results["accuracy"]["all"] = correct_total / (correct_total + wrong_total)
            for t in all_types:
                if len(self.results["progress"][t]["correct"]) + len(self.results["progress"][t]["wrong"]) != 0:
                    self.results["accuracy"]["types"][t] = len(self.results["progress"][t]["correct"]) / (
                        len(self.results["progress"][t]["correct"]) + len(self.results["progress"][t]["wrong"])
                    )
                else:
                    self.results["accuracy"]["types"][t] = None
            self.save_results()
            
    

    def _parse_json_obj(self, text: str):
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    

    def _parse_policy_json(self, text: str):
        # Extract first JSON object if model wraps it
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None

        # basic validation
        decision = obj.get("decision", "").strip().lower()
        if decision not in ("skip", "call_wm"):
            return None
        actions = obj.get("actions", [])
        if decision == "skip":
            return {"decision": "skip", "actions": [], "reason": obj.get("reason", "")}

        if not isinstance(actions, list) or len(actions) == 0:
            return None

        parsed = []
        for a in actions:
            if not isinstance(a, dict): 
                return None
            t = a.get("type", "").strip()
            v = a.get("value", None)
            if t not in ("move-forward", "turn-left", "turn-right"):
                return None
            try:
                v = float(v)
            except Exception:
                return None
            
            # process into fixed intervals
            if t == 'move-forward':
                n = int(v / self.model_args.sampling_interval_meter)
                for _ in range(n):
                    parsed.append({"type": t, "value": self.model_args.sampling_interval_meter})
            else:
                n = int(v / self.model_args.sampling_interval_angle)
                for _ in range(n):
                    parsed.append({"type": t, "value": self.model_args.sampling_interval_angle})
                    
            # parsed.append({"type": t, "value": v})

        return {"decision": "call_wm", "actions": parsed, "reason": obj.get("reason", "")}


    # ------------------------------------------------------------------
    #  LLM RESPONSE PARSING
    # ------------------------------------------------------------------
    def _process_answer(self, response: str, question: dict, fwd=0.075, turn=3):
        """Parse LLM response and map to (result, actions, magnitude)."""
        response_l = response.lower()
        try:
            if any(c.lower() in response_l for c in question["answer_choices"]):
                return "correct" if question["correct_answer"].lower() in response_l.split("\n")[-1] else "wrong"
        except Exception:
            pass
        return "out of control"
    
    
    def _is_plan_suspicious(self, action_seq_strs):
        """
        Simple static checks to catch common plan failures.
        action_seq_strs like ['turn-right 9.0','turn-left 9.0',...]
        """
        if not action_seq_strs:
            return True
        if len(action_seq_strs) > getattr(self.model_args, "max_action_ids_cap", 6):
            return True
        # immediate cancellation patterns
        for a, b in zip(action_seq_strs, action_seq_strs[1:]):
            if ("turn-left" in a and "turn-right" in b) or ("turn-right" in a and "turn-left" in b):
                return True
        return False
    
    
    def _process_score(self, response: str):
        """Parse LLM response to a list."""
        try:
            if "Output:" in response:
                response = response.split("Output:")[1]
            score = int(response.strip())
            return score
        except Exception:
            pass
        return "out of control"
    
    
    
    def _process_bbox(self, response: str):
        """Parse LLM response to a bounding box: [(x1,y1), (x2,y2)]. response is in format (150,160):(180,220)."""
        try:
            if "Output:" in response:
                response = response.split("Output:")[1]
            if "None" in response:
                return None
            list_= []
            coordinates = response.split(":")
            for coordinate in coordinates:
                coordinate = coordinate.strip()[1:-1].split(',')
                list_.append((int(coordinate[0]), int(coordinate[1])))
            return list_
        except Exception:
            pass
        return "out of control"

    def bbox_mask(self, bbox, image):
        (x1, y1), (x2, y2) = bbox
        h, w = image.shape[:2]
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        return mask
    
    def get_action_command_and_magnitude(self, action_str):
        return action_str.split(" ")[0], float(action_str.split(" ")[1])


    
    def _dump_llm_interaction(self, save_dir, step, question, response, result, actions, magnitude, planning, policies):
        prompt = copy.deepcopy(self.vlm.curr_prompt)
        log = {
            "question": question,
            "result": result,
            "action_list": actions,
            "magnitude": magnitude,
            "llm_response": response,
            "planning": planning,
            "prompt": prompt,
            "policies": policies,
        }
        step_dir = os.path.join(save_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "gpt.json"), "w") as f:
            json.dump(log, f, indent=2)
    
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

        # all_trajectories, all_trajectories_json = zip(*results)
        # return all_trajectories, all_trajectories_json
        return
    
    def _plan_to_action_ids(self, planned_actions):
        action_ids = []
        action_seq_strs = []  # for readable keys / logging

        for a in planned_actions:
            t, v = a["type"], float(a["value"])
            if t == "move-forward":
                num = max(1, round(v / self.model_args.sampling_interval_meter))
                action_ids += [ActionSpace.MOVE_FORWARD] * num
                action_seq_strs.append(f"move-forward {v:.2f}")
                
            elif t == "turn-left":
                num = max(1, round(v / self.model_args.sampling_interval_angle))
                action_ids += [ActionSpace.TURN_LEFT] * num
                action_seq_strs.append(f"turn-left {v:.1f}")
                
            elif t == "turn-right":
                num = max(1, round(v / self.model_args.sampling_interval_angle))
                action_ids += [ActionSpace.TURN_RIGHT] * num
                action_seq_strs.append(f"turn-right {v:.1f}")

        return action_ids, action_seq_strs
    
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
        step_dir = os.path.join(save_dir, f"step_{step_idx}")
        os.makedirs(step_dir, exist_ok=True)

        folder_name = "plan_" + "_".join(s.replace(" ", "_") for s in action_seq_strs)
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


        
    def _simulate_all_actions(
        self,
        image_path: str,
        step_idx: int,
        actions: List[str],  # e.g. ["move-forward 0.75", "turn-left 30"]
        sampling_interval_angle: int,  # e.g. 10° → sample every 10, 20, 30° …
        sampling_interval_meter: float,  # e.g. 0.25 m → sample every 0.25 m …
        model_args,
        save_dir: str,
        previous_action_sequences: Optional[List[List[str]]] = None,
        previous_action_lists: Optional[List[List[int]]] = None,
        sequential: bool = False,  # <- currently unused but kept for backward-compatibility
    ) -> Dict[str, Dict[str, str]]:
        """Simulate a batch of candidate actions and return sampled frame paths.

        Parameters
        ----------
        image_path
            Path to the current RGB frame that represents the agent's egocentric view.
        step_idx
            Index of the current decision step (0-based).
        actions
            Human-readable action strings - MUST follow the pattern
            ``"<verb> <magnitude>"``, e.g. ``"turn-left 30"`` or ``"move-forward 0.75"``.
        sampling_interval_angle
            Angular step (in degrees) at which to sample frames **for turning actions**.
        sampling_interval_meter
            Linear step (in meters) at which to sample frames **for forward actions**.
        model_args
            Namespace / dataclass that carries meta-args for the world-model (must expose
            ``num_frames`` and ``frame_interval``).
        save_dir
            Root directory where all rendered videos & sampled frames will be stored.
        previous_action_sequences, previous_action_lists
            Optional history of actions already executed in this episode.  They are used
            to stitch *new* candidate actions onto *previous* ones so that a single video
            containing the full compound trajectory can be generated and sampled.
            - ``previous_action_sequences`` contains the *raw* strings, e.g.
            ``[["move-forward 0.5", "turn-right 30"], …]``.
            - ``previous_action_lists``     contains the *encoded* ActionSpace IDs that
            the low-level controller needs, same outer list structure as above.

        Returns
        -------
        Dict[str, Dict[str, str]]
            ``{ top_key → { sub_key → frame_path } }`` where
            - *top_key*  encodes the **action family** (e.g. "turn left").
            - *sub_key*  is an individual sample (e.g. "turn left 20 degrees").
            - *frame_path* is a PNG outside ``save_dir`` pointing to the sampled frame.
        """

        # ---------------------------------------------------------------------
        # 0. Book-keeping variables & hyper-parameters
        # ---------------------------------------------------------------------
        action_candidates: Dict[str, Dict[str, str]] = {}

        # Folder layout:  <save_dir>/step_<step_idx>/<action_folder>/pred.mp4
        step_dir = os.path.join(save_dir, f"step_{step_idx}")
        os.makedirs(step_dir, exist_ok=True)

        fixed_length: int = model_args.num_frames - 1  # frames per **new** action
        turn_size: int = model_args.frame_interval * 3  # degrees per frame when turning
        forward_size: float = model_args.frame_interval * (0.25 / 3)  # metres per frame

        # We accumulate the following four lists to run the world-model **once**
        # for all candidate branches - provides huge speed-ups on GPUs.
        action_lists: List[List[int]] = []          # low-level ActionSpace IDs
        magnitudes: List[float]       = []          # angle / distance for each branch
        top_key_list: List[str]       = []          # family key → first-level dict
        action_folder_name_list: List[str] = []     # filesystem folder for the branch
        prev_action_len_list: List[int] = []        # #frames of *prepended* history

        # ---------------------------------------------------------------------
        # 1. Parse *each* provided high-level action string
        # ---------------------------------------------------------------------
        print("[SpatialVQA] Simulating actions:", actions)
        for action_str in actions:
            # --- 1-A. Validate & split ------------------------------------------------
            tokens = action_str.strip().split()
            if len(tokens) != 2:
                print(f"[WARN] Skip invalid action '{action_str}' - expected '<verb> <mag>'.")
                continue

            raw_action, magnitude_s = tokens
            try:
                magnitude = float(magnitude_s)  # "30" → 30.0
            except ValueError:
                print(f"[WARN] Cannot parse magnitude in '{action_str}'.  Skipping…")
                continue

            # Helper to map the *verb* into ActionSpace & human-readable family key.
            def _verb_to_ids(verb: str):
                if verb == "turn-left":
                    return ActionSpace.TURN_LEFT, "turn left"
                if verb == "turn-right":
                    return ActionSpace.TURN_RIGHT, "turn right"
                if verb == "move-forward":
                    return ActionSpace.MOVE_FORWARD, "move forward"
                raise ValueError(f"Unsupported action verb '{verb}'.")

            try:
                action_id, family_key = _verb_to_ids(raw_action)
                # print("action_id:", action_id)
                # print("family_key:", family_key)
            except ValueError as exc:
                print(f"[WARN] {exc}.  Skipping…")
                continue

            # -----------------------------------------------------------------
            # 1-B. Expand *either* plain actions *or* prepend history branches
            # -----------------------------------------------------------------
            if previous_action_sequences is None:
                # --- No history ⇒ single branch --------------------------------
                action_list = [action_id] * fixed_length
                folder_name = f"{raw_action}_{magnitude:.2f}"
                prev_len = 0

                # «Register» this branch
                action_lists.append(action_list)
                magnitudes.append(magnitude)
                top_key_list.append(family_key)
                action_folder_name_list.append(folder_name)
                prev_action_len_list.append(prev_len)
                action_candidates.setdefault(family_key, {})
            else:
                # --- Fan-out: prepend each *history* and test the new action ----
                for hist_raw, hist_ids in zip(previous_action_sequences, previous_action_lists):
                    curr_action_command, curr_action_magnitude = self.get_action_command_and_magnitude(action_str)
                    last_action_command, last_action_magnitude = self.get_action_command_and_magnitude(hist_raw[-1])
                    if curr_action_command == "turn-left" and last_action_command == "turn-right":
                        print(f"[WARN] Skip invalid action '{action_str}' - cannot turn left after turning right.")
                        continue
                    if curr_action_command == "turn-right" and last_action_command == "turn-left":
                        print(f"[WARN] Skip invalid action '{action_str}' - cannot turn right after turning left.")
                        continue
                    if curr_action_command == last_action_command:
                        if curr_action_command in ("turn-left", "turn-right") and (curr_action_magnitude + last_action_magnitude) > self.model_args.max_turn_angle:
                            print(f"[WARN] Skip invalid action '{action_str}' - turning too far.")
                            continue
                        if curr_action_command == "move-forward" and (curr_action_magnitude + last_action_magnitude) > self.model_args.max_forward_distance:
                            print(f"[WARN] Skip invalid action '{action_str}' - moving too far.")
                            continue

                    hist_folder_prefix = "_".join(act.replace(" ", "_") for act in hist_raw) + "_"
                    hist_key_prefix   = ", ".join(act.replace("-", " ") for act in hist_raw) + ", and then "
                    new_action_list = hist_ids + [action_id] * (fixed_length - len(hist_ids))
                    folder_name     = f"{hist_folder_prefix}{raw_action}_{magnitude:.2f}_meters" if curr_action_command == "move-forward" else f"{hist_folder_prefix}{raw_action}_{magnitude:.2f}_degrees"
                    family_full_key = hist_key_prefix + family_key
                    action_lists.append(new_action_list)
                    magnitudes.append(magnitude)
                    top_key_list.append(family_full_key)
                    action_folder_name_list.append(folder_name)
                    prev_action_len_list.append(len(hist_ids))
                    action_candidates.setdefault(family_full_key, {})

        print(f"!!!prev_action_len_list:{prev_action_len_list}")

        # ---------------------------------------------------------------------
        # 2. Run *once* through the world-model to render *ALL* branches
        # ---------------------------------------------------------------------
        self._take_action_with_world_model(
            step_idx=step_idx,
            img_path=image_path,
            action_lists=action_lists,
            magnitudes=magnitudes,
            step_dir=step_dir,
            action_folder_name_list=action_folder_name_list,
            forward_size=forward_size,
            turn_size=turn_size,
            num_workers=self.model_args.max_inference_batch_size
        )

        # ---------------------------------------------------------------------
        # 3. Sample frames from the rendered videos at the desired intervals
        # ---------------------------------------------------------------------
        for family_key, folder_name, prev_len, action_list, magnitude in zip(
            top_key_list,
            action_folder_name_list,
            prev_action_len_list,
            action_lists,
            magnitudes,
        ):
            video_path = os.path.join(step_dir, folder_name, "pred.mp4")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[ERR ] Could not open {video_path!r} - skip this branch.")
                continue

            # Edge-case: corrupted or too-short videos
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < 2:
                print(f"[ERR ] Not enough frames in {video_path!r} - skip.")
                cap.release()
                continue

            # -----------------------------------------------------------------
            # Compute the frame indices we want to grab (domain-specific maths)
            # -----------------------------------------------------------------
            if action_list[-1] != ActionSpace.MOVE_FORWARD:  # ↻ turning
                arr = np.arange(0, magnitude + 1, turn_size)
                targets = np.arange(0, magnitude + 1, sampling_interval_angle)
            else:  # → moving forward
                arr = np.arange(0, magnitude + 1e-3, forward_size)
                targets = np.arange(0, magnitude + 1e-3, sampling_interval_meter)

            sampled_indices = [int(np.abs(arr - t).argmin()) for t in targets]
            print("!!!sampled_indices", sampled_indices)
            if self.world_model_type == "cogvideox" or self.world_model_type == None:
                sampled_indices = sampled_indices[1:]  # drop the first frame (identical to input)
            elif self.world_model_type == "svc":
                sampled_indices = [x * 2 for x in sampled_indices]
                sampled_indices = sampled_indices[:-1]
            print("!!!sampled_indices", sampled_indices)

            # -----------------------------------------------------------------
            # Read & save sampled frames ☑
            # -----------------------------------------------------------------
            for i, frame_idx in enumerate(sampled_indices, start=1):
                if self.world_model_type == "cogvideox" or self.world_model_type == None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + prev_len)
                elif self.world_model_type == "svc":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + prev_len*2)
                success, frame = cap.read()
                if not success:
                    print(f"[WARN] Cannot grab frame {frame_idx+prev_len} from {video_path!r}.")
                    continue

                # Derive human-readable *sub-key* and filename ------------------
                if action_list[-1] != ActionSpace.MOVE_FORWARD:  # turning
                    metric_val = i * sampling_interval_angle
                    fname = f"sample_{metric_val}.png"
                    sub_key = f"{family_key} {metric_val} degrees"
                else:  # forward
                    metric_val = i * sampling_interval_meter
                    fname = f"sample_{metric_val}.png"
                    sub_key = f"{family_key} {metric_val} meters"

                out_path = os.path.join(step_dir, folder_name, fname)
                cv2.imwrite(out_path, frame)
                action_candidates[family_key][sub_key] = out_path

            cap.release()

        return action_candidates

# -----------------------------------------------------------------------------
#  CLI ENTRY
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise pipeline & run
    pipeline = SpatialVQAPipelineSVC()
    print("!!!!!!!!!!!!!!!!")
    pipeline.run()