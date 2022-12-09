import torch

import utils
from .other import device
from model import ACModel
import numpy as np
import openai

import time


class GPTAgent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False, backend="text-curie-001"):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.acmodel = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.argmax = argmax
        self.num_envs = num_envs
        with open('utils/open_ai_key.txt', 'r') as f:
            openai.api_key = f.readlines()[0]
        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))
        self.backend = backend


    def relative_goal_location(self, image):
        # size is 7x7
        idx = np.where(image == 8)

        if len(idx[0]) == 0:  # the goal cannot be seen
            return "at an unknown location"
        row = idx[0].item()
        col = idx[1].item()
        how_far_fwd = 6 - col
        how_far_right = row - 3
        if how_far_fwd > 0 :
            fwd = f"forward {how_far_fwd} squares"
            if how_far_right != 0:
                fwd += ' and '
        else:
            fwd = ''
        # if how_far_fwd > 1:
        #     fwd += 's'
        lateral = ''
        if how_far_right == 0:
            lateral = ''
        elif how_far_right > 0:
            if how_far_right > 1:
                word = 'squares'
            else:
                word = 'square'
            lateral = f'{how_far_right} {word} to the right'
        elif how_far_right < 0:
            if how_far_right < -1:
                word = 'squares'
            else:
                word = 'square'
            lateral = f'{-how_far_right} {word} to the left'
        return fwd + lateral


    def relative_wall_location(self, image):
        # size is 7x7
        # print(np.equal(image, [2, 5, 0]).all(-1))
        cols, = np.where(np.equal(image[3], [2, 5, 0]).all(-1))
        rows, = np.where(~np.equal(image[:, -1], [2, 5, 0]).all(-1))
        if len(rows) == 0:  # the walls cannot be seen
            return ""
        how_far_fwd = 6 - cols.max() - 1
        # how_far_left = 4 - rows.min() - 1
        # how_far_right = rows.max() - 2 - 1  # TODO handle cases when all some other walls cannot be seen.
        output = f"There is a wall located {how_far_fwd} squares in front of you."
        # , {how_far_right} squares to the right, and {how_far_left} to the left."
        return output

    def get_actions(self, obss):
        """
        Direction 0 is right, 1 down, 3 up, 2 left
        actions
        0: turn left
        1: turn right
        2: move forward one square

        """
        dirs = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
        actions = np.zeros((len(obss),))
        for i, obs in enumerate(obss):


            example = (
                "You are in a grid-world and the goal square is located forward 1 square and 1 square to the right relative to your current position. There is a wall located 1 squares in front of you. The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square.\n"
                "1. move forward\n"
                "2. turn right\n"
                "3. move forward\n"
                "Goal reached!\n"
            )
            example += '\n\n'
            example += (
                "You are in a grid-world and the goal square is located 2 squares to the left relative to your current position. There is a wall located 2 squares in front of you. The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square. \n"
                "1. turn left\n"
                "2. move forward\n"
                "3. move forward\n"
                "Goal reached!\n"
            )
            example += '\n\n'
            example += (
                "You are in a grid-world and the goal square is located at an unknown location relative to your current position. There is a wall located 1 squares in front of you. The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square.\n "
                "1. turn left\n"
            )
            example += '\n\n'
            prompt = (f"You are in a grid-world and the goal square is located {self.relative_goal_location(obs['image'])} relative to your current position. {self.relative_wall_location(obs['image'])} The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square. \n")
            example += prompt
            # example = (
            #     "You are in a grid-world and the goal square is located forward 1 square and 1 square to the right relative to your current position. There are walls located 1 squares in front of you, 1 squares to the right, and 1 to the left. The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square.\n"
            #     "1. move forward\n"
            #     "2. turn right\n"
            #     "3. move forward\n"
            #     "Goal reached!\n"
            # )
            # example += '\n\n'
            # example += (
            #     "You are in a grid-world and the goal square is located 2 squares to the left relative to your current position. There are walls located 2 squares in front of you, 0 squares to the right, and 2 to the left. The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square. \n"
            #     "1. turn left\n"
            #     "2. move forward\n"
            #     "3. move forward\n"
            #     "Goal reached!\n"
            # )
            # example += '\n\n'
            # example += (
            #     "You are in a grid-world and the goal square is located at an unknown location relative to your current position. There are walls located 1 squares in front of you, 0 squares to the right, and 2 to the left. The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square.\n "
            #     "1. turn left\n"
            # )
            # example += '\n\n'
            # prompt = (f"You are in a grid-world and the goal square is located {self.relative_goal_location(obs['image'])} relative to your current position. {self.relative_wall_location(obs['image'])} The possible actions you can take are to turn left, turn right, or move forward one square. Write a list of actions to reach the goal square. \n")
            # example += prompt
            time.sleep(10.0)
            response = openai.Completion.create(
                # model="text-curie-001",
                model=self.backend,
                prompt=example,
                temperature=0.6,
                max_tokens=1000,
            )
            response = response.choices[0].text
            first_step = response.split('\n')[0]
            if "forward" in first_step:
                actions[i] = 2
            elif "right" in first_step:
                actions[i] = 1
            elif "left" in first_step:
                actions[i] = 0
            # actions[i] = np.random.randint(3)
        return actions

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        pass

    def analyze_feedback(self, reward, done):
        pass
