import os
import torch
from torch.utils.data import Dataset, DataLoader
from grader.utils import load_recording
from jurgen_agent.player import Team, extract_featuresV2
import pickle


def create_trajectories(match_file):
    team = Team()
    # Returns dictionary of trajectories for a single match
    if match_file.startswith('data2') or match_file.startswith('valid2') or match_file.startswith('dagger2'):
        player1_id = 1
        player2_id = 3
    else:
        player1_id = 0
        player2_id = 2
    game = load_recording(match_file)
    # match_traj = []
    match_traj = dict()
    for index, frame in enumerate(game):
        team1_state = frame['team1_state']
        team2_state = frame['team2_state']
        soccer_state = frame['soccer_state']

        if player1_id == 0:
            player1 = extract_featuresV2(team1_state[0], soccer_state, team2_state, 0)
            player2 = extract_featuresV2(team1_state[1], soccer_state, team2_state, 0)
            state = torch.stack([player1, player2], dim=0)
        else:
            player1 = extract_featuresV2(team2_state[0], soccer_state, team1_state, 1)
            player2 = extract_featuresV2(team2_state[1], soccer_state, team1_state, 1)
            state = torch.stack([player1, player2], dim=0)

        if match_file.startswith('dagger'):
            if player1_id == 0:
                team.team = 0
                actions = team.act(team1_state, team2_state, soccer_state)
            else:
                team.team = 1
                actions = team.act(team2_state, team1_state, soccer_state)

            player1_actions = torch.Tensor(list(actions[0].values()))
            player2_actions = torch.Tensor(list(actions[1].values()))
            actions = torch.stack([player1_actions, player2_actions], dim=0)

        else:
            player1_actions = torch.Tensor(list(frame['actions'][player1_id].values()))
            player2_actions = torch.Tensor(list(frame['actions'][player2_id].values()))
            actions = torch.stack([player1_actions, player2_actions], dim=0)

        # Check for Nan values
        if torch.isnan(state).any().item() or torch.isnan(actions).any().item():
            continue

        match_traj[index] = (state, actions)
    return match_traj


class TrainingDataset(Dataset):
    def __init__(self, train_type):
        self.train_type = train_type

        if not os.getcwd().endswith('final'):
            os.chdir("..")
        with open(self.train_type + '_data.pkl', 'rb') as f:
            self.data = pickle.load(f)

        self.data = list(self.data.values())
        self.states = torch.stack([d[0][0] for d in self.data])
        self.actions = torch.stack([d[1][0] for d in self.data])

        if train_type == 'dagger':
            with open('train_data.pkl', 'rb') as f:
                self.data2 = pickle.load(f)
            self.data2 = list(self.data2.values())
            self.states2 = torch.stack([d[0][0] for d in self.data2])
            self.actions2 = torch.stack([d[1][0] for d in self.data2])
            self.actions = torch.cat([self.actions, self.actions2], dim=0)
            self.states = torch.cat([self.states, self.states2], dim=0)

        elif train_type == 'dagger_valid':
            with open('valid_data.pkl', 'rb') as f:
                self.data2 = pickle.load(f)
            self.data2 = list(self.data2.values())
            self.states2 = torch.stack([d[0][0] for d in self.data2])
            self.actions2 = torch.stack([d[1][0] for d in self.data2])
            self.actions = torch.cat([self.actions, self.actions2], dim=0)
            self.states = torch.cat([self.states, self.states2], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        states = self.states[idx]
        actions = self.actions[idx]
        return states, actions


def load_data(train_type, num_workers=0, batch_size=128):
    dataset = TrainingDataset(train_type)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dagger', action='store_true')

    args = parser.parse_args()

    # Convert individual matches into a dictionary of trajectories
    if not args.dagger:
        data_names = ['data', 'data2', 'valid1', 'valid2']
        pkl_names = ['train_data.pkl', 'valid_data.pkl']
    else:
        data_names = ['dagger1', 'dagger2', 'dagger_valid1', 'dagger_valid2']
        pkl_names = ['dagger_data.pkl', 'dagger_valid_data.pkl']

    from os import path
    from glob import glob

    traj_dict = {}
    for i, name in enumerate(data_names):
        print('Aggregating ' + name + ' trajectories')
        files = []

        if not os.getcwd().endswith('final'):
            os.chdir("..")
        for file in glob(path.join(name, '*.pkl')):
            files.append(file)

        if len(traj_dict) == 0:
            j = 0
        print('Match Id:')
        for idx, match in enumerate(files):
            data = create_trajectories(match)
            for value in data.values():
                traj_dict[j] = value
                j += 1
            print(idx)
        if name.endswith('2'):
            with open(pkl_names[i//2], 'wb') as f:
                pickle.dump(traj_dict, f)
            traj_dict = {}
