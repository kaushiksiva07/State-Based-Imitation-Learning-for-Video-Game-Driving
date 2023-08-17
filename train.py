import torch
from model import Network, save_model
from os import path
import torch.utils.tensorboard as tb
from utils import load_data
from state_agent import player


def train(args):
    model = Network()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Storing Training Data
    if not args.dagger:
        train_type = 'train'
        valid_type = 'valid'
    else:
        train_type = 'dagger'
        valid_type = 'dagger_valid'
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)),
                                                   'model.th'),
                                         map_location=torch.device('cpu')))

    train_data = load_data(train_type)
    valid_data = load_data(valid_type)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Train
    model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'state_agent/model.th')))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    accel_loss = torch.nn.MSELoss()
    steer_loss = torch.nn.MSELoss()
    brake_loss = torch.nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        for train_states, train_actions in train_data:
            train_states = train_states.to(device)
            train_actions = train_actions.to(device)
            output = model(train_states)

            accel_loss_train = accel_loss(output[:, 0], train_actions[:, 0])
            steer_loss_train = steer_loss(output[:, 1], train_actions[:, 1])
            brake_loss_train = brake_loss(output[:, 2], train_actions[:, 2])
            loss_train = accel_loss_train + steer_loss_train + brake_loss_train

            if train_logger is not None:
                train_logger.add_scalar('accel_loss', accel_loss_train, global_step)
                train_logger.add_scalar('steer_loss', steer_loss_train, global_step)
                train_logger.add_scalar('brake_loss', brake_loss_train, global_step)
                train_logger.add_scalar('train_loss', loss_train, global_step)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            global_step += 1
        save_model(model)
        # Validation
        if valid_logger is not None:
            model.eval()
            for valid_states, valid_actions in valid_data:
                valid_states, valid_actions = valid_states.to(device), valid_actions.to(device)
                output = model(valid_states)
                accel_loss_valid = accel_loss(output[:, 0], valid_actions[:, 0])
                steer_loss_valid = steer_loss(output[:, 1], valid_actions[:, 1])
                brake_loss_valid = brake_loss(output[:, 2], valid_actions[:, 2])
                loss_valid = accel_loss_valid + steer_loss_valid + brake_loss_valid
                valid_logger.add_scalar('valid_loss', loss_valid, global_step)
        scheduler.step()
    # torch.jit.script(model).save('state_agent.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=150)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('--dagger', action='store_true')

    args = parser.parse_args()
    train(args)
