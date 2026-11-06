# adopted from https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo.py

import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim
from utils.logger import Logger, SummaryWriter
from utils.paths import ensure_parent_dir, runs_path

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, args, envs, run_name, bc_dataset=None):
        super().__init__()
        self.args = args
        self.actor_critic = ActorCritic(envs, logstd_init=float(getattr(args, "actor_logstd_init", -0.2)))
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=args.learning_rate, eps=1e-5)

        writer = SummaryWriter(runs_path(run_name))
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        self.logger = Logger(log_swanlab=args.track, tensorboard=writer)

        self.bc_data_path = getattr(
            args,
            "bc_data_path",
            "data/teleop_reduced_to_full_outputs",
        )
        self.bc_coef = float(getattr(args, "bc_coef", 1.0))
        self.bc_min_coef = float(getattr(args, "bc_min_coef", 0.0))
        self.bc_decay = bool(getattr(args, "bc_decay", True))
        self.bc_loss_type = getattr(args, "bc_loss_type", "mse")
        self.use_bc_loss = bool(getattr(args, "use_bc_loss", True))

        default_decay_steps = int(getattr(args, "total_timesteps", 1))
        bc_decay_fraction = float(getattr(args, "bc_decay_steps", 1.0))
        if bc_decay_fraction <= 0:
            bc_decay_fraction = 1.0
        self.bc_decay_steps = max(1, int(default_decay_steps * bc_decay_fraction))

        bc_batch_size = int(getattr(args, "bc_batch_size", args.minibatch_size))
        self.bc_batch_size = bc_batch_size if bc_batch_size > 0 else args.minibatch_size

        self.use_bc = False
        self.num_bc_samples = 0
        self.bc_obs = None
        self.bc_actions = None

        if (
            bc_dataset is not None
            and self.use_bc_loss
            and self.bc_coef > 0
            and bc_dataset.bc_obs is not None
            and bc_dataset.bc_actions is not None
        ):
            self.bc_obs = bc_dataset.bc_obs
            self.bc_actions = bc_dataset.bc_actions
            self.num_bc_samples = self.bc_obs.shape[0]
            self.use_bc = self.num_bc_samples > 0
        elif self.use_bc_loss and self.bc_coef > 0 and self.bc_data_path:
            print(f"BC dataset was not preloaded from {self.bc_data_path}. BC disabled.")

        if self.args.checkpoint:
            self.load()

    def _get_bc_coef(self, global_step):
        if not self.use_bc or self.bc_coef <= 0:
            return 0.0
        if not self.bc_decay:
            return self.bc_coef
        progress = min(max(global_step / max(1, self.bc_decay_steps), 0.0), 1.0)
        return max(self.bc_min_coef, self.bc_coef * (1.0 - progress))

    def _sample_bc_batch(self, device):
        batch_size = min(self.bc_batch_size, self.num_bc_samples)
        sample_inds = torch.randint(0, self.num_bc_samples, (batch_size,))
        bc_obs = self.bc_obs[sample_inds].to(device=device, non_blocking=True)
        bc_actions = self.bc_actions[sample_inds].to(device=device, non_blocking=True)
        return bc_obs, bc_actions

    def _compute_bc_loss(self, bc_obs, bc_actions):
        if self.bc_loss_type == "nll":
            _, bc_logprob, _, _ = self.actor_critic.get_action_and_value(bc_obs, bc_actions)
            return -bc_logprob.mean()
        action_mean = self.actor_critic.get_action(bc_obs, deterministic=True)
        return ((action_mean - bc_actions) ** 2).mean()

    def load(self):
        self.actor_critic.load_state_dict(torch.load(self.args.checkpoint))

    def save(self, model_path):
        model_path = ensure_parent_dir(model_path)
        torch.save(self.actor_critic.state_dict(), model_path)
        print(f"model saved to {model_path}")

    def bc_pretrain(self, num_steps, log_every=100):
        """Pure-BC warmup: pretrain the actor mean on demos before PPO starts.
        Only the actor_mean is updated (BC uses the deterministic mean); the
        actor_logstd and critic are untouched."""
        if not self.use_bc or num_steps <= 0:
            return
        device = next(self.actor_critic.parameters()).device
        self.actor_critic.train()
        loss_item = 0.0
        for i in range(num_steps):
            bc_obs, bc_actions = self._sample_bc_batch(device)
            loss = self._compute_bc_loss(bc_obs, bc_actions)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            loss_item = loss.item()
            if i % log_every == 0:
                self.logger.add_scalar("bc_pretrain/loss", loss_item, i)
                print(f"bc_pretrain step {i}/{num_steps}: loss={loss_item:.4f}")
        print(f"BC pretrain done: {num_steps} steps, final loss={loss_item:.4f}")

    def update(self, start_time, rollout_time, global_step, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        # Optimizing the policy and value network
        update_time = time.time()

        self.actor_critic.train()
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        bc_coef = self._get_bc_coef(global_step)
        bc_loss_item = 0.0
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.actor_critic.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                bc_loss = torch.zeros((), device=b_obs.device)
                if bc_coef > 0.0:
                    bc_obs, bc_actions = self._sample_bc_batch(b_obs.device)
                    bc_loss = self._compute_bc_loss(bc_obs, bc_actions)
                bc_loss_item = bc_loss.item()

                loss = (pg_loss - self.args.ent_coef * entropy_loss) + v_loss * self.args.vf_coef + bc_coef * bc_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        self.logger.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        self.logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.logger.add_scalar("losses/bc_loss", bc_loss_item, global_step)
        self.logger.add_scalar("losses/bc_coef", bc_coef, global_step)
        self.logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        self.logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        self.logger.add_scalar("time/step", global_step, global_step)
        self.logger.add_scalar("time/update_time", update_time, global_step)
        self.logger.add_scalar("time/rollout_time", rollout_time, global_step)
        self.logger.add_scalar("time/rollout_fps", self.args.num_envs * self.args.num_steps / rollout_time, global_step)

class ActorCritic(nn.Module):
    def __init__(self, envs, logstd_init: float = -0.2):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * logstd_init)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
