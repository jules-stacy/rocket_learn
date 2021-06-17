from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition
from stable_baselines3.common.callbacks import CheckpointCallback
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from stable_baselines3 import PPO
from rlgym.utils.reward_functions.common_rewards import ConstantReward
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
from rlgym.wrappers.sb3_wrappers import SB3MultipleInstanceWrapper, SB3VecMonitor
from rlgym.utils.reward_functions.common_rewards import EventReward
from rlgym.utils.reward_functions.common_rewards import RewardIfBehindBall
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards import SaveBoostReward

if __name__ == '__main__':
    # set episode timeout (1800 sec = 30 min)
    default_tick_skip = 8
    physics_ticks_per_second = 120
    ep_len_seconds = 150
    max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

    # specify terminal conditions from commons
    term_cond1 = TimeoutCondition(max_steps)
    term_cond2 = GoalScoredCondition()

    # specify reward functions
    fps = 15
    obs_builder = AdvancedObs()
    all_rewards = CombinedReward((EventReward(goal=85, team_goal=0, concede=-55, touch=1, shot=10, save=25, demo=10),
                                  VelocityPlayerToBallReward(use_scalar_projection=False),
                                  RewardIfBehindBall(ConstantReward()),
                                  SaveBoostReward()),
                                 (1,
                                  (0.1 / fps),
                                  (0.01 / fps),
                                  (0.05 / fps)))


    # BallYCoordinateReward(),
    # RewardIfBehindBall(ConstantReward()),
    # VelocityPlayerToBallReward(use_scalar_projection=False),
    # SaveBoostReward(),
    # VelocityReward(),
    # FaceBallReward()),

    # reward_weights=(1))

    # 0.1 / fps,
    # 0.01 / fps,
    # 0.1 / fps / 23,
    # 0.01 / fps,
    # 1,
    # 1))

    # def get_match_args():
    #    return dict(
    #        team_size=1,
    #        tick_skip=8,
    #        reward_function=all_rewards,
    #        self_play=True,
    #        terminal_conditions=[TimeoutCondition(900), GoalScoredCondition()],
    #        obs_builder=obs_builder)
    # learner = PPO.load('C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\PPO\\ppo_bot__1000002_steps')
    # okgo = learner.policy

    def get_match_args():
        return dict(
            team_size=1,
            tick_skip=8,
            reward_function=all_rewards,
            self_play=True,
            terminal_conditions=[term_cond1],
            obs_builder=obs_builder)


    epic_path = r"Z:\Epic Games\rocketleague\Binaries\Win64\RocketLeague.exe"
    num_procs = 3
    num_ts = 25_000_000
    env = SB3VecMonitor(SB3MultipleInstanceWrapper(epic_path, num_procs, get_match_args, wait_time=45),
                        filename='C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\Reward_Details')

    # env = Monitor(rlgym.make("Duel", spawn_opponents=True, team_size=1, ep_len_minutes=1,
    #                 obs_builder=obs_builder, terminal_conditions=[term_cond1], reward_fn=all_rewards, game_speed=1))

    checkpoint = CheckpointCallback(save_freq=1_000_000 // env.num_envs + 1,
                                    save_path='C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\PPO\\',
                                    name_prefix="ppo_bot_",
                                    verbose=1)

    #learner = PPO("MlpPolicy", env, verbose=3, device='cuda', n_epochs=1, target_kl=0.02 / 1.5)
    learner = PPO.load(path='C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\PPO\\ppo_bot__70_000_140_steps', env=env, device='cuda', n_epochs=1, target_kl=0.02 / 1.5)
    learner.learn(total_timesteps=num_ts, callback=checkpoint)
    # learner = PPO(env=env, verbose=1, device='cuda', policy_kwargs=DaanPolicy)
    # learner.learn(total_timesteps=num_ts, callback=checkpoint)
    # learner.save('C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\PPO\\ppo_bot_175_000_000')
    # DaanPolicy = dict(policy=okgo)
    # model = PPO("MlpPolicy", env,
    # verbose=3, device="cpu")
    # model.learn(100_000_000)
    # model.save("policy")
    # create the environment
    # log_dir = "tmp/"
    # os.makedirs(log_dir, exist_ok=True)
    # env = SubprocVecEnvWrapper(rlgym.make("DuelSelf", spawn_opponents=True, team_size=1,
    #                               ep_len_minutes=1, obs_builder=obs_builder, terminal_conditions=[term_cond1],
    #                              reward_fn=all_rewards), num_instances=3)  # make the environment |

    # env = VecEnvWrapper(rlgym.make("DuelSelf", spawn_opponents=True, team_size=1, ep_len_minutes=1,
    # obs_builder=obs_builder, terminal_conditions=[term_cond1,term_cond2], reward_fn=all_rewards))  # make the
    # environment | env = Monitor(env, log_dir)

    # model = PPO("MlpPolicy", env, n_epochs=1, target_kl=0.02 / 1.5, learning_rate=1e-4, ent_coef=0.01, vf_coef=1,
    # gamma=0.995, verbose=3, batch_size=128, n_steps=2048, tensorboard_log="./logs/", device="cuda") checkpoint =
    # CheckpointCallback(10_000_000 // env.num_envs + 1, "policy")  # Only increments once all agents take step
    # model.learn(100_000_000, callback=checkpoint) model = PPO.load(
    # 'C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\PPO\\ppo_bot_first', env=env, verbose=1,
    # tensorboard_log="./ppo_rl_tensorboard/", device='cuda') model.learn(total_timesteps=39e5, callback=)
    # model.save('C:\\Users\\Daanesh\\PycharmProjects\\RocketLeagueRL\\PPO\\ppo_bot_first')

    # policy = learner.policy

    # obs = env.reset()
    # for i in range(5000):
    #    action, _states = learner.predict(obs, deterministic=True)
    #    obs, rewards, dones, info = env.step(action)
    # env.render()
