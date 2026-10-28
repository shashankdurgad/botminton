"""PPO runner configurations for the badminton stages."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BadmintonPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "badminton_kuka_allegro"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BadmintonGraspPPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 1500
    run_name = "grasp"


@configclass
class BadmintonHitPPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 4000
    run_name = "hit"


@configclass
class BadmintonTargetPPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 4000
    run_name = "target"


@configclass
class BadmintonMobilePPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 4000
    run_name = "mobile"
    # Actions are clipped to [-1, 1], so beyond ~1 a larger std costs nothing and the entropy bonus inflates it
    # without limit (the stationary stages ended near 80). Cap it so exploration stays small around a good policy.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_range=(0.02, 0.5)),
    )
    # The warm-started arm already plays well: no entropy bonus (it only pushed the std up and the grip and swing
    # got noisier every iteration), and smaller policy steps.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


# -- the curriculum from scratch on the wheeled robot: the stationary settings, with the std capped at 1 so the
# entropy bonus cannot inflate it (beyond ~1 the clipped actions saturate and the extra noise carries no signal) --
_WHEELED_ACTOR = RslRlMLPModelCfg(
    hidden_dims=[512, 256, 128],
    activation="elu",
    obs_normalization=True,
    distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6, std_range=(0.02, 1.0)),
)


@configclass
class BadmintonMobileGraspPPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 1500
    run_name = "mobile_grasp"
    actor = _WHEELED_ACTOR


@configclass
class BadmintonMobileHitPPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 4000
    run_name = "mobile_hit"
    actor = _WHEELED_ACTOR


@configclass
class BadmintonMobileTargetPPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 4000
    run_name = "mobile_target"
    actor = _WHEELED_ACTOR


@configclass
class BadmintonMobileCoveragePPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 4000
    run_name = "mobile_coverage"
    actor = _WHEELED_ACTOR


@configclass
class BadmintonMobileRallyPPORunnerCfg(BadmintonPPORunnerCfg):
    max_iterations = 3000
    run_name = "mobile_rally"
    actor = _WHEELED_ACTOR
