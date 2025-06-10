from rl_games.common.algo_observer import AlgoObserver

from isaacgymenvs.utils.utils import retry
from isaacgymenvs.utils.reformat import omegaconf_to_dict
import os


class WandbAlgoObserver(AlgoObserver):
    """Need this to propagate the correct experiment name after initialization."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.logged_checkpoints = set()
        self.checkpoints_dir = None  # Will be set after first call

    def before_init(self, base_name, config, experiment_name):
        """
        Must call initialization of Wandb before RL-games summary writer is initialized, otherwise
        sync_tensorboard does not work.
        """

        import wandb

        self.wandb_unique_id = f"uid_{experiment_name}"
        print(f"Wandb using unique id {self.wandb_unique_id}")

        cfg = self.cfg

        # this can fail occasionally, so we try a couple more times
        @retry(3, exceptions=(Exception,))
        def init_wandb():
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                group=cfg.wandb_group,
                tags=cfg.wandb_tags,
                sync_tensorboard=True,
                id=self.wandb_unique_id,
                name=experiment_name,
                resume=True,
                settings=wandb.Settings(start_method='fork'),
            )
       
            if cfg.wandb_logcode_dir:
                wandb.run.log_code(root=cfg.wandb_logcode_dir)
                print('wandb running directory........', wandb.run.dir)

        print('Initializing WandB...')
        try:
            init_wandb()
        except Exception as exc:
            print(f'Could not initialize WandB! {exc}')

        if isinstance(self.cfg, dict):
            wandb.config.update(self.cfg, allow_val_change=True)
        else:
            wandb.config.update(omegaconf_to_dict(self.cfg), allow_val_change=True)

    def after_steps(self):
        import wandb

        # Set checkpoints_dir if not set
        if self.checkpoints_dir is None:
            # Example: runs/EXPERIMENT_NAME/nn/
            experiment_name = wandb.run.name if wandb.run else "default"
            self.checkpoints_dir = os.path.join("runs", experiment_name, "nn")

        if not os.path.isdir(self.checkpoints_dir):
            return

        # Scan for new checkpoint files
        for fname in os.listdir(self.checkpoints_dir):
            fpath = os.path.join(self.checkpoints_dir, fname)
            if os.path.isfile(fpath) and fpath not in self.logged_checkpoints:
                # Log new checkpoint to wandb
                artifact = wandb.Artifact(
                    self.wandb_unique_id,
                    type="model",
                    description=f"Checkpoint {fname}",
                )
                artifact.add_file(fpath)
                wandb.log_artifact(artifact)
                print(f"Logged checkpoint to wandb: {fpath}")
                self.logged_checkpoints.add(fpath)
