"""Keep the pipeline's user-simulator decoding contract on the native tau user."""

from tau2.user.user_simulator import UserSimulator


class PipelineUserSimulator(UserSimulator):
    """Native prompts and behavior, with deliberately omitted sampling arguments."""

    def __init__(self, *args, **kwargs):
        args_config = kwargs.get("llm_args") or {}
        if any(key in args_config for key in ("seed", "temperature", "top_p")):
            raise ValueError("Pipeline user must omit seed, temperature and top_p")
        super().__init__(*args, **kwargs)

    def set_seed(self, seed):
        # The native orchestrator invokes this hook. The confirmed pipeline
        # intentionally omits the provider's seed parameter for its user model.
        self.llm_args.pop("seed", None)
