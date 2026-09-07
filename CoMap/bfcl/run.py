"""Register CoMAP, then dispatch the unmodified official BFCL CLI."""

import os


def main():
    from bfcl_eval.constants.eval_config import DOTENV_PATH
    from dotenv import load_dotenv

    load_dotenv(DOTENV_PATH, override=True)
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from .handler import CoMAPHandler

    name = os.getenv("COMAP_REGISTRY_NAME", "CoMAP-FC")
    MODEL_CONFIG_MAPPING[name] = ModelConfig(
        model_name=os.getenv("COMAP_POLICY_MODEL", "comap-policy"),
        display_name=name,
        url="https://github.com/loyiv/CoMAP",
        org="Local",
        license="See underlying model license",
        model_handler=CoMAPHandler,
        is_fc_model=True,
        underscore_to_dot=True,
    )
    from bfcl_eval.__main__ import cli

    cli()


if __name__ == "__main__":
    main()
