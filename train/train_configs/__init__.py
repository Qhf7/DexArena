"""Training configurations, one module per trainer.

The package-level names below are resolved on first use, so importing one
config module (for example ``train.train_configs.pi05_config`` from the PI0.5
policy worker, which runs in an interpreter without the simulator) does not
import every trainer's dependencies.
"""
import importlib

_EXPORTS = {
    "AWRArgs": ("awr_config", "Args"),
    "parse_awr_args": ("awr_config", "parse_args"),
    "BCArgs": ("bc_config", "Args"),
    "parse_bc_args": ("bc_config", "parse_args"),
    "CQLArgs": ("cql_config", "Args"),
    "parse_cql_args": ("cql_config", "parse_args"),
    "ACTArgs": ("ACT_config", "Args"),
    "parse_act_args": ("ACT_config", "parse_args"),
    "DPArgs": ("DP_config", "Args"),
    "parse_dp_args": ("DP_config", "parse_args"),
    "FQEArgs": ("fqe_config", "Args"),
    "parse_fqe_args": ("fqe_config", "parse_args"),
    "IQLArgs": ("iql_config", "Args"),
    "parse_iql_args": ("iql_config", "parse_args"),
    "IQLOnlineArgs": ("iql_online_config", "Args"),
    "parse_iql_online_args": ("iql_online_config", "parse_args"),
    "OPTArgs": ("opt_config", "Args"),
    "parse_opt_args": ("opt_config", "parse_args"),
    "PPOArgs": ("ppo_config", "Args"),
    "parse_ppo_args": ("ppo_config", "parse_args"),
    "RLPDArgs": ("rlpd_config", "Args"),
    "parse_rlpd_args": ("rlpd_config", "parse_args"),
    "SACArgs": ("sac_config", "Args"),
    "parse_sac_args": ("sac_config", "parse_args"),
    "TD3BCArgs": ("td3bc_config", "Args"),
    "parse_td3bc_args": ("td3bc_config", "parse_args"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(f"{__name__}.{module}"), attribute)
