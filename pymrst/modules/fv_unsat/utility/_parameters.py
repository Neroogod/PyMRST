"""Parameter access shared by van Genuchten-Mualem utilities."""


def get_parameter(phys, name):
    flow = phys.get("flow", phys) if isinstance(phys, dict) else getattr(phys, "flow", phys)
    if isinstance(flow, dict):
        return flow[name]
    return getattr(flow, name)
