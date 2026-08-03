"""Convert a mapping to typed settings (MRST ``private/struct2settings.m``)."""

from collections.abc import Mapping

from pymrst.utils.settings_manager.settings_struct import SettingsStruct


def struct_to_settings(oldsettings) -> SettingsStruct:
    if not isinstance(oldsettings, Mapping):
        raise TypeError("settings conversion requires a mapping")
    supported = SettingsStruct.__dataclass_fields__
    unknown = sorted(set(oldsettings) - set(supported))
    if unknown:
        raise KeyError(f"unsupported setting names: {', '.join(unknown)}")
    defaults = SettingsStruct().to_dict()
    defaults.update(oldsettings)
    return SettingsStruct(**defaults).validate()


struct2settings = struct_to_settings
