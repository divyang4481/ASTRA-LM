import yaml
from dataclasses import fields, is_dataclass
from typing import Type, TypeVar, Any, Dict

T = TypeVar("T")

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Loads a YAML file and returns a dictionary."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively converts a dictionary to a dataclass."""
    if not is_dataclass(cls):
        return data
    
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    
    for name, value in data.items():
        if name in field_types:
            target_type = field_types[name]
            if isinstance(value, dict) and is_dataclass(target_type):
                kwargs[name] = dict_to_dataclass(target_type, value)
            else:
                try:
                    if target_type in (int, float, str, bool) and value is not None:
                        kwargs[name] = target_type(value)
                    else:
                        kwargs[name] = value
                except (TypeError, ValueError):
                    kwargs[name] = value
                
    return cls(**kwargs)

def load_config_from_yaml(cls: Type[T], file_path: str) -> T:
    """Loads a YAML file directly into a dataclass."""
    data = load_yaml(file_path)
    return dict_to_dataclass(cls, data)
