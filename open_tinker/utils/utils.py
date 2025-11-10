import fire
import tomllib
from types import SimpleNamespace

# System prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def parse_toml_to_args(config_path: str):
    """将 TOML 文件转换为支持点式访问的对象"""
    with open(config_path, 'rb') as f:
        # Python >= 3.11 使用 tomllib
        # Python < 3.11 使用 tomli (需要: pip install tomli)
        import tomli
        config = tomli.load(f)
    
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_namespace(item) for item in d]
        else:
            return d
    
    return dict_to_namespace(config)


def main(config_path: str):
    flat_toml = parse_toml_to_args(config_path)
    for k, v in flat_toml.items():
        print(f"{k} = {v}")

if __name__ == "__main__":
    fire.Fire(main)
