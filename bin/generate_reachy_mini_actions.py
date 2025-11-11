#!/usr/bin/env python3
import json
from pathlib import Path

# Output to ../actions relative to this script
output_dir = Path(__file__).resolve().parent.parent / "actions"
output_dir.mkdir(parents=True, exist_ok=True)

# Category → Emotion → Movements and Sounds
ACTION_MAP = {
    "dance-reachy-dance": {
        "joyful": {
            "moves": ["groovy_sway_and_roll", "side_to_side_sway", "headbanger_combo", "interwoven_spirals"],
            "sounds": ["hf:cheerful1.wav", "hf:enthusiastic1.wav", "hf:success2.wav"]
        },
        "proud": {
            "moves": ["groovy_sway_and_roll", "jackson_square"],
            "sounds": ["hf:proud1.wav", "hf:proud2.wav", "hf:proud3.wav"]
        },
        "welcoming": {
            "moves": ["groovy_sway_and_roll", "side_to_side_sway"],
            "sounds": ["hf:welcoming1.wav", "hf:welcoming2.wav"]
        },
        "dance": {
            "moves": ["groovy_sway_and_roll", "headbanger_combo"],
            "sounds": ["hf:dance1.wav", "hf:dance2.wav", "hf:dance3.wav"]
        }
    },
    "deep-in-thought": {
        "curious": {
            "moves": ["head_tilt_roll", "uh_huh_tilt"],
            "sounds": ["hf:curious1.wav", "hf:inquiring1.wav", "hf:attentive1.wav"]
        },
        "understanding": {
            "moves": ["uh_huh_tilt", "pendulum_swing"],
            "sounds": ["hf:understanding1.wav", "hf:understanding2.wav"]
        },
        "thoughtful": {
            "moves": ["head_tilt_roll", "sharp_side_tilt"],
            "sounds": ["hf:thoughtful1.wav", "hf:thoughtful2.wav"]
        }
    },
    "surprise-surprise": {
        "confused": {
            "moves": ["neck_recoil", "side_peekaboo"],
            "sounds": ["hf:confused1.wav", "hf:oops1.wav", "hf:uncertain1.wav"]
        },
        "uncertain": {
            "moves": ["side_glance_flick", "neck_recoil"],
            "sounds": ["hf:uncertain1.wav", "hf:lost1.wav"]
        },
        "surprised": {
            "moves": ["neck_recoil", "sharp_side_tilt"],
            "sounds": ["hf:surprised1.wav", "hf:surprised2.wav"]
        },
        "shy": {
            "moves": ["side_glance_flick", "side_peekaboo"],
            "sounds": ["hf:shy1.wav", "hf:welcoming1.wav"]
        }
    },
    "lets-sit-and-think": {
        "serene": {
            "moves": ["side_to_side_sway", "pendulum_swing"],
            "sounds": ["hf:serenity1.wav", "hf:calming1.wav", "hf:relief1.wav"]
        },
        "sad": {
            "moves": ["simple_nod", "side_to_side_sway"],
            "sounds": ["hf:sad1.wav", "hf:sad2.wav", "hf:yes_sad1.wav"]
        },
        "tired": {
            "moves": ["pendulum_swing", "simple_nod"],
            "sounds": ["hf:tired1.wav", "hf:sleep1.wav", "hf:exhausted1.wav"]
        }
    },
    "grumpy-robot": {
        "irritated": {
            "moves": ["chicken_peck", "grid_snap"],
            "sounds": ["hf:irritated1.wav", "hf:irritated2.wav"]
        },
        "frustrated": {
            "moves": ["stumble_and_recover", "jackson_square"],
            "sounds": ["hf:frustrated1.wav", "hf:impatient1.wav", "hf:impatient2.wav"]
        },
        "furious": {
            "moves": ["grid_snap", "chicken_peck"],
            "sounds": ["hf:rage1.wav", "hf:furious1.wav"]
        }
    }
}

def build_action(name: str, moves: list[str], sounds: list[str]) -> dict:
    step = {
        "move": {"pick": {"items": moves}},
        "sound": {"pick": {"items": sounds}}
    }
    return {
        "name": name,
        "steps": [step, step]
    }

def main():
    for category, emotions in ACTION_MAP.items():
        for emotion, data in emotions.items():
            action_name = f"{category}_{emotion}"
            filename = f"{action_name}.json"
            action = build_action(action_name, data["moves"], data["sounds"])
            output_path = output_dir / filename
            with open(output_path, "w") as f:
                json.dump(action, f, indent=2)
            print(f"✅ Wrote: {output_path.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()
