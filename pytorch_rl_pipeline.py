import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Force CUDA deployment profile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.network(state)


def run_pytorch_pipeline():
    print("\n" + "=" * 60)
    print("[PYTORCH] Pretraining & Man-In-The-Loop RL Engine Initialization")
    print("=" * 60)
    print(f"Target Hardware Backend: {device} ({torch.cuda.get_device_name(0)})")

    # Dimensions for tracking configurations
    state_dim = 8
    action_dim = 2

    model = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    benchmark_logs = []
    total_episodes = 30

    print("\n[1/2] Initiating Autonomous Pretraining Phase...")
    for episode in range(1, total_episodes + 1):
        start_time = time.time()

        # Simulate environment state vector batch
        states = torch.randn(32, state_dim, device=device)
        action_probs = model(states)

        # Calculate structural mock policy loss
        dummy_targets = torch.randint(0, action_dim, (32,), device=device)
        loss = nn.CrossEntropyLoss()(action_probs, dummy_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_latency = (time.time() - start_time) * 1000  # ms
        allocated_vram = torch.cuda.memory_allocated(0) / (1024**2)
        reserved_vram = torch.cuda.memory_reserved(0) / (1024**2)

        # [2/2] Man-in-the-Loop Intervention (Simulated or Manual Prompt)
        human_reward_modifier = 1.0
        if episode % 15 == 0:
            print(
                f"\n[!!!] MAN-IN-THE-LOOP INTERVENTION TRIGGERED AT EPISODE {episode}"
            )
            print(f"Current VRAM Allocation: {allocated_vram:.2f} MB")
            # For automation during unattended scripts, fallback to a defaults pattern
            user_input = input(
                "Enter reward scalar adjustments (-1.0 to 2.0) or press Enter to accept baseline [1.0]: "
            ).strip()
            if user_input:
                try:
                    human_reward_modifier = float(user_input)
                except ValueError:
                    pass
            print(
                f">> Applied Human Heuristic Reward Scalar: {human_reward_modifier}\n"
            )

        benchmark_logs.append(
            {
                "framework": "PyTorch",
                "episode": episode,
                "loss": round(float(loss.item()), 4),
                "step_time_ms": round(step_latency, 2),
                "allocated_vram_mb": round(allocated_vram, 2),
                "reserved_vram_mb": round(reserved_vram, 2),
                "human_modifier": human_reward_modifier,
            }
        )

        if episode % 5 == 0:
            print(
                f"  Episode {episode:02d}/{total_episodes} | Base Loss: {loss.item():.4f} | Sync Time: {step_latency:.1f}ms"
            )

    with open("pytorch_benchmark.json", "w") as f:
        json.dump(benchmark_logs, f, indent=4)
    print(
        "\n[+] PyTorch Pipeline Execution complete. Metrics logged to 'pytorch_benchmark.json'."
    )


if __name__ == "__main__":
    run_pytorch_pipeline()
