"""PPO buffer for slot-level experience."""


class SlotBuffer:
    """Slot-level PPO Buffer for storing experience."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Clear all stored experience."""
        self.slot_states = []
        self.slot_actions = []
        self.slot_logprobs = []
        self.slot_values = []
        self.slot_rewards = []
        self.slot_dones = []
        self.slot_durations = []
        self.slot_static_infos = []  # Static topology info (shared reference per problem)
    
    def add(self, state_info, actions, logprob_slot, value, reward, done, static_info=None, duration=1.0):
        """Add a slot transition to the buffer.
        
        Args:
            state_info: Dynamic per-slot state
            actions: Micro-actions taken in this slot
            logprob_slot: Log probability of the slot
            value: Value estimate
            reward: Reward for this slot
            done: Whether episode ended
            static_info: Static topology info (shared reference, not copied)
        """
        self.slot_states.append(state_info)
        self.slot_actions.append(actions)
        self.slot_logprobs.append(logprob_slot)
        self.slot_values.append(value)
        self.slot_rewards.append(reward)
        self.slot_durations.append(duration)
        self.slot_dones.append(done)
        self.slot_static_infos.append(static_info)
    
    def __len__(self):
        return len(self.slot_rewards)
