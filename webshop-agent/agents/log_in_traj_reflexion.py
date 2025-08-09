from llm import call_llm

class LogInTrajReflexionAgent:
    def __init__(self):
        self.lt_memory_str = ""
        self.trigger_steps = [6, 9, 11, 13]

        with open('prompts/react_few_shot.txt', 'r') as f:
            self.react_few_shot_prompt = f.read()
        with open('prompts/webshop_reflexion_few_shot.txt', 'r') as f:
            self.reflexion_few_shot_prompt = f.read()

    def run_episode(self, env, session_id, instruction, max_steps=15, to_print=True, num_traces=1):
        """
        Run a single ReAct reasoning trace with in-trajectory reflection.
        """
        action = 'reset'
        st_memory = ""  # Short-term memory for in-trajectory reflections

        # Build initial prompt
        initial_prompt = self.react_few_shot_prompt + "\n"
        if self.lt_memory_str:
            initial_prompt += f"You have the following long-term memories to help you:\n{self.lt_memory_str}\n\n"
        initial_prompt += f"Instruction: {instruction}\n[Search]\n"

        prompt_history = ''
        trajectory = []
        llm_calls = 0
        reward = 0
        done = False

        for i in range(max_steps):
            try:
                observation, reward, done = env.step(session_id, action)
            except AssertionError:
                observation, reward, done = 'Invalid action!', 0.0, False

            if action.startswith('think['):
                observation = 'OK.'

            trajectory.append({'step': i, 'action': action, 'observation': observation.strip()})
            if to_print:
                print(f"\n--- Step {i+1}/{max_steps} ---")
                print(f"Action: {action}")
                print(f"Observation:\n    " + "\n    ".join(observation.strip().split('\n')))

            if done:
                if to_print: print(f"Episode finished with reward: {reward}")
                break

            # In-trajectory reflexion
            if (i + 1) in self.trigger_steps:
                if to_print:
                    print(f"\n--- Step {i+1}/{max_steps}: Triggering In-Trajectory Reflexion ---")

                reflection = self._generate_in_traj_reflection(trajectory, instruction, st_memory)
                st_memory += f"- {reflection}\n"
                learning = self._summarize_learning(reflection)
                self.lt_memory_str += f"- {learning}\n"

                if to_print:
                    print(f"\n--- Reflection for This Trajectory ---")
                    print(reflection)
                    print(f"\n--- Learning for Long-Term Memory ---")
                    print(learning)

            if i == 0:
                prompt_history = f"Observation: {observation}\n\nAction:"
            else:
                prompt_history += f" {action}\nObservation: {observation}\n\nAction:"

            # Construct the full prompt
            full_prompt = initial_prompt
            if st_memory:
                full_prompt += f"You have the following reflections from this trajectory:\n{st_memory}\n\n"
            full_prompt += prompt_history[-(6000 - len(full_prompt)):]

            action = call_llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
            llm_calls += 1
            if not action:
                break

        # Final reflection for long-term memory
        if to_print:
            print(f"\n--- End of Trajectory: Generating Final Reflection for Long-Term Memory ---")

        final_reflection = self._generate_in_traj_reflection(trajectory, instruction, st_memory)
        final_learning = self._summarize_learning(final_reflection)
        self.lt_memory_str += f"- {final_learning}\n"

        if to_print:
            print(f"\n--- Final Reflection ---")
            print(final_reflection)
            print(f"\n--- Final Learning for Long-Term Memory ---")
            print(final_learning)

        if to_print and not done:
            print(f"Max steps reached. Ending episode with reward: {reward}")

        return reward, trajectory, llm_calls

    def _format_trajectory(self, trajectory, instruction):
        """Formats a trajectory for reflection generation."""
        formatted = f"Instruction:\n{instruction}\n[Search]\n\n"
        for step in trajectory:
            formatted += f"Action: {step['action']}\nObservation:\n{step['observation']}\n\n"
        return formatted.strip()

    def _generate_in_traj_reflection(self, trajectory, instruction, st_memory):
        """
        Generates a reflection on the current trajectory to improve future actions.
        """
        reflection_prompt = "You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You are in the middle of the task. Do not summarize your environment, but rather think about the strategy and path you have taken so far. Devise a concise, new plan of action that accounts for your progress and potential mistakes with reference to specific actions that you should take next. There are two examples below." + self.reflexion_few_shot_prompt + "\n"

        if st_memory:
            reflection_prompt += f"You have already reflected on this task. Your previous reflections didn't work, so analyze what went wrong in these attempts and devise a different strategy to complete the task:\n{st_memory}\n\n"

        formatted_trajectory = self._format_trajectory(trajectory, instruction)
        reflection_prompt += f"{formatted_trajectory}\n\nInstruction: {instruction}\n\nReflection: "

        return call_llm(reflection_prompt, stop=["Action:"], num_traces=2)

    def _summarize_learning(self, reflection):
        """
        Summarizes a reflection into a concise, one-sentence learning.
        """
        prompt = f"Summarize the following reflection into a concise, one-sentence learning that can be used to improve future performance:\n\nReflection: {reflection}\n\nLearning:"
        return call_llm(prompt, stop=['\n']).strip()
