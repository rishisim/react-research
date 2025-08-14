import os
from llm import call_llm

class LogInTrajReflexionAgent:
    def __init__(self):
        self.ltm_file_path = 'lt_memory.txt'
        self.lt_memory_str = self._load_lt_memory()
        self.trigger_steps = [6, 9, 11, 13]
        self.lt_update_steps = [step + 1 for step in self.trigger_steps]  # [7, 10, 12, 14]

        with open('prompts/react_few_shot.txt', 'r') as f:
            self.react_few_shot_prompt = f.read()
        with open('prompts/webshop_LT_reflexion_few_shot.txt', 'r') as f:
            self.reflexion_few_shot_prompt = f.read()

    def _load_lt_memory(self):
        """Load long-term memory from file, or create default if file doesn't exist."""
        if os.path.exists(self.ltm_file_path):
            with open(self.ltm_file_path, 'r') as f:
                return f.read()
        else:
            default_memory = "| Activity | Problem | Solution | Outcome| \n| Empty | Empty | Empty | Empty |\n"
            self._save_lt_memory(default_memory)
            return default_memory

    def _save_lt_memory(self, memory_str):
        """Save long-term memory to file."""
        with open(self.ltm_file_path, 'w') as f:
            f.write(memory_str)

    def run_episode(self, env, session_id, instruction, max_steps=15, to_print=True, num_traces=1):
        """
        Run a single ReAct reasoning trace with in-trajectory reflection.
        """
        action = 'reset'
        st_memory = ""  # Short-term memory combining trajectory and reflections

        # Build initial prompt
        initial_prompt = self.reflexion_few_shot_prompt + "\n"
        if self.lt_memory_str:
            initial_prompt += f"You have the following long-term memories to help you:\n{self.lt_memory_str}\n\n"
        initial_prompt += f"Instruction: {instruction}\n[Search]\n"

        prompt_history = ''
        trajectory = []  # Structured trajectory with all information
        llm_calls = 0
        reward = 0
        done = False

        for i in range(max_steps):
            try:
                observation, reward, done = env.step(session_id, action)
            except AssertionError:
                observation, reward, done = 'Invalid action!', 0.0, False

            if action.startswith('think['):
                observation = observation

            # Add this step to structured trajectory
            step_data = {
                'step': i, 
                'action': action, 
                'observation': observation.strip(),
                'reward': reward,
                'done': done
            }
            trajectory.append(step_data)
            
            # Add this step to ST memory
            st_memory += f"Action: {action}\nObservation: {observation.strip()}\n\n"
            
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

                reflection = self._generate_in_traj_reflection(instruction, st_memory, self.lt_memory_str)
                
                # Append reflection to ST memory
                st_memory += f"REFLECTION: {reflection}\n\n"
                
                # Add reflexion to structured trajectory
                reflexion_data = {
                    'step': i + 0.5,  # Use .5 to indicate this happens between steps
                    'type': 'reflexion',
                    'reflexion': reflection,
                    'trigger_step': i + 1
                }
                trajectory.append(reflexion_data)

                if to_print:
                    print(f"\n--- Reflection for This Trajectory ---")
                    print(reflection)

            # LT memory update (one step after each reflection)
            if (i + 1) in self.lt_update_steps:
                if to_print:
                    print(f"\n--- Step {i+1}/{max_steps}: Updating Long-Term Memory ---")

                learning = self.update_Long_Term_Memory(st_memory)
                self.lt_memory_str = f"{learning}\n"
                self._save_lt_memory(self.lt_memory_str)
            
                # Add LTM update to structured trajectory
                ltm_data = {
                    'step': i + 0.6,  # Use .6 to indicate this happens after reflexion
                    'type': 'ltm_update',
                    'ltm_update': learning,
                    'update_step': i + 1
                }
                trajectory.append(ltm_data)
                
                if to_print:
                    print(f"\n--- Long-Term Memory ---")
                    print(self.lt_memory_str)

            

            if i == 0:
                prompt_history = f"Observation: {observation}\n\nAction:"
            else:
                prompt_history += f" {action}\nObservation: {observation}\n\nAction:"

            # Construct the full prompt
            full_prompt = initial_prompt
            if st_memory:
                full_prompt += f"Here is your trajectory and reflections so far:\n{st_memory}\n"
            full_prompt += prompt_history[-(6000 - len(full_prompt)):]

            action = call_llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
            llm_calls += 1
            if not action:
                break

        # Final reflection for long-term memory
        if to_print:
            print(f"\n--- End of Trajectory: Generating Final Reflection for Long-Term Memory ---")

        final_reflection = self._generate_in_traj_reflection(instruction, st_memory, self.lt_memory_str)
        
        # Append final reflection to ST memory
        st_memory += f"FINAL REFLECTION: {final_reflection}\n\n"
        
        # Add final reflexion to structured trajectory
        final_reflexion_data = {
            'step': 'final_reflexion',
            'type': 'final_reflexion',
            'reflexion': final_reflection
        }
        trajectory.append(final_reflexion_data)
        
        final_learning = self.update_Long_Term_Memory(st_memory)
        self.lt_memory_str = f"{final_learning}\n"
        self._save_lt_memory(self.lt_memory_str)
        
        # Add final LTM update to structured trajectory
        final_ltm_data = {
            'step': 'final_ltm_update',
            'type': 'final_ltm_update',
            'ltm_update': final_learning
        }
        trajectory.append(final_ltm_data)

        if to_print:
            print(f"\n--- Final Reflection ---")
            print(final_reflection)
            print(f"\n--- Final Long-Term Memory ---")
            print(final_learning)

        if to_print and not done:
            print(f"Max steps reached. Ending episode with reward: {reward}")

        return reward, trajectory, llm_calls, st_memory

    def _format_trajectory(self, trajectory, instruction):
        """Formats a trajectory for reflection generation."""
        formatted = f"Instruction:\n{instruction}\n[Search]\n\n"
        for step in trajectory:
            formatted += f"Action: {step['action']}\nObservation:\n{step['observation']}\n\n"
        return formatted.strip()

    def _generate_in_traj_reflection(self, instruction, st_memory, lt_memory_str):
        """
        Generates a reflection on the current trajectory to improve future actions.
        """
        reflection_prompt = "You are in the middle of completing your current WebShop task. Do not describe the environment itself—focus on analyzing your progress so far, including the strategy and path you have taken up to this point. If you are progressing in the task, continue. Otherwise, to solve the issue: \n 1. Consult to your Long-Term Memory (LTM) for past reflections and learnings. \n 2. If you have no relevant issues in LTM or the solutions related to problem are unhelpful, reflect on your current trajectory of actions and observations to identify what might have went wrong and propose a novel solution. Novel solutions are encouraged. \n Your entire reflection should be short MAX 100 words.\n\n"
        # + self.reflexion_few_shot_prompt + "\n"

        # if st_memory:
        #     reflection_prompt += f"You have already reflected on this task. Your previous reflections didn't work, so analyze what went wrong in these attempts and devise a different strategy to complete the task:\n{st_memory}\n\n"

        # formatted_trajectory = self._format_trajectory(trajectory, instruction)
        reflection_prompt += f"Instruction: {instruction}\n\n{st_memory}\n\nLT Memory:{lt_memory_str}\n\nReflection: "

        return call_llm(reflection_prompt, stop=["Action:"], num_traces=2)

    def update_Long_Term_Memory(self, trajectory):
        """
        Creates new LTM str.
        """
        prompt = f"{trajectory}\nBased on the trajectory and the recent reflection, update Long Term Memory.\nGoal: Capture one recent learning to guide the NEXT ACTION.\nDefinition: 'Latest Observation' = the final block starting with 'Observation:' in the provided trajectory.\nContent to extract (single row, keywords only): 1. Activity (<5 words) 2. Problem (<5 words) 3. Solution (<5 words, high-level idea) 4. Outcome (Helpful/Unhelpful) 5. Next step (atomic)\nRules:\n1. If Outcome=Unhelpful for any pattern in this session, do NOT repeat solution; develop a novel solution.\n2. Next step must be one of the exact control labels present in the Latest Observation; copy the label verbatim and pair it with an action verb (e.g., click[Back to Search]). Never reference controls from earlier Observations.\n3. Keep fields terse (MAX 5 words; no articles, punctuation, emojis).\n4. Format the entire updated LT Memory as a table with headers: | Activity | Problem | Solution | Outcome | Next step |\n5. Final self-check (silent): verify the chosen Next step's label appears verbatim in the Latest Observation; if not, replace it with any label that does appear there.\nExisting LT Memory (append/update rows by loosely matching Activity+Problem):\n{self.lt_memory_str}"
        return call_llm(prompt, stop=[], num_traces=2)
