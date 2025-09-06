from llm import call_llm

class LogInTrajReflexionAgent:
    def __init__(self):
        self.trigger_steps = [6, 9, 11, 13]

        with open('prompts/CASE 2A Prompts/task_base_few_shot.txt', 'r') as f:
            self.react_few_shot_prompt = f.read()
        with open('prompts/CASE 2A Prompts/in_traj_reflection_generation_prompt.txt', 'r') as f:
            self.reflexion_few_shot_prompt = f.read()

    def run_episode(self, env, session_id, instruction, max_steps=15, to_print=True, num_traces=1):
        """
        Run a single ReAct reasoning trace with in-trajectory reflection.
        """
        action = 'reset'
        Instruction = f"Instruction: {instruction}\n"
        st_memory = ""  # Short-term memory for in-trajectory reflections
        reflexions = []  # Store all generated reflexions

        # Build initial prompt
        initial_prompt = self.reflexion_few_shot_prompt + "\n" + Instruction + "[Search]\n"

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
                observation = observation

            # Add step to trajectory with additional metadata
            step_data = {
                'step': i, 
                'action': action, 
                'observation': observation.strip(),
                'reward': reward,
                'done': done
            }

            trajectory.append(step_data)
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

                reflection = self._generate_in_traj_reflection(instruction, st_memory)
                st_memory += f"REFLECTION: {reflection}\n\n"
                
                # Store the reflexion with step information
                reflexion_data = {
                    'step': i + 0.5,  # Use .5 to indicate this happens between steps
                    'type': 'reflexion',
                    'reflexion': reflection,
                    'trigger_step': i + 1
                }
                trajectory.append(reflexion_data)
                
                reflexions.append({
                    'step': i + 1,
                    'reflexion': reflection
                })

                if to_print:
                    print(f"\n--- Reflection for This Trajectory ---")
                    print(reflection)

            if i == 0:
                prompt_history = f"Observation: {observation}\n\nAction:"
            else:
                prompt_history += f" {action}\nObservation: {observation}\n\nAction:"

            # Construct the full prompt
            # full_prompt = initial_prompt
            # if st_memory:
            #     full_prompt += f"You have the following reflections from this trajectory:\n{st_memory}\n\n"
            # full_prompt += prompt_history[-(6000 - len(full_prompt)):]


            few_shot_example = self.react_few_shot_prompt + "\n"
            STM = f"Here is your trajectory and reflections so far:\n{st_memory}\n"
            Instruction = f"Instruction: {instruction}\n"

            full_prompt = few_shot_example + Instruction + "\n" + STM + "Action: "

            action = call_llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
            llm_calls += 1
            if not action:
                break

        if to_print and not done:
            print(f"Max steps reached. Ending episode with reward: {reward}")

        return reward, trajectory, llm_calls, reflexions

    def _format_trajectory(self, trajectory, instruction):
        """Formats a trajectory for reflection generation."""
        formatted = f"Instruction:\n{instruction}\n[Search]\n\n"
        for step in trajectory:
            # Skip reflexion entries when formatting trajectory for reflection
            if step.get('type') == 'reflexion':
                continue
            formatted += f"Action: {step['action']}\nObservation:\n{step['observation']}\n\n"
        return formatted.strip()

    def _generate_in_traj_reflection(self, instruction, st_memory):
        """
        Generates a reflection on the current trajectory to improve future actions.
        """
        reflection_instructions = "You are in the middle of completing your current WebShop task. Do not describe the environment itself—focus on analyzing your progress so far, including the strategy and path you have taken up to this point. From your journey so far within this trajectory, identify: \n Successful outcomes — steps and past reflections that moved you closer to completing the task that can help complete the task. \n Missteps or errors — actions that stalled progress, caused problems, or led to loops, and how you could have avoided them. \n If you have encountered similar problems before (such as invalid actions, loops, or unhelpful states), recall how you overcame them and apply those solutions now. \n If the problem is new or looped and you have no prior solution to draw from, reason about the current observations, available actions, and your understanding of the task to form a novel, logical plan that can overcome this obstacle. Think about alternative strategies, backtracking, or re-approaching the goal from a different angle if necessary. \n Your goal is to decide the most effective next action that will overcome the current obstacle and move you closer to completing the task. Keep the reflections concise and focused on actionable insights." 
        # + self.reflexion_few_shot_prompt + "\n"

        # if st_memory:
        #     reflection_prompt += f"You have already reflected on this task. Your previous reflections didn't work, so analyze what went wrong in these attempts and devise a different strategy to complete the task:\n{st_memory}\n\n"

        # formatted_trajectory = self._format_trajectory(trajectory, instruction)
        # reflection_prompt += f"{formatted_trajectory}\n\nInstruction: {instruction}\n\nReflection: "

        full_reflection_prompt = f"""{reflection_instructions}

        ---
        ## CURRENT TASK & TRAJECTORY
        Instruction: {instruction}
        {st_memory}

        ---
        ## REFLECTION
        """

        return call_llm(full_reflection_prompt, stop=["Action:"], num_traces=2)
