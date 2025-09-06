import os
from llm import call_llm
import textwrap
import pandas as pd

class LogInTrajReflexionAgent:
    def __init__(self):
        self.LTM_COLUMNS = ['Activity', 'Problem', 'Solution', 'Outcome']
        self.ltm_file_path = 'lt_memory.csv'
        self.lt_memory_df = self._load_lt_memory()
        self.trigger_steps = [6, 9, 11, 13]
        self.lt_update_steps = [step + 1 for step in self.trigger_steps]  # [7, 10, 12, 14]

        with open('prompts/CASE 2B Prompts/task_base_few_shot.txt', 'r') as f:
            self.task_base_few_shot_example = f.read()
        with open('prompts/CASE 2B Prompts/in_traj_reflection_generation_prompt.txt', 'r') as f:
            self.reflection_one_shot_example = f.read()
        
        self.LTM_context_prompt = textwrap.dedent("""
            Learnings from your past experiences are saved in your Long-Term Memory (LTM). Each memory entry has the following structure:
            * Activity: The overall task you were trying to accomplish
            * Problem: The specific obstacle you encountered
            * Reasoning: Your thought process for choosing a solution
            * Solution: The action or sequence of actions you took
            * Outcome: The result, labeled as 'Helpful' or 'Unhelpful'.

            Helpful outcomes are those that moved you closer to completing the task. Unhelpful outcomes are those that resulted in an error or no progress. 
                                                  
            ---
            Long Term Memory (LTM):""")

    def _load_lt_memory(self):
        try:
            return pd.read_csv(self.ltm_file_path)
        except FileNotFoundError:
            df = pd.DataFrame([['Empty'] * len(self.LTM_COLUMNS)], columns=self.LTM_COLUMNS)
            df.to_csv(self.ltm_file_path, index=False)
            return df

    def _save_lt_memory(self, df):
        df.to_csv(self.ltm_file_path, index=False)

    def _formatted_ltm_for_prompt(self):
        """Convert DataFrame to a markdown table string for LLM prompts."""
        if self.lt_memory_df.empty:
            # Return only the header if the DataFrame is empty
            return f"| {' | '.join(self.LTM_COLUMNS)} |"
        return self.lt_memory_df.to_markdown(index=False)

    def _format_trajectory(self, trajectory, instruction):
        """Formats a trajectory for reflection generation."""
        formatted = f"Instruction:\n{instruction}\n[Search]\n\n"
        for step in trajectory:
            formatted += f"Action: {step['action']}\nObservation:\n{step['observation']}\n\n"
        return formatted.strip()

    def _generate_in_traj_reflection(self, instruction, st_memory, lt_memory_str):
        # reflection_instructions = "You are in the middle of completing your current WebShop task. Do not describe the environment itself—focus on analyzing your progress so far, including the strategy and path you have taken up to this point. If you are progressing in the task, continue. Otherwise, to solve the issue: \n 1. Consult to your Long-Term Memory (LTM) for past reflections and learnings. \n 2. If you have no relevant issues in LTM or the solutions related to problem are unhelpful, reflect on your current trajectory of actions and observations to identify what might have went wrong and propose a novel solution. Novel solutions are encouraged. \n Your entire reflection should be short MAX 100 words.\n\n"

        # reflection_instructions = textwrap.dedent("""
        #     You are in the middle of a WebShop task. Reflect on your strategy.
                                                  
        #     Analyze your last few actions and observations. Are you stuck? A 'stuck' state could be:
        #     * Seeing the exact same page content after multiple actions.
        #     * Encountering repeated errors.
        #     * Failing to find the target item after a reasonable number of steps.

        #     If you are making clear progress, briefly state your next intended action and continue.

        #     If you are stuck, you must change your strategy:
        #     1.  Consult LTM: Search for past activities with similar problems. If a Helpful Solution exists, apply it.
        #     2.  Innovate: If LTM is unhelpful or has no similar problems, do not repeat failing actions. Analyze why your current path failed and propose a novel, different action or approach to try next. Novelty is encouraged.
                                                  
        #     Your next action must be based only on the 'Latest Observation'. Identify an interactable element (like a button, link, or input field) from the observation and copy its label verbatim. Pair it with an action verb (e.g., click[Search Results]).  

        #     Keep your reflection under 100 words. \n""")

        # reflection_instructions = textwrap.dedent("""
        #     You are in the middle of a WebShop task. Reflect on your strategy.
                                      
        #     Analyze your last few actions and observations. Are you stuck? A 'stuck' state could be:
        #     * Seeing the exact same page content after multiple actions.
        #     * Encountering repeated errors.
        #     * Failing to find the target item after a reasonable number of steps.

        #     If you are making clear progress, briefly state your next intended action and continue.

        #     If you are stuck, you must change your strategy:
        #     1.Consult LTM: Search for past activities with "similar" problems. 
        #     2.If a Helpful Solution exists from a similar past activity, apply it. 
        #     3.Do not repeat solutions with "Unhelpful" outcomes. This includes solutions that are functionally similar, such as 'broaden the search query', 'look for a broader item' etc.
        #     3.Innovate: Analyze why your current path failed and propose a novel, different action or approach to try next. Novelty is encouraged.
                                                
        #     Your next action must be based only on the 'Latest Observation'. Identify an interactable element (like a button, link, or input field) from the observation and copy its label verbatim. Pair it with an action verb (e.g., click[Search Results]).  

        #     Keep your reflection under 100 words. \n""")

        reflection_instructions = textwrap.dedent("""
            You are in the middle of a WebShop task. Reflect on your strategy.
                              
            Analyze your last few actions and observations. Are you stuck? A 'stuck' state could be:
            * Seeing the exact same page content after multiple actions.
            * Encountering repeated errors like "Invalid action!".
            * Failing to find the target item after a reasonable number of steps.

            If you are making clear progress, briefly state your next intended action and continue.

            If you are stuck, you must change your strategy:
            
            **Special Heuristic: Handling Invalid Searches**
            If your last action was `search[...]` and the observation was "Invalid action!", this is a critical error. It almost certainly means you are not on a page that has a search bar.
            1.  **Your immediate priority is to navigate back.** Do NOT try another `search` action.
            2.  **Scan the 'Latest Observation' for navigation keywords** like "Back", "Return", "Search", or "Home".
            3.  Your next action must be to `click` the element that allows you to return to a previous page or the search page.
            
            **General Strategy Change:**
            For all other "stuck" situations:
            1.  **Consult LTM:** Look for past activities with similar problems. If a Helpful Solution exists, apply it.
            2.  **Innovate:** If LTM is unhelpful, analyze why your current path failed and propose a novel, different action.
                                                
            Your next action must be based only on the 'Latest Observation'.

            Keep your reflection under 100 words and end with next action. \n""")

        LTM_Context_Prompt = self.LTM_context_prompt

        LTM = self._formatted_ltm_for_prompt()
        STM = st_memory
        reflection_one_shot_example = self.reflection_one_shot_example

        # Combine using clear headings
        full_reflection_prompt = f"""{reflection_instructions}

        ---
        ## REFLECTION EXAMPLE
        {reflection_one_shot_example}

        ---
        ## LONG-TERM MEMORY (LTM) CONTEXT
        {LTM_Context_Prompt}
        {LTM}

        ---
        ## CURRENT TASK & SHORT-TERM MEMORY (STM)
        Instruction: {instruction}
        {STM}

        ---
        ## REFLECTION
        """

        return call_llm(full_reflection_prompt, stop=["\n\n"], num_traces=2)

    def update_long_term_memory(self, st_memory):
        prompt = textwrap.dedent(f"""Analyze the following trajectory and reflection to extract a key learning.
        **Trajectory:**
        {st_memory}

        **Instructions:**
        Summarize the learning as a single row with four categories separated by a pipe '|'. Each category should be a concise summary phrase of no more than five words.

        1.  **Activity:** The overall task being attempted.
        2.  **Problem:** The specific obstacle encountered.
        3.  **Solution:** The action taken to overcome the obstacle.
        4.  **Outcome:** The result, labeled as either 'Helpful' or 'Unhelpful'. A 'Helpful' outcome moved the task forward; 'Unhelpful' resulted in an error or no progress.

        **Example Output:**
        "| Find and add camera | Could not find search bar | Clicked magnifying glass icon | Unhelpful |"

        **Your Extraction:**
        """)

        learning_text = call_llm(prompt, stop=[], num_traces=2)
        
        # Parse the pipe-separated response
        parts = [part.strip() for part in learning_text.strip().split('|') if part.strip()]
        
        if len(parts) >= 4:
            # 1. Remove the default "Empty" row if it's the only one present
            if not self.lt_memory_df.empty and self.lt_memory_df.iloc[0]['Activity'] == 'Empty':
                self.lt_memory_df = self.lt_memory_df.iloc[1:].reset_index(drop=True)

            # 2. Append the new learning
            new_row = dict(zip(self.LTM_COLUMNS, parts[:4]))
            self.lt_memory_df.loc[len(self.lt_memory_df)] = new_row
            
            # 3. Save memory
            self._save_lt_memory(self.lt_memory_df)
        
        return learning_text

    def run_episode(self, env, session_id, instruction, max_steps=15, to_print=True, num_traces=1):
        """
        Run a single ReAct reasoning trace with in-trajectory reflection.
        """
        action = 'reset'

        # Build initial prompt
        LTM_Context_Prompt = self.LTM_context_prompt
        
        LTM = self._formatted_ltm_for_prompt()
        Instruction = f"Instruction: {instruction}\n"
        st_memory = "ST Memory"  # Short-term memory combining trajectory and reflections

        initial_prompt = self.task_base_few_shot_example + "\n" + LTM_Context_Prompt + LTM + "\n" + Instruction + "[Search]\n"

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
                

            full_prompt_reflection_update = False
            # In-trajectory reflexion
            if (i + 1) in self.trigger_steps:
                full_prompt_reflection_update = True

                if to_print:
                    print(f"\n--- Step {i+1}/{max_steps}: Triggering In-Trajectory Reflexion ---")

                reflection = self._generate_in_traj_reflection(instruction, st_memory, self._formatted_ltm_for_prompt())
                
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

                learning = self.update_long_term_memory(st_memory)
                learning = self._formatted_ltm_for_prompt()
                
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
                    print(self._formatted_ltm_for_prompt())

            

            if i == 0:
                prompt_history = f"Observation: {observation}\n\nAction:"
            else:
                prompt_history += f" {action}\nObservation: {observation}\n\nAction:"

            # Construct the full prompt
            few_shot_example = self.task_base_few_shot_example
            LTM_Context_Prompt = self.LTM_context_prompt
            LTM = self._formatted_ltm_for_prompt()
            STM = f"Here is your trajectory and reflections so far. If most recent is a reflection, next step must be an action:\n{st_memory}\n"

            # Build the complete prompt with task examples, LTM, instruction, STM, and current interaction
            full_prompt = few_shot_example + "\n" + LTM_Context_Prompt + LTM + "\n" + Instruction + "\n" + STM + "Action: "

            # if full_prompt_reflection_update:
            #     full_prompt = f"""{few_shot_example}
            #         {LTM_Context_Prompt}
            #         {LTM}
            #         {Instruction}

            #         --- CURRENT TRAJECTORY ---
            #         {STM}

            #         --- YOUR ANALYSIS AND PLAN ---
            #         Based on your last reflection, you have formed the following plan:
            #         "{reflection}"

            #         Given the learnings from LTM and explicit plan, what is the single next action you must take?
            #         Action: """
            # else:
            #     full_prompt = f"""{few_shot_example}
            #         {LTM_Context_Prompt}
            #         {LTM}
            #         {Instruction}

            #         --- PREVIOUS TRAJECTORY ---
            #         {st_memory}

            #         Given learnings from LTM and this trajectory, what is the single next action you must take?
            #         Action: """

            action = call_llm(full_prompt, stop=['\n'], num_traces=num_traces).strip()
            
            llm_calls += 1
            if not action:
                break

        # Final reflection for long-term memory
        if to_print:
            print(f"\n--- End of Trajectory: Generating Final Reflection for Long-Term Memory ---")

        final_reflection = self._generate_in_traj_reflection(instruction, st_memory, self._formatted_ltm_for_prompt())
        
        # Append final reflection to ST memory
        st_memory += f"FINAL REFLECTION: {final_reflection}\n\n"
        
        # Add final reflexion to structured trajectory
        final_reflexion_data = {
            'step': 'final_reflexion',
            'type': 'final_reflexion',
            'reflexion': final_reflection
        }
        trajectory.append(final_reflexion_data)
        
        final_learning = self.update_long_term_memory(st_memory)
        final_learning = self._formatted_ltm_for_prompt()
        
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
