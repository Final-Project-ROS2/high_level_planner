#!/usr/bin/env python3
"""
ros2_high_level_agent_with_vision.py

High-level ROS2 LLM agent with integrated vision tools as LangChain @tool wrappers.

Vision services exposed to the agent:
- /vision/classify_all      -> std_srvs.srv.Trigger
- /vision/understand_scene  -> custom_interfaces.srv.UnderstandScene

"""
import os
import threading
import time
from typing import List, Dict, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse

from std_srvs.srv import SetBool, Trigger
from std_msgs.msg import String
from geometry_msgs.msg import Pose

# Action used for inter-level communication (re-using your Prompt action)
from custom_interfaces.action import Prompt

# Vision service types (assumes these exist in your workspace)
from custom_interfaces.srv import UnderstandScene

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

from dotenv import load_dotenv

ENV_PATH = '/home/group11/final_project_ws/src/high_level_planner/.env'
load_dotenv(dotenv_path=ENV_PATH)


class Ros2HighLevelAgentNode(Node):
    """
    High level planner node:
     - subscribes to /transcript (natural language instructions)
     - uses an LLM-based tool-calling agent to break instructions into ordered steps
     - for each resulting step, it sends the step as a goal to the medium-level action server /medium_level (Prompt action)
     - exposes an action server prompt_high_level mirroring the medium-level style
    """

    def __init__(self):
        super().__init__("ros2_high_level_agent")
        self.get_logger().info("Initializing Ros2 High-Level Agent Node...")

        self.declare_parameter("real_hardware", False)
        self.real_hardware: bool = self.get_parameter("real_hardware").get_parameter_value().bool_value

        # LLM init
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.get_logger().warn("No LLM API key found in environment variables GEMINI_API_KEY.")

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.0)

        # Subscribe to transcript topic (MUST)
        self.transcript_sub = self.create_subscription(String, "/transcript", self.transcript_callback, 10)
        self._last_transcript_lock = threading.Lock()
        self._last_transcript: Optional[str] = None

        # Action client to medium-level planner (send one step at a time)
        self.medium_level_client = ActionClient(self, Prompt, "/medium_level")

        # Vision service clients - real types from your specification
        self.vision_classify_all_client = self.create_client(Trigger, "/vision/classify_all")
        self.vision_understand_scene_client = self.create_client(UnderstandScene, "/vision/understand_scene")

        # Track tools called (for feedback)
        self._tools_called: List[str] = []
        self._tools_called_lock = threading.Lock()

        # Initialize LangChain tools (vision wrappers, medium-level submitter, etc.)
        self.tools = self._initialize_tools()

        # Create LangChain agent similar to medium-level
        self.agent_executor = self._create_agent_executor()

        # Action server to accept high-level Prompt requests
        self._action_server = ActionServer(
            self,
            Prompt,
            "prompt_high_level",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.response_pub = self.create_publisher(String, "/response", 10)

        self.get_logger().info("Ros2 High-Level Agent Node ready (listening /transcript, Prompt action server running).")

    # -----------------------
    # Transcript handling
    # -----------------------
    def transcript_callback(self, msg: String):
        """
        Called whenever a new natural-language instruction arrives on /transcript.
        We store the last transcript and start planning in a background thread.
        """
        text = msg.data.strip()
        if not text:
            return

        with self._last_transcript_lock:
            self._last_transcript = text

        self.get_logger().info(f"Received transcript: {text}")

        # Run planning asynchronously (don't block the subscriber thread)
        plan_thread = threading.Thread(target=self._plan_and_dispatch_from_transcript, args=(text,), daemon=True)
        plan_thread.start()

    def _plan_and_dispatch_from_transcript(self, instruction_text: str):
        """
        Use agent to break down instruction_text into steps and dispatch them to /medium_level
        """
        # reset tools called
        with self._tools_called_lock:
            self._tools_called = []

        try:
            self.get_logger().info("High-level agent: breaking instruction into ordered steps...")
            self.response_pub.publish(String(data="Hey there! I'm thinking about how to handle your request..."))

            agent_resp = self.agent_executor.invoke({"input": instruction_text})
            # agent_resp commonly has "output" key (LangChain pattern)
            final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)
            self.get_logger().info(f"Agent final text:\n{final_text}")
            # Publish LLM response to /response
            self.response_pub.publish(String(data=f"Alright! Here's what I plan to do: {final_text}"))


            # The LLM should produce ordered steps. We'll attempt to parse them.
            steps = self._parse_steps_from_text(final_text)
            if not steps:
                self.get_logger().warn("No steps parsed from LLM response. Aborting dispatch.")
                self.response_pub.publish(String(data="Hmm... I couldn’t figure out any clear steps. Could you try rephrasing that?"))
                return

            self.get_logger().info(f"Parsed {len(steps)} step(s). Dispatching to /medium_level one-by-one...")
            self.response_pub.publish(String(data=f"I’ve got {len(steps)} steps to do. Let’s get started!"))

            for i, step in enumerate(steps, start=1):
                start_msg = f"Okay! Starting step {i}: {step}"
                self.response_pub.publish(String(data=start_msg))

                self.get_logger().info(f"Sending step {i}/{len(steps)} to medium_level: {step}")
                result = self.send_step_to_medium(step)

                if result is None:
                    fail_msg = f"Oops! I couldn’t complete step {i}: {step}. I’ll stop here for now."
                    self.response_pub.publish(String(data=fail_msg))
                    self.get_logger().error(f"Failed to send step {i}. Aborting remaining steps.")
                    break
                else:
                    if result.success:
                        done_msg = f"Nice! Step {i} is all done. Here’s what I got: {result.final_response}"
                    else:
                        done_msg = f"Alright, I tried step {i} but it didn’t quite work out. The system said: {result.final_response}"
                    self.response_pub.publish(String(data=done_msg))
                    self.get_logger().info(f"Step {i} finished with success={result.success}. Response: {result.final_response}")

            self.get_logger().info("Dispatch complete.")
            self.response_pub.publish(String(data="All steps completed! Great teamwork!"))

        except Exception as e:
            self.get_logger().error(f"Error during planning/dispatch: {e}")
            self.response_pub.publish(String(data="Uh oh, something went wrong while I was planning that."))

    def _parse_steps_from_text(self, text: str) -> List[str]:
        """
        Very simple step parser:
        - Look for lines starting with numbers (1., 1), '-', or 'Step X:' and collect them
        - Otherwise, split by newline and treat each line as a candidate step, filtering short lines.
        You can replace this with a more robust parser if needed.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        steps = []
        for ln in lines:
            # numbered or dashed lines
            if ln[0].isdigit() or ln.startswith("-") or ln.lower().startswith("step"):
                # remove leading numbering/dash
                cleaned = ln.lstrip("-0123456789. ").strip()
                if len(cleaned) > 3:
                    steps.append(cleaned)
            else:
                # if line contains verbs, consider it a step (heuristic)
                if any(ln.lower().startswith(v) for v in ("move", "pick", "place", "approach", "grasp", "segment", "detect", "classify", "scan", "align", "rotate")):
                    steps.append(ln)
        # fallback: if nothing found, try sentence-splitting
        if not steps:
            for part in text.split(". "):
                p = part.strip()
                if len(p) > 5:
                    steps.append(p)
        # final cleanup: unique & trimmed
        final_steps = []
        for s in steps:
            s = s.strip()
            if s and s not in final_steps:
                final_steps.append(s)
        return final_steps

    # -----------------------
    # Tools (LangChain wrappers)
    # -----------------------
    def _initialize_tools(self) -> List[BaseTool]:
        tools: List[BaseTool] = []

        # ---------------- Vision tools ----------------

        @tool
        def classify_all() -> str:
            """
            Trigger /vision/classify_all (std_srvs/Trigger) to classify entire frame or all detections.
            """
            tool_name = "classify_all"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_classify_all_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/classify_all unavailable"
                req = Trigger.Request()
                future = self.vision_classify_all_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/classify_all"
                return f"classify_all: success={resp.success}, message={resp.message}"
            except Exception as e:
                return f"ERROR in classify_all: {e}"

        tools.append(classify_all)

        @tool
        def understand_scene() -> str:
            """
            Call /vision/understand_scene which returns a SceneUnderstanding message.
            We extract a short natural-language summary and a few stats for the LLM.
            """
            tool_name = "understand_scene"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_understand_scene_client.wait_for_service(timeout_sec=5.0):
                    return "Service /vision/understand_scene unavailable"
                req = UnderstandScene.Request()
                future = self.vision_understand_scene_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)
                resp = future.result()
                if resp is None:
                    return "No response from /vision/understand_scene"
                if not resp.success:
                    return f"understand_scene failed: {resp.error_message or 'unknown'}"
                summary = getattr(resp.scene, "scene_description", None)
                if summary:
                    return f"scene_summary: {summary}"
                total_objects = getattr(resp.scene, "total_objects", None)
                labels = getattr(resp.scene, "object_labels", None)
                return f"scene_summary: total_objects={total_objects}, labels={labels}"
            except Exception as e:
                return f"ERROR in understand_scene: {e}"

        tools.append(understand_scene)

        # ------------- Medium-level dispatch tool -------------
        @tool
        def send_to_medium_level(step_text: str, wait_for_result: bool = True) -> str:
            """
            Send a single textual step to the medium-level planner (/medium_level action server, Prompt action).
            If wait_for_result is True, we wait for the medium-level response and return a short summary.
            """
            tool_name = "send_to_medium_level"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                if not self.medium_level_client.wait_for_server(timeout_sec=5.0):
                    return "Medium-level action server /medium_level unavailable"
                goal = Prompt.Goal()
                goal.prompt = step_text
                send_future = self.medium_level_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, send_future)
                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    return "Medium-level goal rejected"

                if wait_for_result:
                    result_future = goal_handle.get_result_async()
                    rclpy.spin_until_future_complete(self, result_future)
                    result = result_future.result().result
                    return f"medium_level result: success={result.success}, response={result.final_response}"
                else:
                    return "Sent step to medium_level (not waiting for result)."
            except Exception as e:
                return f"ERROR in send_to_medium_level: {e}"

        tools.append(send_to_medium_level)

        return tools

    # -----------------------
    # Create agent executor
    # -----------------------
    def _create_agent_executor(self) -> AgentExecutor:
        system_message = (
            "You are a High-Level ROS2 planning assistant. You have access to tools that query vision "
            "capabilities (classify_all, understand_scene) "
            "and a tool send_to_medium_level which sends a single step to the medium-level planner (/medium_level). "
            "Your job: given a natural-language instruction, **EITHER** produce a short ordered list of actionable steps "
            "that a medium-level planner can execute. Keep steps concise, unambiguous and in the form "
            "'Action: <verb> <object/pose/params>' **OR** complete the instruction yourself. "
            "If you are able to complete the instruction directly without calling the medium-level planner, do so and DO NOT CALL the medium-level planner. "
            "When appropriate you may call vision tools to inspect and understand the scene. "
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=12)

    # -----------------------
    # Action server callbacks (high-level)
    # -----------------------
    def goal_callback(self, goal_request) -> GoalResponse:
        self.get_logger().info(f"[high-level action] Received goal: {getattr(goal_request, 'prompt', '')}")
        with self._tools_called_lock:
            self._tools_called = []
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("[high-level action] Cancel request received.")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Handles the incoming Prompt action (high-level). Breaks prompt into steps and dispatches them.
        """
        prompt_text = goal_handle.request.prompt
        self.get_logger().info(f"[high-level action] Executing prompt: {prompt_text}")

        feedback_msg = Prompt.Feedback()

        result_container: Dict[str, Any] = {"success": False, "final_response": "Internal error"}

        def run_agent_action():
            try:
                self.get_logger().info("High-level agent: breaking instruction into ordered steps...")
                self.response_pub.publish(String(data="Hey there! I'm thinking about how to handle your request..."))

                agent_resp = self.agent_executor.invoke({"input": prompt_text})
                final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)
                result_container["success"] = True
                result_container["final_response"] = final_text
                # Publish LLM response to /response
                self.response_pub.publish(String(data=f"Alright! Here's what I plan to do: {final_text}"))

                # Parse steps and dispatch
                steps = self._parse_steps_from_text(final_text)
                if not steps:
                    result_container["final_response"] += "\n[No steps parsed]"
                    self.response_pub.publish(String(data="Hmm... I couldn’t figure out any clear steps. Could you try rephrasing that?"))
                    return

                self.get_logger().info(f"Parsed {len(steps)} step(s). Dispatching to /medium_level one-by-one...")
                self.response_pub.publish(String(data=f"I’ve got {len(steps)} steps to do. Let’s get started!"))

                for i, step in enumerate(steps, start=1):
                    start_msg = f"Okay! Starting step {i}: {step}"
                    self.response_pub.publish(String(data=start_msg))

                    self.get_logger().info(f"Sending step {i}/{len(steps)} to medium_level: {step}")
                    # publish feedback with tools called snapshot
                    with self._tools_called_lock:
                        tools_snapshot = list(self._tools_called)
                    feedback_msg.tools_called = tools_snapshot
                    try:
                        goal_handle.publish_feedback(feedback_msg)
                    except Exception:
                        pass

                    # send step and wait
                    # send_future = self.medium_level_client.wait_for_server(timeout_sec=5.0)
                    send_result = self.send_step_to_medium_and_return_result_obj(step)
                    if send_result is None:
                        fail_msg = f"Oops! I couldn’t complete step {i}: {step}. I’ll stop here for now."
                        self.response_pub.publish(String(data=fail_msg))
                        result_container["final_response"] += f"\nStep {i} failed to start"
                        break
                    else:
                        if send_result.success:
                            done_msg = f"Nice! Step {i} is all done. Here’s what I got: {send_result.final_response}"
                        else:
                            done_msg = f"Alright, I tried step {i} but it didn’t quite work out. The system said: {send_result.final_response}"
                        self.response_pub.publish(String(data=done_msg))
                        result_container["final_response"] += f"\nStep {i} result: success={send_result.success}"

            except Exception as e:
                self.get_logger().error(f"Error during planning/dispatch: {e}")
                self.response_pub.publish(String(data="Uh oh, something went wrong while I was planning that."))
                result_container["success"] = False
                result_container["final_response"] = f"Agent error: {e}"

        agent_thread = threading.Thread(target=run_agent_action, daemon=True)
        agent_thread.start()

        # Publish periodic feedback while running
        while agent_thread.is_alive():
            with self._tools_called_lock:
                tools_snapshot = list(self._tools_called)
            feedback_msg.tools_called = tools_snapshot
            try:
                goal_handle.publish_feedback(feedback_msg)
            except Exception:
                pass
            time.sleep(0.5)

        # final feedback
        with self._tools_called_lock:
            tools_snapshot = list(self._tools_called)
        feedback_msg.tools_called = tools_snapshot
        try:
            goal_handle.publish_feedback(feedback_msg)
        except Exception:
            pass

        result_msg = Prompt.Result()
        result_msg.success = bool(result_container.get("success", False))
        result_msg.final_response = str(result_container.get("final_response", ""))

        goal_handle.succeed()
        self.get_logger().info(f"[high-level action] Goal finished. success={result_msg.success}")
        return result_msg

    # -----------------------
    # Helpers: send step and return result object
    # -----------------------
    def send_step_to_medium(self, step_text: str, timeout: float = 30.0) -> Optional[Prompt.Result]:
        """
        Synchronous helper: sends a step to the /medium_level Prompt action server and waits for the result.
        Returns the Prompt.Result object or None on failure/timeouts.
        """
        try:
            if not self.medium_level_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("/medium_level action server unavailable")
                return None
            goal = Prompt.Goal()
            goal.prompt = step_text
            send_future = self.medium_level_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Medium-level goal rejected")
                return None
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
            result = result_future.result().result
            return result
        except Exception as e:
            self.get_logger().error(f"Exception when sending to medium: {e}")
            return None

    def send_step_to_medium_and_return_result_obj(self, step_text: str) -> Optional[Prompt.Result]:
        """
        Same as send_step_to_medium but wraps errors and returns Prompt.Result or None.
        """
        return self.send_step_to_medium(step_text)


# -----------------------
# Entrypoint
# -----------------------
def main(args=None):
    rclpy.init(args=args)
    node = Ros2HighLevelAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Ros2 High-Level Agent Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
