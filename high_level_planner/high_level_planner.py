import os
import threading
import time
from typing import List, Dict, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.task import Future as RclpyFuture

from std_srvs.srv import SetBool
from std_msgs.msg import String
from geometry_msgs.msg import Pose

# Action used for inter-level communication (re-using your Prompt action)
from custom_interfaces.action import Prompt

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
        # self.transcript_sub = self.create_subscription(String, "/transcript", self.transcript_callback, 10)
        self._last_transcript_lock = threading.Lock()
        self._last_transcript: Optional[str] = None

        # Action client to medium-level planner (send one step at a time)
        self.medium_level_client = ActionClient(self, Prompt, "/medium_level")

        # For diagnostics & optional low-level services
        # (we declare vision service clients — replace types with your actual types)
        # These are placeholders for your vision node; implement/replace with actual srv/action types.
        self.vision_detect_client = self.create_client(SetBool, "/vision/detect_objects")  # placeholder
        self.vision_segment_client = self.create_client(SetBool, "/vision/segment_object")  # placeholder
        self.vision_classify_client = self.create_client(SetBool, "/vision/classify_region")  # placeholder
        self.vision_depth_client = self.create_client(SetBool, "/vision/get_depth_at_pixel")  # placeholder

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

        # Vision: detect_objects (calls vision node)
        @tool
        def detect_objects(image_hint: Optional[str] = "") -> str:
            """
            Ask visual node to detect objects in the scene. 'image_hint' can help (e.g., "top view" or "rgb").
            Returns a textual list of detected objects or an error message.
            """
            tool_name = "detect_objects"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                if not self.vision_detect_client.wait_for_service(timeout_sec=2.0):
                    return "Vision detect service unavailable"
                request = SetBool.Request()
                # We use SetBool as placeholder: put hint in 'data' via True/False is not suitable — this is a stub.
                # Replace this block with the actual service type and fields.
                request.data = True
                future = self.vision_detect_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                resp = future.result()
                return f"detect_objects response: {getattr(resp, 'message', str(resp))}"
            except Exception as e:
                return f"ERROR in detect_objects: {e}"

        tools.append(detect_objects)

        @tool
        def segment_object(object_name: str) -> str:
            """
            Request segmentation of a specific object (by name or hint). Returns segmentation result summary.
            """
            tool_name = "segment_object"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                if not self.vision_segment_client.wait_for_service(timeout_sec=2.0):
                    return "Vision segment service unavailable"
                request = SetBool.Request()
                request.data = True
                future = self.vision_segment_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                resp = future.result()
                return f"segment_object response: {getattr(resp, 'message', str(resp))}"
            except Exception as e:
                return f"ERROR in segment_object: {e}"

        tools.append(segment_object)

        @tool
        def classify_region(region_hint: str) -> str:
            """
            Ask the vision node to classify a region or crop (e.g., 'red cup on left').
            """
            tool_name = "classify_region"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_classify_client.wait_for_service(timeout_sec=2.0):
                    return "Vision classify service unavailable"
                request = SetBool.Request()
                request.data = True
                future = self.vision_classify_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                resp = future.result()
                return f"classify_region response: {getattr(resp, 'message', str(resp))}"
            except Exception as e:
                return f"ERROR in classify_region: {e}"

        tools.append(classify_region)

        @tool
        def get_depth_at_pixel(x: int, y: int) -> str:
            """
            Query depth camera for depth at pixel (x,y). Returns depth in meters or an error string.
            """
            tool_name = "get_depth_at_pixel"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_depth_client.wait_for_service(timeout_sec=2.0):
                    return "Depth service unavailable"
                request = SetBool.Request()
                request.data = True
                future = self.vision_depth_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                resp = future.result()
                return f"depth_at_pixel response: {getattr(resp, 'message', str(resp))}"
            except Exception as e:
                return f"ERROR in get_depth_at_pixel: {e}"

        tools.append(get_depth_at_pixel)

        # Tool to dispatch a step to medium-level planner, waits for completion
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
            "capabilities (detect_objects, segment_object, classify_region, get_depth_at_pixel) and a "
            "tool send_to_medium_level which sends a single step to the medium-level planner (/medium_level). "
            "Your job: given a natural-language instruction, produce a short ordered list of actionable steps "
            "that a medium-level planner can execute. Keep steps concise, ambiguous-free and in the form "
            "'Action: <verb> <object/pose/params>'. When appropriate you may call vision tools to inspect the scene. "
            "Return the final step list as the agent output."
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
                    send_future = self.medium_level_client.wait_for_server(timeout_sec=5.0)
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
