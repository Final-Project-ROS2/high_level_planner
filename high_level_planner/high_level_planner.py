#!/usr/bin/env python3
"""
ros2_high_level_agent_with_vision.py

High-level ROS2 LLM agent with integrated vision tools as LangChain @tool wrappers.

Vision services exposed to the agent:
- /vision/detect_objects    -> custom_interfaces.srv.DetectObjects
- /vision/classify_all      -> std_srvs.srv.Trigger
- /vision/classify_bb       -> custom_interfaces.srv.ClassifyBBox
- /vision/detect_grasp      -> custom_interfaces.srv.DetectGrasps
- /vision/detect_grasp_bb   -> custom_interfaces.srv.DetectGraspBBox
- /vision/understand_scene  -> custom_interfaces.srv.UnderstandScene

"""
import contextlib
import io
import sys

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
from custom_interfaces.srv import (
    DetectObjects,
    ClassifyBBox,
    DetectGrasps,
    DetectGraspBBox,
    UnderstandScene,
)

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama


from dotenv import load_dotenv

import re
import json

import time

def clean_agent_text(text: str) -> str:
    """Clean and humanize raw AI agent log text for TTS."""
    if not text:
        return ""

    # 1. Remove ANSI escape codes (color/style control chars)
    text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)

    # 2. Remove retry/error messages or technical logs
    technical_patterns = [
        r"Retrying .* in \d+(\.\d+)? seconds.*",
        r"> Entering new AgentExecutor chain.*",
        r"> Finished chain.*",
        r"langchain.*",
        r"InternalServe.*",
    ]
    for pattern in technical_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = text.strip()
    if not text:
        return ""

    # 3. Handle structured invocation logs
    # Example: Invoking: `send_to_medium_level` with `{'step_text': 'open the gripper'}`
    invoke_match = re.search(
        r"Invoking:\s*`send_to_medium_level`\s*with\s*`({.*})`", text)
    if invoke_match:
        try:
            params_str = invoke_match.group(1)
            # fix malformed single quotes and normalize JSON
            params_str = params_str.replace("''", '"').replace("'", '"')
            params = json.loads(params_str)
            step_text = params.get("step_text", "").strip()
            if step_text:
                # Remove "Action:" prefix if present
                step_text = re.sub(r'^[Aa]ction:\s*', '', step_text)
                # Remove wrapping quotes if any remain
                step_text = step_text.strip(" '\"")
                return f"I'm going to {step_text}."
        except Exception:
            pass
        return ""  # if malformed or unknown, drop it


    # 4. Filter out medium_level result lines
    # e.g. medium_level result: success=True, response=The gripper is now open.
    if text.lower().startswith("medium_level result:"):
        return ""

    # 5. Convert JSON/dict-like text to human-friendly sentences
    try:
        if text.strip().startswith("{") and text.strip().endswith("}"):
            data = json.loads(text)
            parts = [f"{k.replace('_', ' ')} is {v}" for k, v in data.items()]
            text = ". ".join(parts)
    except Exception:
        pass

    # 6. Remove stray escape sequences, quotes, and collapse whitespace
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" '\"")

    # 7. Filter short/meaningless fragments
    if len(text) < 3 or text.lower() in ["none", "null"]:
        return ""

    return text.strip()



ENV_PATH = '/home/group11/final_project_ws/src/high_level_planner/.env'
load_dotenv(dotenv_path=ENV_PATH)

class ROSLogPublisher(io.TextIOBase):
    def __init__(self, publisher):
        self.publisher = publisher

    def write(self, text):
        text = clean_agent_text(text)
        if text:
            self.publisher.publish(String(data=text))
        return len(text)

    def flush(self):
        pass


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

        # Initialization flag: will be set to True after scene description is obtained
        self.initialized = False
        self.scene_description: Optional[str] = None
        self._init_lock = threading.Lock()

        self.declare_parameter("real_hardware", False)
        self.real_hardware: bool = self.get_parameter("real_hardware").get_parameter_value().bool_value
        self.declare_parameter("use_ollama", False)
        self.use_ollama: bool = self.get_parameter("use_ollama").get_parameter_value().bool_value
        self.declare_parameter("confirm", True)
        self.confirm: bool = self.get_parameter("confirm").get_parameter_value().bool_value
        self.declare_parameter("format_response", False)
        self.format_response: bool = self.get_parameter("format_response").get_parameter_value().bool_value
        self.declare_parameter("ollama_model", "qwen3:8b")
        self.ollama_model: str = self.get_parameter("ollama_model").get_parameter_value().string_value

        # -----------------------------
        # LLM Selection: Gemini or Ollama
        # -----------------------------
        if self.use_ollama:
            self.get_logger().info("Using local LLM via Ollama.")
            # Example: using llama3.1 or any model installed in `ollama list`
            self.llm = ChatOllama(
                model=self.ollama_model,   # <--- change to any local model you want
                temperature=0.0
            )
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self.get_logger().warn("No LLM API key found in environment variables GEMINI_API_KEY.")
            self.get_logger().info("Using Google Gemini API LLM.")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.0,
            )


        # Subscribe to transcript topic (MUST)
        self.transcript_sub = self.create_subscription(String, "/transcript", self.transcript_callback, 10)
        self._last_transcript_lock = threading.Lock()
        self._last_transcript: Optional[str] = None

        # Action client to medium-level planner (send one step at a time)
        self.medium_level_client = ActionClient(self, Prompt, "/medium_level")

        # Vision clients
        self.vision_vqa_client = ActionClient(self, Prompt, "/vqa")

        # Track tools called (for feedback)
        self._tools_called: List[str] = []
        self._tools_called_lock = threading.Lock()

        # Initialize LangChain tools (vision wrappers, medium-level submitter, etc.)
        self.tools = self._initialize_tools()

        # Create LangChain agent after scene initialization
        self.agent_executor: Optional[AgentExecutor] = None

        # Chat history
        self.chat_history: List[Dict[str, str]] = []  # [{'role': 'user', 'content': ...}, {'role': 'ai', 'content': ...}]
        self.latest_plan: Optional[List[str]] = None

        # Create a new service for confirmation
        self.confirm_srv = self.create_service(Trigger, "/confirm", self.confirm_service_callback)

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
        self.benchmark_pub = self.create_publisher(String, "/benchmark_logs", 10)

        self.get_logger().info("Ros2 High-Level Agent Node initialized. Fetching scene description...")
        
        # Start scene description initialization in background thread
        init_thread = threading.Thread(target=self._initialize_scene_description, daemon=False)
        init_thread.start()

        self.get_logger().info("Ros2 High-Level Agent Node ready (waiting for scene description before accepting requests).")

    def _benchmark_log(self, label: str):
        t = self.get_clock().now()
        t_sec = t.nanoseconds * 1e-9
        self.benchmark_pub.publish(
            String(data=f"{label},{t_sec:.9f}")
        )

    def _initialize_scene_description(self):
        """
        Fetch the initial scene description from /vqa action.
        Sets self.initialized to True once complete.
        """
        try:
            self.get_logger().info("Waiting for /vqa action server...")
            if not self.vision_vqa_client.wait_for_server(timeout_sec=30.0):
                self.get_logger().error("/vqa action server unavailable after 30 seconds. Node will not initialize.")
                with self._init_lock:
                    self.scene_description = "Scene not available"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return
            
            self.get_logger().info("Calling /vqa to describe the scene...")
            goal = Prompt.Goal()
            goal.prompt = "Describe the scene, including what object exists and how many of each object"

            goal_event = threading.Event()
            result_event = threading.Event()
            goal_handle_container = [None]
            result_container = [None]

            def goal_response_callback(future):
                goal_handle_container[0] = future.result()
                goal_event.set()

            def result_callback(future):
                result_container[0] = future.result()
                result_event.set()

            send_future = self.vision_vqa_client.send_goal_async(goal)
            send_future.add_done_callback(goal_response_callback)

            if not goal_event.wait(timeout=30.0):
                self.get_logger().error("Timeout waiting for VQA goal acceptance")
                with self._init_lock:
                    self.scene_description = "Scene description unavailable"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return

            goal_handle = goal_handle_container[0]
            if not goal_handle.accepted:
                self.get_logger().error("VQA goal rejected during scene initialization")
                with self._init_lock:
                    self.scene_description = "Scene description unavailable"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return

            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(result_callback)

            if not result_event.wait(timeout=120.0):
                self.get_logger().error("Timeout waiting for VQA result")
                with self._init_lock:
                    self.scene_description = "Scene description unavailable"
                    scene_desc = self.scene_description
                self.agent_executor = self._create_agent_executor(scene_desc=scene_desc)
                with self._init_lock:
                    self.initialized = True
                return

            result = result_container[0].result
            scene_response = None
            if result is not None:
                scene_response = getattr(result, "final_response", None) or str(result)

            with self._init_lock:
                self.scene_description = scene_response if scene_response else "Scene description unavailable"
                scene_desc = self.scene_description
            self.agent_executor = self._create_agent_executor(scene_desc=scene_desc)
            with self._init_lock:
                self.initialized = True

            self.get_logger().info(f"Scene description obtained: {self.scene_description}")
            self.response_pub.publish(String(data=f"Scene analysis: {self.scene_description}"))
            
        except Exception as e:
            self.get_logger().error(f"Exception during scene initialization: {e}")
            with self._init_lock:
                self.scene_description = "Scene description unavailable"
                scene_desc = self.scene_description
            self.agent_executor = self._create_agent_executor(scene_desc=scene_desc)
            with self._init_lock:
                self.initialized = True

    # -----------------------
    # Transcript handling
    # -----------------------
    def transcript_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        if not self.initialized:
            self.get_logger().warn("Node not fully initialized yet. Ignoring transcript.")
            self.response_pub.publish(String(data="I'm still initializing. Please wait a moment."))
            return

        start_time = time.perf_counter()
        self._benchmark_log("transcript_received")

        with self._last_transcript_lock:
            self._last_transcript = text

        self.get_logger().info(f"Received transcript: {text}")

        plan_thread = threading.Thread(
            target=self._plan_and_dispatch_from_transcript,
            args=(text, start_time),
            daemon=True
        )
        plan_thread.start()

    def _generate_plan(self, instruction_text: str, start_time: Optional[float] = None) -> List[str]:
        """
        Generate a plan (list of steps) from the user's instruction but do NOT execute.
        The plan is stored internally for later confirmation.
        """
        # Add user message to chat history
        self.chat_history.append({"role": "user", "content": instruction_text})

        try:
            self.get_logger().info("High-level agent: thinking and generating plan...")
            self.response_pub.publish(String(data="Got it! Let me think through that..."))
        
            langchain_history = []
            for msg in self.chat_history[:-1]:  # Exclude the current message we just added
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))
            # Invoke agent with chat history
            agent_resp = self.agent_executor.invoke({
                "input": instruction_text,
                "chat_history": langchain_history
            })
        
            final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)

            # Add AI response to chat history
            self.chat_history.append({"role": "assistant", "content": final_text})

            # Parse steps
            steps = self._parse_steps_from_text(final_text)
            self.latest_plan = steps

            if start_time is not None:
                end_time = time.perf_counter()
                self._benchmark_log("plan_generated")
                self.benchmark_pub.publish(String(data=f"Plan generated in: ,{end_time - start_time:.2f}"))

            if not steps:
                msg = "Hmm... I couldn't figure out any clear steps. Could you try rephrasing that?"
                self.response_pub.publish(String(data=msg))
                return []

            # Present plan to user for confirmation
            if self.format_response:
                readable_plan = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)])
            else:
                readable_plan = final_text
            self.response_pub.publish(String(
                data=f"Here's what I plan to do:\n{readable_plan}\n\nPlease review and confirm if this looks good!"
            ))
            if self.confirm:
                self.get_logger().info(f"Generated plan with {len(steps)} steps, waiting for /confirm.")
            else:
                # Execute the plan in a separate thread to avoid blocking the service callback
                def execute_plan():
                    self.response_pub.publish(String(data="Got it! Executing your approved plan now..."))
                    self.get_logger().info("Executing confirmed plan...")

                    for i, step in enumerate(self.latest_plan, start=1):
                        self.response_pub.publish(String(data=f"Starting step {i}: {step}"))
                        result = self.send_step_to_medium_async(step)

                        if result is None or not result.success:
                            msg = f"Step {i} failed: {step}. Stopping execution."
                            self.response_pub.publish(String(data=msg))
                            self.get_logger().error(msg)
                            break
                        else:
                            done_msg = f"Step {i} completed successfully."
                            self.response_pub.publish(String(data=done_msg))
                            self.get_logger().info(done_msg)

                    end_time = time.perf_counter()
                    benchmark_info = f"High-level action completed in {end_time - self.start_time:.2f} seconds"
                    self.benchmark_pub.publish(String(data=benchmark_info))
                    self.response_pub.publish(String(data="Plan execution finished."))
                    self.get_logger().info("All steps done. Clearing chat history and plan.")
                    self.chat_history.clear()
                    self.latest_plan.clear()

                # Start execution in background thread
                execution_thread = threading.Thread(target=execute_plan, daemon=True)
                execution_thread.start()

            return steps
        except Exception as e:
            self.get_logger().error(f"Error generating plan: {e}")
            self.response_pub.publish(String(data="Sorry, something went wrong while planning."))
            return []

    def _plan_and_dispatch_from_transcript(self, instruction_text: str, start_time: float):
        self._generate_plan(instruction_text, start_time=start_time)

    def confirm_service_callback(self, request, response):
        """
        When the user confirms, execute the latest plan step-by-step.
        Clears chat history and latest plan after execution.
        """
        if not self.initialized:
            response.success = False
            response.message = "Node not yet initialized. Please wait for scene analysis to complete."
            self.response_pub.publish(String(data=response.message))
            return response
        
        if not self.latest_plan:
            response.success = False
            response.message = "No plan to confirm. Please give a new instruction first."
            self.response_pub.publish(String(data=response.message))
            return response

        # Execute the plan in a separate thread to avoid blocking the service callback
        def execute_plan():
            self.response_pub.publish(String(data="Got it! Executing your approved plan now..."))
            self.get_logger().info("Executing confirmed plan...")

            for i, step in enumerate(self.latest_plan, start=1):
                self.response_pub.publish(String(data=f"Starting step {i}: {step}"))
                result = self.send_step_to_medium_async(step)

                if result is None or not result.success:
                    msg = f"Step {i} failed: {step}. Stopping execution."
                    self.response_pub.publish(String(data=msg))
                    self.get_logger().error(msg)
                    break
                else:
                    done_msg = f"Step {i} completed successfully."
                    self.response_pub.publish(String(data=done_msg))
                    self.get_logger().info(done_msg)

            self.response_pub.publish(String(data="Plan execution finished."))
            self.get_logger().info("All steps done. Clearing chat history and plan.")
            self.chat_history.clear()
            self.latest_plan.clear()

        # Start execution in background thread
        execution_thread = threading.Thread(target=execute_plan, daemon=True)
        execution_thread.start()

        # Return immediately from service call
        response.success = True
        response.message = "Plan execution started."
        return response

    def _parse_steps_from_text(self, text: str) -> List[str]:
        """
        Very simple step parser:
        - Look for lines starting with numbers (1., 1), '-', or 'Step X:' and collect them
        - Otherwise, split by newline and treat each line as a candidate step, filtering short lines.
        You can replace this with a more robust parser if needed.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        steps = []

        # First, capture explicit "Action:" patterns anywhere in the text so we handle
        # responses like "Action: move to ready" or multiple actions in a single message.
        action_matches = re.findall(r"Action:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
        for match in action_matches:
            cleaned = match.strip().strip("- .")
            if len(cleaned) > 3:
                steps.append(cleaned)

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
        def vqa(question: str) -> str:
            """
            Call /vqa which returns an answer to a visual question.
            """
            tool_name = "vqa"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                if not self.vision_vqa_client.wait_for_server(timeout_sec=5.0):
                    self.get_logger().error("/vqa action server unavailable")
                    return None
                goal = Prompt.Goal()
                goal.prompt = question
                send_future = self.vision_vqa_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, send_future)
                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    self.get_logger().error("VQA goal rejected")
                    return None
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                result = result_future.result().result
                return result
            except Exception as e:
                self.get_logger().error(f"Exception when sending to VQA: {e}")
                return None

        tools.append(vqa)

        return tools

    # -----------------------
    # Create agent executor
    # -----------------------
    def _create_agent_executor(self, scene_desc: Optional[str] = None) -> AgentExecutor:
        if scene_desc is None:
            with self._init_lock:
                scene_desc = self.scene_description if self.scene_description else "Scene not yet analyzed."
        
        system_message = (
            "You are a High-Level ROS2 planning assistant for a Robotic Arm."
            "Your job: given a natural-language instruction, produce a short ordered list of actionable steps "
            "that a medium-level planner can execute. Keep steps concise, unambiguous and in the form "
            "'Action: <verb> <object/pose/params>'. "
            "The robot has 3 setpoints: 'home', 'ready', and 'handover'. Use these names when referring to them. "
            "The medium-level planner can handle commands like 'move to <setpoint>', 'move <direction>', 'move to <object>, 'pick up <object>', 'place at <location>', "
            "Add context to your steps by referencing the scene when relevant, for example 'pick up the second screwdriver from the left'"
            # "**ALWAYS** produce steps that can be executed by the medium-level planner. "
            "You have access to vision tools like 'vqa' to inspect the scene. You can ask visual questions to gather information about the environment."
            "Use 'vqa' to IDENTIFY object when the user doesn't explicitly give you an object name, for example: If the user asks to pickup the left most object, use 'vqa' to ask 'Which object is the left most?' to get the name of the object"
            "Then send the result to the medium-level planner to execute the action.\n"
            "If the user explicitly gives you an object name, you can directly use it in your steps without calling 'vqa'. For example, if the user says 'pick up the red cup', you can directly generate a step 'Action: pick up the red cup' without using 'vqa'."
            "You can add modifiers to the object name, like screwdriver_leftmost, so you DO NOT need to ask clarifying questions"
            f"Current scene description: {scene_desc}"
            "If the instruction specify an existing object, no need to use 'vqa'."
            "If the instruction is unclear, RESPONSE with a clarifying questions before proceeding."
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
        if not self.initialized:
            self.get_logger().warn("[high-level action] Goal received but node not initialized yet.")
            return GoalResponse.REJECT
        self.get_logger().info(f"[high-level action] Received goal: {getattr(goal_request, 'prompt', '')}")
        with self._tools_called_lock:
            self._tools_called = []
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("[high-level action] Cancel request received.")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        start_time = time.perf_counter()
        self._benchmark_log("action_goal_received")

        prompt_text = goal_handle.request.prompt
        self.get_logger().info(f"[high-level action] Executing prompt: {prompt_text}")

        feedback_msg = Prompt.Feedback()

        result_container: Dict[str, Any] = {"success": True, "final_response": "Internal error"}

        def run_agent_action():
                goal_text = goal_handle.request.prompt.strip()
                if not goal_text:
                    goal_handle.abort()
                    return Prompt.Result(success=False, final_response="Empty prompt")

                self.get_logger().info(f"High-level Prompt action received: {goal_text}")
                # self.response_pub.publish(String(data=f"You said: {goal_text}"))

                # Generate plan but do not execute
                steps = self._generate_plan(goal_text, start_time=start_time)
                if not steps:
                    goal_handle.abort()
                    return Prompt.Result(success=False, final_response="Failed to generate plan")

                if self.confirm:
                    # Wait for confirmation
                    msg = f"Generated {len(steps)} step(s). Please review and confirm via /confirm to execute."

                # self.response_pub.publish(String(data=msg))
                return Prompt.Result(success=True, final_response=msg)


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

    def send_step_to_medium_async(self, step_text: str, timeout: float = 120.0) -> Optional[Prompt.Result]:
        """
        Thread-safe version that uses threading.Event instead of spin_until_future_complete.
        """
        try:
            if not self.medium_level_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("/medium_level action server unavailable")
                return None
            
            goal = Prompt.Goal()
            goal.prompt = step_text
            
            # Use events to wait for async operations
            goal_event = threading.Event()
            result_event = threading.Event()
            goal_handle_container = [None]
            result_container = [None]
            
            # Callback for goal response
            def goal_response_callback(future):
                goal_handle_container[0] = future.result()
                goal_event.set()
            
            # Callback for result
            def result_callback(future):
                result_container[0] = future.result()
                result_event.set()
            
            # Send goal
            send_future = self.medium_level_client.send_goal_async(goal)
            send_future.add_done_callback(goal_response_callback)
            
            # Wait for goal acceptance
            if not goal_event.wait(timeout=5.0):
                self.get_logger().error("Timeout waiting for goal acceptance")
                return None
            
            goal_handle = goal_handle_container[0]
            if not goal_handle.accepted:
                self.get_logger().error("Medium-level goal rejected")
                return None
            
            # Get result
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(result_callback)
            
            # Wait for result
            if not result_event.wait(timeout=timeout):
                self.get_logger().error("Timeout waiting for result")
                return None
            
            result = result_container[0].result
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
