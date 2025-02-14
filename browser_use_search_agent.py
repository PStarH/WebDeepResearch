import asyncio
import json
import re
from browser_use import Agent as BrowserAgentExecutor
from langchain_openai import ChatOpenAI

class PlanningAgent:
    def __init__(self, llm):
        self.llm = llm
        # Memory module: stores verified facts, pending verifications, contradictions, and route history.
        self.memory = {
            "verified_facts": [],
            "pending_verifications": [],
            "contradictions": [],
            "route_history": []
        }
        self.context = (
            "# Role: Senior Research Validator\n\n"
            "## Core Principles\n"
            "1. Strict validation: All key information must be verified by at least two independent, reliable sources.\n"
            "2. Handle contradictions: If conflicting data is found, trigger a third validation query.\n"
            "3. Credibility assessment: Prioritize authoritative sources (academic journals > institutional websites > Wikipedia > news > personal blogs).\n"
            "4. Format control: Follow the output format exactly. Numbers must not include units/symbols; strings must use full names.\n\n"
            "## Workflow\n"
            "1. Initial search: Gather a basic information framework.\n"
            "2. Validation Phase:\n"
            "   a) Cross-check sources: Compare key data points from different sources.\n"
            "   b) Freshness check: Prefer studies from the last 3 years.\n"
            "   c) Reverse validation: Trace back from conclusions to supporting evidence.\n"
            "3. Final Confirmation: All information must be:\n"
            "   - Verified by at least two independent sources\n"
            "   - Free of major contradictions (minor differences allowed)\n"
            "   - Detailed enough based on the original research\n\n"
            "## Output Format\n"
            "Strictly use the following format:\n"
            "Validation Conclusion: [verified/unverified]\n"
            "Pending Items: [...]\n"
            "Next Command: [next command]\n"
            "or\n"
            "FINAL ANSWER: [final answer in required format]\n\n"
            "Only output FINAL ANSWER when all conditions are met."
        )

    async def verify_result(self, browser_result: str) -> dict:
        """
        Validate the browser result and return structured JSON with:
         - credibility score (1-5)
         - list of missing info
         - list of contradictions
         - boolean flag for freshness check (recent data preferred)
         - a verification summary.
        """
        verify_prompt = (
            "## Browser Result\n"
            f"{browser_result}\n\n"
            "## Validation Task\n"
            "1. Score credibility (1-5).\n"
            "2. Identify any missing information.\n"
            "3. Detect any contradictions.\n"
            "4. Evaluate timeliness (prefer results from the last 3 years).\n\n"
            "Reply in the following JSON format:\n"
            '{\n'
            '  "credibility_score": int,\n'
            '  "missing_info": [str],\n'
            '  "contradictions": [str],\n'
            '  "needs_fresh_check": bool,\n'
            '  "verification_summary": str\n'
            '}'
        )
        verification = await self.llm.apredict(verify_prompt)
        try:
            return json.loads(verification)
        except Exception:
            return {"verification_summary": f"Failed to parse JSON. Response: {verification}"}

    async def dynamic_route_decision(self, verification: dict, browser_result: str) -> str:
        """
        Using short-term memory and the current verification details, decide which route to take.
        Options:
            - supplemental_validation: if credibility < 4.
            - third_party_validation: if contradictions exist.
            - query_missing_info: if missing info is present.
            - continue_search: if additional details are needed.
            - finalize_answer: if all conditions are met.
        The LLM is prompted to provide a decision along with a brief reasoning.
        """
        prompt = (
            "Current route history: " + str(self.memory.get("route_history", [])) + "\n"
            "Current verification details:\n"
            f"- Credibility Score: {verification.get('credibility_score', 'N/A')}\n"
            f"- Missing Info: {verification.get('missing_info', [])}\n"
            f"- Contradictions: {verification.get('contradictions', [])}\n\n"
            "Based on the above, choose the best route from the following options:\n"
            "1. supplemental_validation (if credibility < 4)\n"
            "2. third_party_validation (if contradictions exist)\n"
            "3. query_missing_info (if missing info is present)\n"
            "4. continue_search (if more data is needed)\n"
            "5. finalize_answer (if conditions are met: credibility >= 4, no contradictions, at least 2 verified facts)\n\n"
            "Respond in the format: \"ROUTE: <option>\" with a brief reasoning."
        )
        route_decision = await self.llm.apredict(prompt)
        match = re.search(r"ROUTE:\s*(\w+)", route_decision)
        if match:
            route = match.group(1).strip().lower()
        else:
            route = "continue_search"
        self.memory.setdefault("route_history", []).append(route)
        return route

    async def decide_next_step(self, browser_result: str) -> str:
        # Update memory with facts extracted from the browser result.
        await self._update_memory(browser_result)
        
        # Get verification details.
        verification = await self.verify_result(browser_result)
        
        # Use dynamic route decision based on verification and short-term memory.
        route = await self.dynamic_route_decision(verification, browser_result)
        
        # Decide next action based on the chosen route.
        if route == "finalize_answer":
            return self._generate_final_answer()
        elif route == "supplemental_validation":
            next_cmd = "Supplemental validation: Please perform an additional search to verify all key data."
        elif route == "third_party_validation":
            next_cmd = "Third-party validation: Please search for a trusted independent source to resolve the contradictions."
        elif route == "query_missing_info":
            missing = verification.get("missing_info", [])
            missing_str = ", ".join(missing) if missing else "unspecified details"
            next_cmd = f"Query missing info: Please search specifically for additional details on {missing_str}."
        else:  # continue_search or default
            next_cmd = "Continue search: Please refine the query to gather more comprehensive information."
        
        # Format and return the next command.
        decision_prompt = (
            "Current Verification Status:\n"
            f"Credibility Score: {verification.get('credibility_score', 'N/A')}\n"
            f"Missing Info: {verification.get('missing_info', [])}\n"
            f"Contradictions: {verification.get('contradictions', [])}\n\n"
            "Chosen Route: " + route + "\n\n"
            "Based on the above, the next command is:\n" + next_cmd
        )
        decision = await self.llm.apredict(decision_prompt)
        return self._format_output(decision)

    def _format_output(self, text: str) -> str:
        """Format the output: remove extraneous commas in numbers and ensure the proper structure."""
        if "FINAL ANSWER:" in text:
            text = re.sub(r"(\d),(\d)", r"\1\2", text)
        return text

    async def _update_memory(self, result: str):
        """
        Extract verifiable fact assertions from the result and update the memory.
        """
        extract_prompt = (
            "Extract all verifiable fact assertions from the following text. "
            "Return a JSON array where each element is formatted as:\n"
            '{"fact": str, "source": str, "year": int}\n\n'
            "Text:\n" + result + "\n"
        )
        facts = await self.llm.apredict(extract_prompt)
        try:
            extracted = json.loads(facts)
            for f in extracted:
                if f not in self.memory["verified_facts"]:
                    self.memory["verified_facts"].append(f)
        except Exception:
            pass

    def _compile_answer(self) -> str:
        """
        Compile the final answer by summarizing the verified facts.
        """
        facts_list = [
            f"Fact: {f.get('fact', '')} | Source: {f.get('source', '')} ({f.get('year', 'N/A')})"
            for f in self.memory.get("verified_facts", [])
        ]
        return " ; ".join(facts_list) if facts_list else "Insufficient verified information."

    def _generate_final_answer(self) -> str:
        """
        Generate the final structured answer based on the memory.
        """
        verified_sources = list({(f['source'], f['year']) for f in self.memory.get("verified_facts", [])})[:3]
        answer_template = (
            "\nComprehensive Verified Sources:\n"
            "{verified_sources}\n\n"
            "Final Conclusion: {final_answer}"
        )
        return answer_template.format(
            verified_sources="\n".join([f"{s[0]} ({s[1]})" for s in verified_sources]),
            final_answer=self._compile_answer()
        )

async def main():
    # Initialize two different LLM models
    planning_llm = ChatOpenAI(model="gpt-4o")  # More capable model for planning and validation
    browser_llm = ChatOpenAI(model="gpt-4o")  # Faster model for browser interactions
    
    planning_agent = PlanningAgent(planning_llm)
    
    current_command = (
        "Open the browser and search for: 'Animals mentioned in Ilias Lagkouvardos and Olga Tapia papers on the alvei species that also appear in the 2021 Wikipedia article about a multicenter, randomized, double-blind study.'"
    )
    overall_report = f"Initial Command: {current_command}\n"
    max_time = 300
    start_time = asyncio.get_event_loop().time()
    
    step = 1
    while (asyncio.get_event_loop().time() - start_time) < max_time:
        print(f"\n--- Step {step} ---")
        print(f"Browser Command: {current_command}")
        
        # Use browser_llm for BrowserAgentExecutor
        browser_agent = BrowserAgentExecutor(task=current_command, llm=browser_llm)
        browser_result = await browser_agent.run()
        print(f"Browser Result:\n{browser_result}\n")
        
        overall_report += f"\nStep {step} Command: {current_command}\nResult:\n{browser_result}\n"
        
        next_instruction = await planning_agent.decide_next_step(browser_result)
        print(f"Planning Agent Instruction:\n{next_instruction}\n")
        
        if next_instruction.upper().startswith("FINAL ANSWER:"):
            overall_report += "\n" + next_instruction
            print("Research complete. Final answer generated.")
            break
        else:
            current_command = next_instruction
        
        step += 1
    else:
        overall_report += "\nResearch terminated due to time limit."
        print("Research terminated due to time limit.")
    
    print("\n--- Final Combined Report ---")
    print(overall_report)

if __name__ == "__main__":
    asyncio.run(main())
