import os
import json
import ast

def get_system_prompt_len(classifier):
    filepath = f"spp_15052026/data_annotation/stage_3_subs/{classifier}/{classifier}_classifier.py"
    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("SYSTEM_PROMPT_"):
                        return len(node.value.value)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return 0

classifiers = ["themes", "connotation", "approach", "context"]
num_reviews = 13000
input_price_per_1m = 3.00
output_price_per_1m = 15.00
char_to_token = 0.25

results = {}
grand_base_total = 0

report = "# Cost Breakdown Report (13,000 Reviews per Classifier)\n\n"

for cls in classifiers:
    json_path = f"spp_15052026/data_annotation/stage_3_subs/{cls}/cost_analysis_b1/length_analysis.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    
    cost_est = data["cost_estimate"]
    avg_sent_chars = cost_est["avg_sent_chars_review"]
    est_input_tok = cost_est["est_input_tok_per_call"]
    est_output_tok = cost_est["est_output_tok_per_call"]
    
    sys_prompt_len = get_system_prompt_len(cls)
    
    prompt_tokens = sys_prompt_len * char_to_token
    review_tokens = avg_sent_chars * char_to_token
    wrapper_tokens = max(0, est_input_tok - prompt_tokens - review_tokens)
    
    input_cost = (est_input_tok * num_reviews / 1_000_000) * input_price_per_1m
    output_cost = (est_output_tok * num_reviews / 1_000_000) * output_price_per_1m
    base_cost = input_cost + output_cost
    grand_base_total += base_cost
    
    results[cls] = {
        "prompt_tokens": prompt_tokens,
        "review_tokens": review_tokens,
        "wrapper_tokens": wrapper_tokens,
        "input_tokens": est_input_tok,
        "output_tokens": est_output_tok,
        "base_cost": base_cost
    }
    
    report += f"## Classifier: {cls.capitalize()}\n"
    report += "| Component | Tokens per Call | Total Tokens (13k) | Cost ($) |\n"
    report += "| :--- | :--- | :--- | :--- |\n"
    report += f"| System Prompt | {prompt_tokens:.0f} | {prompt_tokens*num_reviews:,.0f} | - |\n"
    report += f"| Review Content | {review_tokens:.0f} | {review_tokens*num_reviews:,.0f} | - |\n"
    report += f"| Message Wrapper | {wrapper_tokens:.0f} | {wrapper_tokens*num_reviews:,.0f} | - |\n"
    report += f"| **Total Input** | **{est_input_tok:.0f}** | **{est_input_tok*num_reviews:,.0f}** | **{input_cost:.2f}** |\n"
    report += f"| **Total Output** | **{est_output_tok:.0f}** | **{est_output_tok*num_reviews:,.0f}** | **{output_cost:.2f}** |\n"
    report += f"| **Base Total** | | | **{base_cost:.2f}** |\n"
    report += f"| Retry (10%) | | | {base_cost*0.10:.2f} |\n\n"
    
    report += "### Retry Sensitivity\n"
    report += "| Retry Rate | Additional Cost ($) | Total with Retry ($) |\n"
    report += "| :--- | :--- | :--- |\n"
    for rate in [0.05, 0.10, 0.20]:
        report += f"| {rate*100:.0f}% | {base_cost*rate:.2f} | {base_cost*(1+rate):.2f} |\n"
    report += "\n"

grand_retry_total = grand_base_total * 1.10
report += "## Grand Totals (All 4 Classifiers)\n"
report += f"- **Base Total cost:** ${grand_base_total:.2f}\n"
report += f"- **Total with 10% Retry:** ${grand_retry_total:.2f}\n"

output_path = "spp_15052026/data_annotation/stage_3_subs/cost_breakdown_13000_reviews.md"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(report)

print(f"Report saved to: {output_path}")
print(f"Grand Base Total: {grand_base_total:.2f}")
print(f"Grand Total (10% retry): {grand_retry_total:.2f}")
