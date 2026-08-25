import os
import json
import ast

def get_system_prompt_str(classifier):
    filepath = f"spp_15052026/data_annotation/stage_3_subs/{classifier}/{classifier}_classifier.py"
    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, (ast.Name)) and target.id.startswith("SYSTEM_PROMPT_"):
                        if isinstance(node.value, ast.Constant):
                            return node.value.value
                        elif isinstance(node.value, ast.Str): # for older python
                            return node.value.s
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return ""

classifiers = ["themes", "connotation", "approach", "context"]
num_reviews = 13000
input_price_per_1m = 3.00
output_price_per_1m = 15.00
char_to_token = 0.25

report_content = "# Cost Breakdown for 13,000 Reviews Analysis\n\n"
grand_base_total = 0.0
grand_total_10_retry = 0.0

classifier_results = []

for cls in classifiers:
    analysis_path = f"spp_15052026/data_annotation/stage_3_subs/{cls}/cost_analysis_b1/length_analysis.json"
    with open(analysis_path, "r") as f:
        data = json.load(f)
    
    # These fields are nested inside "cost_estimate"
    cost_est = data["cost_estimate"]
    avg_sent_chars_review = cost_est["avg_sent_chars_review"]
    est_input_tok_per_call = cost_est["est_input_tok_per_call"]
    est_output_tok_per_call = cost_est["est_output_tok_per_call"]
    
    prompt_str = get_system_prompt_str(cls)
    prompt_tokens = round(len(prompt_str) * char_to_token)
    review_tokens = round(avg_sent_chars_review * char_to_token)
    
    # ensure prompt + review doesn't exceed total input
    if (prompt_tokens + review_tokens) > est_input_tok_per_call:
         prompt_tokens = est_input_tok_per_call - review_tokens
         if prompt_tokens < 0:
             prompt_tokens = 0
             review_tokens = est_input_tok_per_call

    message_tokens = est_input_tok_per_call - prompt_tokens - review_tokens
    
    total_input_tokens = est_input_tok_per_call
    total_output_tokens = est_output_tok_per_call
    
    def calc_cost(tokens, rate):
        return (tokens * num_reviews * rate) / 1000000

    c_prompt = calc_cost(prompt_tokens, input_price_per_1m)
    c_message = calc_cost(message_tokens, input_price_per_1m)
    c_review = calc_cost(review_tokens, input_price_per_1m)
    c_input_total = calc_cost(total_input_tokens, input_price_per_1m)
    c_output = calc_cost(total_output_tokens, output_price_per_1m)
    
    base_total = c_input_total + c_output
    retry_10 = base_total * 0.10
    total_with_10 = base_total + retry_10
    
    grand_base_total += base_total
    grand_total_10_retry += total_with_10
    
    classifier_results.append({
        "name": cls,
        "base": base_total,
        "total_10": total_with_10
    })

    report_content += f"## {cls.capitalize()} Classifier\n"
    report_content += "| Component | Tokens per Call | Total Tokens (13k) | Cost ($) |\n"
    report_content += "| :--- | :--- | :--- | :--- |\n"
    report_content += f"| Prompt input | {int(prompt_tokens)} | {int(prompt_tokens)*num_reviews:,} | {c_prompt:.2f} |\n"
    report_content += f"| Message input | {int(message_tokens)} | {int(message_tokens)*num_reviews:,} | {c_message:.2f} |\n"
    report_content += f"| Review input | {int(review_tokens)} | {int(review_tokens)*num_reviews:,} | {c_review:.2f} |\n"
    report_content += f"| **Total Input** | **{int(total_input_tokens)}** | **{int(total_input_tokens)*num_reviews:,}** | **{c_input_total:.2f}** |\n"
    report_content += f"| Output | {int(total_output_tokens)} | {int(total_output_tokens)*num_reviews:,} | {c_output:.2f} |\n"
    report_content += f"| **Base Total** | - | - | **{base_total:.2f}** |\n"
    report_content += f"| Retries (10%) | - | - | {retry_10:.2f} |\n\n"
    
    report_content += "### Retry Sensitivity\n"
    report_content += "| Retry Rate | Addl. Cost ($) | Total Cost ($) |\n"
    report_content += "| :--- | :--- | :--- |\n"
    for rate in [0.05, 0.10, 0.20]:
        addl = base_total * rate
        report_content += f"| {int(rate*100)}% | {addl:.2f} | {base_total + addl:.2f} |\n"
    report_content += "\n"

report_content += "## Grand Totals\n"
report_content += f"| Metric | Cost ($) |\n"
report_content += "| :--- | :--- |\n"
report_content += f"| Total Base Cost | **{grand_base_total:.2f}** |\n"
report_content += f"| Total with 10% Retries | **{grand_total_10_retry:.2f}** |\n\n"

report_content += "## Assumptions\n"
report_content += "- Character to token conversion factor: 0.25.\n"
report_content += f"- Input pricing: ${input_price_per_1m} per 1M tokens.\n"
report_content += f"- Output pricing: ${output_price_per_1m} per 1M tokens.\n"
report_content += "- Retries are modeled as additional full calls (input + output).\n"
report_content += "- Batch size = 1.\n"

output_path = "spp_15052026/data_annotation/stage_3_subs/cost_breakdown_13000_reviews.md"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    f.write(report_content)

print(f"Output path: {output_path}")
for res in classifier_results:
    print(f"{res['name']}: base={res['base']:.2f}, total (10% retry)={res['total_10']:.2f}")
print(f"Grand base: {grand_base_total:.2f}")
print(f"Grand with 10% retry: {grand_total_10_retry:.2f}")
